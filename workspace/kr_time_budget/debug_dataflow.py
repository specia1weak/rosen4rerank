"""用一个真实会话逐段调试 KR 数据流，默认只读、不训练。

直接运行（工作目录不限）：
    uv run python workspace/kr_time_budget/debug_dataflow.py
命令行断点：
    uv run python workspace/kr_time_budget/debug_dataflow.py --stop-at batch
    uv run python workspace/kr_time_budget/debug_dataflow.py --pdb
IDE：在标记 DEBUG 1～7 的下一行设置断点，再 Debug 当前文件。

所有中间表和 tensor 留在 main 的局部变量中；使用 step into 可以进入正式实现。
默认重算聚合与过去日先验，与落盘结果核对；--skip-priors 可跳过全体曝光的先验扫描。
"""
import argparse
from pathlib import Path
import sys

# 支持在 IDE 中直接执行本文件，不依赖 IDE 的 working directory/PYTHONPATH。
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from rosen.datasets.kuairand import KuaiRandDataset
from workspace.kr_time_budget.pipeline.build_requests import sessionize
from workspace.kr_time_budget.pipeline.build_requests_v3 import aggregate, user_priors, item_priors
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    request_manager, FEATURES, LOG1P_FIELDS, EXTRA_COLUMNS, USER_CATEGORY_FIELDS,
    FEATURE_CATEGORIES, BUDGET_PARAMETER_FIELDS, SESSION_CLOCK_FIELDS, CONTENT_TOWER_FIELDS,
)
from workspace.kr_time_budget.pipeline.time_budget_v4 import weibull_parameters
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5, CONTENT_CLOCK_FIELDS

WORKSPACE = Path(__file__).resolve().parent
STAGES = ['events', 'requests', 'priors', 'encoded', 'batch', 'forward', 'backward']
RAW_COLUMNS = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click', 'is_like',
               'is_follow', 'is_comment', 'is_forward', 'is_hate', 'long_view', 'play_time_ms',
               'duration_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']


def compare_frames(actual, expected, columns):
    """比较前必须按 request_id 对齐；出错会明确指出字段。"""
    assert actual['request_id'].to_list() == expected['request_id'].to_list()
    for column in columns:
        left, right = actual[column], expected[column]
        if left.dtype.is_numeric():
            np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), rtol=1e-6, atol=1e-7, err_msg=column)
        else:
            assert left.to_list() == right.to_list(), column


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=WORKSPACE/'data/v3')
    parser.add_argument('--source', type=Path, default=KuaiRandDataset.DATA_KUAI_1K)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--request-id', type=int, help='挑选该Request所在的完整Session；优先于--split')
    parser.add_argument('--batch-size', type=int, default=8, help='前向最多几条Request；会包含指定Request')
    parser.add_argument('--checkpoint', type=Path, default=WORKSPACE/'runs/exit_v5/ours/stage_two/best.pt')
    parser.add_argument('--stop-at', nargs='+', choices=STAGES, default=[], help='在这些阶段停入pdb，可写多个')
    parser.add_argument('--pdb', action='store_true', help='等价于--stop-at forward')
    parser.add_argument('--skip-priors', action='store_true', help='使用落盘先验，不重新扫描全部历史曝光')
    parser.add_argument('--backward', action='store_true', help='额外执行一次loss.backward；不执行optimizer.step')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size必须>=1')
    if 'backward' in args.stop_at:
        args.backward = True
    if args.pdb:
        args.stop_at.append('forward')
    torch.set_num_threads(4)
    pl.Config.set_tbl_rows(12)
    pl.Config.set_tbl_cols(12)

    # 默认挑一个3～12屏的会话，避免只有一屏、看不到历史和继续标签。
    features_scan = pl.scan_parquet(args.data/'features_raw.parquet')
    if args.request_id is not None:
        focus = features_scan.filter(pl.col('request_id') == args.request_id).collect()
    else:
        focus = features_scan.filter((pl.col('split') == args.split) & (pl.col('req_idx') == 1)
                                     & pl.col('session_req_cnt').is_between(3, 12)).head(1).collect()
    if focus.height != 1:
        raise ValueError('没有找到目标Request；检查request_id或split')
    request_id = focus['request_id'].item()
    session_id = focus['session_id'].item()
    user_id = focus['user_id'].item()
    split = focus['split'].item()
    print(f'目标：user_id={user_id}, session_id={session_id}, request_id={request_id}, split={split}', flush=True)

    # DEBUG 1：原始19列事件→排序/切场/倒切。event_id为全局事件键。
    # events.parquet已完成全19列去重；本脚本不重新读取全部CSV做全局去重。
    raw_user_events = pl.scan_parquet(args.data/'events.parquet').filter(pl.col('user_id') == user_id).collect()
    sessionized_user = sessionize(raw_user_events.select('event_id', *RAW_COLUMNS, 'source_order', 'source_row'))
    # 只重放一个用户，所以sessionized_user的session_id是局部编号，不可直接比较全局编号。
    raw_events = raw_user_events.filter(pl.col('session_id') == session_id).sort('event_idx')
    replayed_events = sessionized_user.filter(pl.col('event_id').is_in(raw_events['event_id'].implode())).sort('event_idx')
    assert replayed_events['req_idx'].equals(raw_events['req_idx'])
    assert replayed_events['session_n'].equals(raw_events['session_n'])
    source_files = sorted(args.source.glob('log_*.csv'))
    provenance = raw_events.select('event_id', 'source_order', 'source_row').with_columns(
        pl.col('source_order').replace_strict({i: str(path) for i, path in enumerate(source_files)}).alias('source_file'))
    # source_row是CSV数据行的0起始序号（不含表头）；req_idx=0是头部余数。
    print('\n[1 events] 当前场的完整事件（含req0）：')
    print(raw_events.select('event_id', 'event_idx', 'req_idx', 'video_id', 'time_ms', 'play_time_ms', 'is_click', 'long_view'))
    if 'events' in args.stop_at:
        breakpoint()

    # DEBUG 2：连接视频侧表→正式aggregate函数→Request及前序历史。
    video_features = pl.scan_csv(args.source/'video_features_basic_1k.csv', schema_overrides={'tag': pl.String}).filter(
        pl.col('video_id').is_in(raw_events['video_id'].unique().implode())).select(
            'video_id', 'author_id', 'tag', 'video_type', 'music_type').collect()
    joined_events = raw_events.join(video_features, on='video_id', how='left', validate='m:1', maintain_order='left').with_columns(
        pl.col('tag').fill_null('__MISSING__'), pl.col('video_type').fill_null('__MISSING__'),
        pl.col('music_type').cast(pl.Int64).cast(pl.String).fill_null('__MISSING__'), pl.col('author_id').fill_null(-1),
        pl.any_horizontal(*[pl.col(c) == 1 for c in ['is_like', 'is_comment', 'is_follow', 'is_forward']]).cast(pl.Int8).alias('positive'))
    requests_saved = pl.scan_parquet(args.data/'requests_base.parquet').filter(pl.col('session_id') == session_id).sort('req_idx').collect()

    raw_user_profile = pl.read_csv(args.source/'user_features_1k.csv').filter(pl.col('user_id') == user_id)
    user_profile = raw_user_profile.select('user_id', *USER_CATEGORY_FIELDS).with_columns(
        pl.col('is_live_streamer').eq(1).cast(pl.Int8),
        *[pl.col(f'onehot_feat{i}').cast(pl.Int64).fill_null(-1) for i in range(18)])
    assert user_profile.height == 1
    for name in USER_CATEGORY_FIELDS:
        assert requests_saved[name][0] == user_profile[name][0], name

    requests_rebuilt = aggregate(joined_events).join(requests_saved.select('req_idx', 'request_id'), on='req_idx', validate='1:1').sort('req_idx')
    compare_frames(requests_rebuilt, requests_saved, [c for c in requests_rebuilt.columns if c != 'request_id'])
    event_membership = pl.scan_parquet(args.data/'request_events.parquet').filter(pl.col('session_id') == session_id).sort('event_idx').collect()
    assert event_membership.filter(pl.col('request_id').is_not_null()).group_by('request_id').len()['len'].to_list() == [6]*requests_saved.height
    print('\n[2 requests] 聚合重放与requests_base一致：')
    print(requests_rebuilt.select('request_id', 'req_idx', 'y_continue', 'elapsed_min', 'prev1_click_cnt', 'sess_cum_click_rate', 'slate_tags', 'hist_tags'))
    if 'requests' in args.stop_at:
        breakpoint()

    # DEBUG 3：全部历史→严格过去日/0506快照→平滑先验。
    feature_rows = features_scan.filter(pl.col('session_id') == session_id).sort('req_idx').collect()
    prior_query = requests_saved.select('request_id', 'user_id', pl.when(pl.col('split') == 'train').then(
        pl.col('session_date')).otherwise(pl.lit(20220507)).alias('date'))
    user_prior_rows = item_prior_rows = prior_observations = all_sessions = None
    if not args.skip_priors:
        all_sessions = pl.read_parquet(args.data/'sessions.parquet')
        user_prior_rows = user_priors(prior_query, all_sessions).sort('request_id')
        # 总体平滑率必须来自全体过去曝光，不能仅筛选当前6个视频就计算总体率。
        prior_observations = pl.scan_parquet(args.data/'events.parquet').filter(pl.col('date') <= 20220506).select(
            'video_id', 'date', 'is_click', 'long_view').collect()
        item_query = prior_query.select('request_id', 'date').join(requests_saved.select('request_id', 'slate_items'), on='request_id', validate='1:1')
        item_prior_rows = item_priors(item_query, prior_observations).sort('request_id')
        prior_rows = user_prior_rows.join(item_prior_rows, on='request_id', validate='1:1')
        compare_frames(prior_rows, feature_rows.sort('request_id'), [c for c in prior_rows.columns if c != 'request_id'])
    print('\n[3 priors] 查询date之前的历史可用；valid/test查询20220507，实际只用<=0506：')
    print(prior_query)
    print(feature_rows.select('request_id', 'user_prior_n', 'user_hist_session_min', 'slate_prior_ctr_mean', 'item_prior_cold_fraction'))
    if 'priors' in args.stop_at:
        breakpoint()

    # DEBUG 4：Float32/log1p→已train-fit的Schema.transform→编码parquet核对。
    # 不在这个小样本上重新fit，否则词表与正式checkpoint不一致。
    manager = request_manager(args.data/'schema', load=True)
    schema_input = feature_rows.with_columns(*[pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32) for c in LOG1P_FIELDS])
    output_fields = list(dict.fromkeys(manager.fields() + EXTRA_COLUMNS))
    encoded = manager.transform(schema_input.lazy()).select(output_fields).collect()
    encoded_saved = pl.scan_parquet(args.data/f'{split}.parquet').filter(pl.col('session_id') == session_id).sort('req_idx_raw').collect()
    compare_frames(encoded, encoded_saved, output_fields)
    tag_setting = manager.get_setting('slate_tags')
    tag_vocab = tag_setting.vocab
    tag_tokens = feature_rows['slate_tags'].str.split(',').to_list()
    encoded_tag_ids = encoded['slate_tags'].to_list()
    print('\n[4 encoded] Schema转换与正式parquet一致；标签先拆分，再转ID：')
    print(encoded.select('request_id', 'user_id', 'user_id_raw', 'slate_tags', 'hist_tags', 'elapsed_min'))
    if 'encoded' in args.stop_at:
        breakpoint()

    # DEBUG 5：正式ParquetStreamDataset + DataLoader(batch_size=None)→Interaction。
    # 用LazyFrame选取小片段，避免改动正式parquet；formatter/padding仍走同一实现。
    focus_position = encoded['request_id'].to_list().index(request_id)
    batch_start = max(0, focus_position - args.batch_size//2)
    encoded_slice = encoded.slice(batch_start, args.batch_size)
    dataset = ParquetStreamDataset(encoded_slice.lazy(), manager, batch_size=len(encoded_slice), shuffle=False,
                                   drop_last=False, extra_col_names=EXTRA_COLUMNS)
    interaction = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
    batch = interaction.interaction
    batch_shapes = {name: (tuple(value.shape), str(value.dtype)) for name, value in batch.items()}
    assert len(FEATURES) == 54 and batch['slate_tags'].shape[1] == 12 and batch['hist_tags'].shape[1] == 15
    print('\n[5 batch] 重点变量：interaction、batch、batch_shapes；例如：')
    for name in ['user_id', 'slate_tags', 'hist_tags', 'elapsed_min', 'req_idx', 'y_continue']:
        print(name, batch_shapes[name], batch[name][:2])
    if 'batch' in args.stop_at:
        breakpoint()

    # DEBUG 6：实际checkpoint→逐字段Embedding→塔路由→scale/shape→gamma×stay。
    model = ExitModelV5(manager, mode='ours')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.set_stage(2)
    model.eval()
    field_embeddings = model.omni_embedding(batch, split_by='name')
    budget_x = model.budget_view(batch)
    content_x = model.content_x(batch)
    # 给每个字段标出拼接后的切片；可直接看两个时钟切片是否全0。
    field_slices = {}
    offset = 0
    for name, value in field_embeddings.items():
        field_slices[name] = slice(offset, offset + value.shape[-1])
        offset += value.shape[-1]
    feature_categories = {name: category for category, names in FEATURE_CATEGORIES.items() for name in names}
    # routing仍表示实际加载的v5，target_routing单独展示新设计，不能把归属当成已消费。
    routing = {name: {'category': feature_categories[name], 'budget': name in BUDGET_PARAMETER_FIELDS,
                      'budget_formula': name == 'elapsed_min', 'content': name not in CONTENT_CLOCK_FIELDS,
                      'slice': (section.start, section.stop)} for name, section in field_slices.items()}
    target_routing = {name: {'category': feature_categories[name],
                            'budget_parameters': name in BUDGET_PARAMETER_FIELDS,
                            'budget_clock': name in SESSION_CLOCK_FIELDS,
                            'content': name in CONTENT_TOWER_FIELDS,
                            'clock_formula_pending': name == 'req_idx'} for name in FEATURES}
    for name in CONTENT_CLOCK_FIELDS:
        assert torch.count_nonzero(content_x[:, field_slices[name]]) == 0
    budget_raw = model.budget_net(budget_x)
    scale, shape = weibull_parameters(budget_raw)
    gamma, survival, log_survival = model.budget_components(batch)
    content_logits = model.content_net(content_x).squeeze(-1)
    stay = torch.sigmoid(content_logits)
    prediction = gamma.detach() * stay
    torch.testing.assert_close(prediction, model.predict(batch), rtol=0, atol=0)
    loss = model.calculate_loss(batch)
    prediction_table = pl.DataFrame({'request_id': batch['request_id'].numpy(), 'label': batch['y_continue'].numpy(),
        'elapsed_min': batch['elapsed_min'].numpy(), 'scale': scale.detach().numpy(), 'shape': shape.detach().numpy(),
        'gamma': gamma.detach().numpy(), 'stay': stay.detach().numpy(), 'prediction': prediction.detach().numpy()})
    print('\n[6 forward] loss =', loss.item())
    print(prediction_table)
    print('checkpoint selection:', checkpoint.get('selection'))
    # 训练入口仅保存test预测，因此仅对test样本做同Request核对。
    saved_predictions_path = args.checkpoint.parent.parent/'predictions.parquet'
    saved_predictions = None
    if split == 'test' and saved_predictions_path.exists():
        saved_predictions = prediction_table.select('request_id').join(
            pl.read_parquet(saved_predictions_path, columns=['request_id', 'prediction']),
            on='request_id', how='left', validate='1:1', maintain_order='left')
        assert saved_predictions['prediction'].null_count() == 0, '保存的test预测未覆盖当前Request'
        np.testing.assert_allclose(prediction.detach().numpy(), saved_predictions['prediction'].to_numpy(), rtol=1e-6, atol=1e-7)
    if 'forward' in args.stop_at:
        breakpoint()

    # DEBUG 7（可选）：反向梯度；不创建optimizer，不更新参数，不写回checkpoint。
    gradients = None
    if args.backward:
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients = {name: None if value.grad is None else value.grad.norm().item() for name, value in model.named_parameters()}
        assert all(p.grad is None for module in model.budget_modules for p in module.parameters())
        print('\n[7 backward] 预算路径无梯度；内容末层梯度norm =', gradients['content_net.4.weight'])
        if 'backward' in args.stop_at:
            breakpoint()
    print('\n完成。中间表与正式产物核对通过；未更新参数、未改写数据。')


if __name__ == '__main__':
    main()

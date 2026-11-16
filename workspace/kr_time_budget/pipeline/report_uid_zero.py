"""只读审计UID置0实验，与已有同seed/同epoch对照配对，生成报告及曲线。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from workspace.kr_time_budget.pipeline.exit_training import WORKSPACE
from workspace.kr_time_budget.pipeline.report_epochwise_exit import compact, metric_difference, METRICS
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores


def audit(run, baseline):
    config = json.loads((run/'config.json').read_text())
    reference_config = json.loads((baseline/'config.json').read_text())
    assert hashlib.sha256((baseline/'config.json').read_bytes()).hexdigest() == config['baseline_config_sha256']
    for key in ['epochs', 'batch_size', 'gradient_accumulation', 'optimizer', 'budget_lr',
                'content_or_fusion_lr', 'weight_decay', 'zero_decay', 'scheduler', 'ema', 'train_order']:
        assert config[key] == reference_config[key]
    assert config['input_ablation'].startswith('encoded user_id -> constant padding index 0')
    assert hashlib.sha256((WORKSPACE/'data/v3/manifest.json').read_bytes()).hexdigest() == config['dataset_manifest_sha256']
    for filename, digest in config['source_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    coverage = json.loads((run/'coverage.json').read_text())
    assert coverage == json.loads((baseline/'coverage.json').read_text())
    input_audit = json.loads((run/'input_audit.json').read_text())
    for split in ['train', 'valid', 'test']:
        assert input_audit[split]['user_id_nonzero_after'] == 0
        assert input_audit[split]['every_other_tensor_exact']
        assert input_audit[split]['rows'] == coverage[split]['rows']
    summary = json.loads((run/'summary.json').read_text())
    reference = json.loads((baseline/'summary.json').read_text())
    keys = {f'{seed}/{name}' for seed in config['seeds'] for name in config['experiments']}
    assert set(summary) == keys
    assert json.loads((run/'complete.json').read_text()) == dict(trajectories=len(keys), epoch_test_evaluations=5*len(keys))
    expected = {split: pl.read_parquet(WORKSPACE/f'data/v3/{split}.parquet',
        columns=['request_id', 'session_id', 'user_id_raw', 'req_idx_raw', 'y_continue'])
        for split in ['valid', 'test']}
    maximum_error, hashes, rows, deltas = 0., {}, [], []
    for key in sorted(keys):
        seed, name = key.split('/')
        initial = json.loads((run/key/'initialization.json').read_text())
        original = json.loads((baseline/key/'initialization.json').read_text())
        assert initial['baseline_initialization_exact']
        assert initial['state_sha256'] == original['state_sha256']
        assert initial['optimizer_groups'] == original['optimizer_groups']
        assert json.loads((run/key/'uid_audit.json').read_text())['all_uid_embeddings_unchanged_after_training']
        assert [r['epoch'] for r in summary[key]] == [1, 2, 3, 4, 5]
        for row in summary[key]:
            epoch = row['epoch']
            assert row['step'] == epoch*coverage['train']['batches']
            assert row['optimizer_state_steps'] == [row['step']]
            assert row['evaluation_preserved_model_optimizer_rng']
            checkpoint = torch.load(run/key/f'epoch_{epoch:02d}/checkpoint.pt', map_location='cpu', weights_only=False)
            state = checkpoint['model_state_dict']
            digest = hashlib.sha256()
            for field, tensor in sorted(state.items()):
                digest.update(field.encode()); digest.update(tensor.numpy().tobytes())
            assert digest.hexdigest() == row['state_sha256'] == checkpoint['state_sha256']
            assert checkpoint['global_step'] == row['step'] and checkpoint['epoch'] == epoch
            assert {int(s['step']) for s in checkpoint['optimizer_state_dict']['state'].values()} == {row['step']}
            for tower, field in [('content_or_fusion', 'omni_embedding'), ('budget', 'budget_embedding')]:
                weight = state[f'{field}.emb_modules.user_id.embedding.weight']
                assert torch.count_nonzero(weight[0]) == 0
                assert hashlib.sha256(weight.numpy().tobytes()).hexdigest() == initial['uid_table_sha256'][tower]
            for split in ['valid', 'test']:
                path = run/key/f'epoch_{epoch:02d}/{split}_predictions.parquet'
                frame = pl.read_parquet(path)
                for col in expected[split].columns:
                    assert frame[col].equals(expected[split][col])
                assert frame.height == frame['request_id'].n_unique() == coverage[split]['rows']
                computed = scores(frame)
                maximum_error = max(maximum_error, metric_difference(computed, row[split]))
                assert computed['user_gauc']['eligible_groups'] == reference[key][epoch-1][split]['user_gauc']['eligible_groups']
                assert computed['user_gauc']['eligible_rows'] == reference[key][epoch-1][split]['user_gauc']['eligible_rows']
                if not name.startswith('fusion'):
                    assert (frame['gamma_time'] == 1).all()
                    assert (frame['prediction']-frame['gamma']*frame['content_stay']).abs().max() == 0
                measured, original_scores = compact(computed), compact(reference[key][epoch-1][split])
                rows.append(dict(seed=int(seed), experiment=name, epoch=epoch, split=split, **measured))
                deltas.append(dict(seed=int(seed), experiment=name, epoch=epoch, split=split,
                                   **{m: measured[m]-original_scores[m] for m in METRICS}))
                hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # 复用既有对照审计，并确认原预测文件至今未变。
    baseline_audit = json.loads((baseline/'audit.json').read_text())
    for filename, digest in baseline_audit['prediction_sha256'].items():
        assert hashlib.sha256((baseline/filename).read_bytes()).hexdigest() == digest
    result = dict(trajectories=len(keys), checkpoint_count=5*len(keys), prediction_files=len(hashes),
        paired_initializations_exact=True, optimizer_groups_exact=True, all_non_uid_inputs_exact=True,
        uid_padding_vectors_zero_all_epochs=True, all_uid_tables_unchanged_all_epochs=True,
        raw_uid_and_user_auc_coverage_unchanged=True, all_request_metadata_exact=True,
        metric_absolute_tolerance=1e-12, maximum_metric_absolute_difference=maximum_error,
        baseline_prediction_hashes_unchanged=True, prediction_sha256=hashes,
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return config, reference, pl.DataFrame(rows), pl.DataFrame(deltas)


def generate(run):
    baseline = WORKSPACE/'runs/epochwise5'
    report = WORKSPACE/'回复/18拍板'
    config, reference, frame, deltas = audit(run, baseline)
    reload_path = run/'checkpoint_reload.json'
    reloaded = json.loads(reload_path.read_text()) if reload_path.exists() else {}
    assert all(r['all_prediction_columns_exact'] and r['uid_input_zero'] for r in reloaded.values())
    frame.write_parquet(run/'epoch_metrics.parquet')
    deltas.write_parquet(run/'paired_deltas.parquet')
    def aggregate(data):
        return data.group_by('experiment', 'epoch', 'split').agg(*[expr for m in METRICS for expr in
            [pl.col(m).mean().alias(m+'_mean'), pl.col(m).std(ddof=1).alias(m+'_std')]])
    aggregate(frame).write_parquet(run/'epoch_aggregates.parquet')
    aggregate(deltas).write_parquet(run/'paired_delta_aggregates.parquet')
    values = {(r['experiment'], r['epoch'], r['split']): r for r in aggregate(frame).to_dicts()}
    changes = {(r['experiment'], r['epoch'], r['split']): r for r in aggregate(deltas).to_dicts()}
    def old_mean(name, epoch, metric, split='test'):
        return float(np.mean([compact(reference[f'{s}/{name}'][epoch-1][split])[metric] for s in config['seeds']]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for name in config['experiments']:
            measured = [values[name, e, 'test'][metric+'_mean'] for e in range(1, 6)]
            line, = axis.plot(range(1, 6), measured, marker='o', markersize=3, label=name)
            axis.plot(range(1, 6), [old_mean(name, e, metric) for e in range(1, 6)],
                      linestyle='--', color=line.get_color(), alpha=.55)
        axis.set(xlabel='Epoch', ylabel=metric, xticks=range(1, 6))
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('UID ablation, 3-seed means: solid = UID zero; dashed = original UID')
    fig.savefig(report/'UID置0逐epoch曲线.png', dpi=170)
    plt.close(fig)
    def fmt(name, epoch, metric, split='test'):
        row = values[name, epoch, split]
        return f'{row[metric+"_mean"]:.6f} ± {(row[metric+"_std"] or 0.):.6f}'
    lines = ['# UID置0：连续5轮逐epoch配对实验', '',
        '**只把编码后的user_id输入置0，其余输入、模型形状、同seed初始化和训练协议不变。** 已完成6配置×3seed×5轮，90次完整test评估；与上一轮有UID的同seed、同配置、同epoch末结果配对。', '',
        '## 改动与口径', '',
        '- user_id对应padding_idx=0，Embedding第0行固定零向量。两套Omni和Fusion均被屏蔽；保留UID槽位和表形状以保持共有参数初始化完全一致。',
        '- user_id_raw未改，仍按原始用户计算User-AUC；画像、用户预算先验及其余53字段保留。这是UID字段消融，不是移除全部用户相关信息。',
        '- 置0发生在输入预处理，不在KRExitModel内部。重新加载本轮checkpoint预测时也必须使用zero_uid_inputs或等价处理，直接把原UID送入模型不属于本实验。',
        '- 原始parquet和Schema未改；train截至2022-05-06为1,721,745请求，valid=05-07为57,872，test=05-08为48,898。沿用KR v3的原切分和物理训练顺序。',
        '- 六配置、seed2026/2027/2028、batch4096、421步/轮、连续5轮共2105步；预算LR .005，内容/Fusion LR .001；一个AdamW，普通权重decay1e-6，Embedding/bias/一维参数零decay；clip10，无scheduler/EMA。',
        '- Ours仍是乘积BCE联合训练，无辅助Loss、预热、冻结；每轮末完整valid/test，固定第5轮停止，不按test选点或回滚best。',
        '- 本轮和对照均使用轮末权重；不与第14份预算内valid最优checkpoint混用。±为三个seed的样本标准差，非置信区间。', '',
        '## 第5轮完整test', '',
        '| 配置 | AUC ↑ | LogLoss ↓ | User-AUC ↑ | Intra-Step GAUC ↑ |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in config['experiments']:
        lines.append('| '+name+' | '+' | '.join(fmt(name, 5, m) for m in METRICS)+' |')
    lines += ['', '## 第5轮：置0减原UID（配对均值）', '',
        '| 配置 | ΔAUC ↑ | ΔLogLoss ↓ | ΔUser-AUC ↑ | ΔIntra-Step GAUC ↑ | AUC改善seed数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in config['experiments']:
        wins = deltas.filter((pl.col('experiment') == name) & (pl.col('epoch') == 5) &
                            (pl.col('split') == 'test') & (pl.col('auc') > 0)).height
        lines.append('| '+name+' | '+' | '.join(f'{changes[name,5,"test"][m+"_mean"]:+.6f}' for m in METRICS)+f' | {wins}/3 |')
    means = {name: changes[name,5,'test']['auc_mean'] for name in config['experiments']}
    increased = sum(delta > 0 for delta in means.values())
    lines += ['', f'第5轮AUC均值改善的配置数为{increased}/{len(means)}。'+
              '这些结果衡量同一数据和训练协议下UID字段的增量，不等于移除用户画像或历史统计后的效果。']
    if {'mixture_step_pace','fusion'} <= set(config['experiments']):
        original_gap = old_mean('mixture_step_pace',5,'auc')-old_mean('fusion',5,'auc')
        zero_gap = values['mixture_step_pace',5,'test']['auc_mean']-values['fusion',5,'test']['auc_mean']
        user_gap = values['mixture_step_pace',5,'test']['user_auc_mean']-values['fusion',5,'test']['user_auc_mean']
        lines += ['', f'混合＋pace减Fusion的第5轮AUC均值差：原UID为{original_gap:+.6f}，置0后为{zero_gap:+.6f}；'+
                  f'置0后User-AUC均值差为{user_gap:+.6f}。不能将结构间相对差变化写成每个模型绝对表现均改善。']
    lines += ['', '## 逐epoch曲线', '', '![UID消融逐轮曲线](UID置0逐epoch曲线.png)', '',
        '实线为置0、虚线为原UID；均为三个seed均值，不是独立采样置信区间。', '',
        '## 每轮完整结果与AUC配对差', '',
        '| 配置 | epoch | test AUC | ΔAUC | test LogLoss | test User-AUC | test Intra-Step GAUC | valid LogLoss |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in config['experiments']:
        for epoch in range(1, 6):
            lines.append(f'| {name} | {epoch} | '+fmt(name, epoch, 'auc')+
                f' | {changes[name,epoch,"test"]["auc_mean"]:+.6f} | '+
                ' | '.join(fmt(name, epoch, m) for m in METRICS[1:])+' | '+fmt(name, epoch, 'logloss', 'valid')+' |')
    lines += ['', '## 第5轮逐seed配对', '',
        '| seed | 配置 | 原AUC | 置0 AUC | ΔAUC | 置0 LogLoss | 原User-AUC | 置0 User-AUC |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in frame.filter((pl.col('epoch') == 5) & (pl.col('split') == 'test')).sort('seed','experiment').to_dicts():
        original = compact(reference[f'{row["seed"]}/{row["experiment"]}'][4]['test'])
        lines.append(f'| {row["seed"]} | {row["experiment"]} | {original["auc"]:.6f} | {row["auc"]:.6f} | '+
            f'{row["auc"]-original["auc"]:+.6f} | {row["logloss"]:.6f} | {original["user_auc"]:.6f} | {row["user_auc"]:.6f} |')
    lines += ['', '## 审计与解释边界', '',
        '- 所有配置同seed初始完整模型hash与原对照一致，optimizer groups一致。输入审计逐张量确认只改user_id；raw UID与用户内AUC有效覆盖未变。',
        '- 90份checkpoint的两套UID表从初始化起逐字节未变，第0行恒0；模型/优化器/RNG在评估前后不变，各轮Adam step连续。',
        '- 重新读取180份valid/test预测核对完整请求元数据及全部指标（绝对容差1e-12），Ours乘积逐值相等。原对照180份预测哈希未变。',
        f'- 真实checkpoint重载复核覆盖{len(reloaded)}个配置（记录见下方链接）；显式置0后完整test所有预测列与已保存产物逐值一致。',
        '- 这是固定KR v3切分的配对特征消融，不能把观察到的改善自动解释为某种因果退出机制，也不能据此声称跨用户泛化或线上收益。没有新增bootstrap或独立test。', '',
        '## 产物与复现', '',
        '- [训练日志](../../runs/epochwise5_uid_zero/logs/train.log) · [配置](../../runs/epochwise5_uid_zero/config.json) · [全部原始指标](../../runs/epochwise5_uid_zero/summary.json) · [输入审计](../../runs/epochwise5_uid_zero/input_audit.json) · [独立审计](../../runs/epochwise5_uid_zero/audit.json)。',
        '- [逐seed逐轮配对差](../../runs/epochwise5_uid_zero/paired_deltas.parquet) · [原UID对照报告](逐epoch5轮实验报告.md)。',
        '- [checkpoint重载复核](../../runs/epochwise5_uid_zero/checkpoint_reload.json) · [测试日志](../../runs/epochwise5_uid_zero/logs/tests.log)。',
        '- [训练入口](../../pipeline/run_uid_zero.py) · [只读报告](../../pipeline/report_uid_zero.py) · [实验说明](UID置0实验说明.md)。', '',
        '```bash', 'uv run python -m workspace.kr_time_budget.pipeline.run_uid_zero --output workspace/kr_time_budget/runs/epochwise5_uid_zero_new',
        'uv run python -m workspace.kr_time_budget.pipeline.report_uid_zero', '```', '']
    import os
    relative = os.path.relpath(run, report)
    (report/'UID置0实验报告.md').write_text('\n'.join(lines).replace('../../runs/epochwise5_uid_zero', relative))
    print('EPOCH5', json.dumps({name: {m: values[name,5,'test'][m+'_mean'] for m in METRICS}
                              for name in config['experiments']}, indent=2))
    print('REPORT', report/'UID置0实验报告.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=WORKSPACE/'runs/epochwise5_uid_zero')
    args = parser.parse_args()
    generate(args.run.resolve())


if __name__ == '__main__':
    main()

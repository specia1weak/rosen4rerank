"""Verify full probe predictions/checkpoint lineage and write one linked report."""
import hashlib
import json
from pathlib import Path
import polars as pl
import torch
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_exit_v3 import load_batches, predict, scores
from workspace.kr_time_budget.pipeline.run_exit import state_hash
from workspace.kr_time_budget.pipeline.report_v4 import user_bootstrap
from workspace.kr_time_budget.pipeline.time_budget_v6_probe import ExitModelV6Probe

ROOT = Path('workspace/kr_time_budget')
RUN = ROOT/'runs/exit_v6_probe'


def main():
    torch.set_num_threads(4)
    config = json.loads((RUN/'config.json').read_text())
    assert not config['smoke']
    for source, expected in config['source_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
    assert hashlib.sha256(Path(config['warmup_checkpoint']).read_bytes()).hexdigest() == config['warmup_checkpoint_sha256']
    summary = json.loads((RUN/'summary.json').read_text())
    assert set(summary) == {'control', 'restore_clocks', 'mask_history'}
    v5 = json.loads((ROOT/'runs/exit_v5/summary.json').read_text())
    manager = request_manager(ROOT/'data/v3/schema', load=True)
    test = load_batches(ROOT/'data/v3/test.parquet', manager, 4096)
    expected_ids = torch.cat([b['request_id'] for b in test]).numpy().tolist()
    frames = {}
    audit = {'config': config, 'models': {}, 'comparisons': {}}
    for variant, metric in summary.items():
        assert metric['full_train'] and metric['full_test']
        assert metric['optimizer_steps_executed_here'] == 421
        assert metric['budget_exactly_frozen'] and metric['initial_state_exact']
        pred = pl.read_parquet(RUN/variant/'predictions.parquet')
        assert pred.height == pred['request_id'].n_unique() == 48898
        assert pred['request_id'].to_list() == expected_ids
        check = scores(pred)
        for key in ['overall.auc', 'logloss']:
            assert abs(check[key]-metric[key]) < 1e-12
        assert abs(check['user_gauc']['auc']-metric['user_gauc']['auc']) < 1e-12
        checkpoint = torch.load(RUN/variant/'stage_two/best.pt', map_location='cpu', weights_only=False)
        model = ExitModelV6Probe(manager, variant)
        model.load_state_dict(checkpoint['state_dict'])
        model.set_stage(2)
        reloaded = predict(model, test, 'cpu')
        assert reloaded['prediction'].equals(pred['prediction'])
        assert reloaded['gamma'].equals(pred['gamma'])
        transition = json.loads((RUN/variant/'stage_two/transition.json').read_text())
        assert transition['transition_full_valid_rows'] == 57872
        assert transition['transition_prediction_max_abs_error'] == transition['transition_logloss_difference'] == 0
        manifest = json.loads((RUN/variant/'manifest.json').read_text())
        audit['models'][variant] = {'checkpoint_state_sha256': state_hash(model),
                                    'reload_predictions_exact': True, 'manifest': manifest, 'transition': transition}
        frames[variant] = pred
    assert len({v['manifest']['initial_sha256'] for v in audit['models'].values()}) == 1
    assert len({v['manifest']['train_order_sha256'] for v in audit['models'].values()}) == 1
    old_predictions = pl.read_parquet(ROOT/'runs/exit_v5/ours/predictions.parquet')
    assert frames['control']['prediction'].equals(old_predictions['prediction'])
    for name in ['restore_clocks', 'mask_history']:
        audit['comparisons'][name+'_minus_control'] = user_bootstrap(frames[name], frames['control'])
    (RUN/'audit.json').write_text(json.dumps(audit, indent=2))
    labels = {'control': 'v5原样重放', 'restore_clocks': '恢复内容塔直接时钟', 'mask_history': '额外屏蔽内容塔历史代理'}
    restored, masked = summary['restore_clocks'], summary['mask_history']
    lines = ['# Argon 7：最小路由对照实验', '',
             '**恢复直接时钟挽回了一部分Pooled AUC，但仍未超过Fusion；额外屏蔽历史代理没有把User-GAUC拉回0.60，且损失Pooled AUC与LogLoss。**', '',
             '已完成两项改动、v5原样重放，以及全部选中checkpoint的完整test重载核对。', '',
             '## 结果', '', '| 模型 | Pooled AUC | User-GAUC | LogLoss |', '| --- | ---: | ---: | ---: |']
    for name, metric in [('Fusion（已有v5）', v5['fusion']), ('Budget-only（已有v5）', v5['budget'])] + [(labels[k], summary[k]) for k in labels]:
        lines.append(f"| {name} | {metric['overall.auc']:.6f} | {metric['user_gauc']['auc']:.6f} | {metric['logloss']:.6f} |")
    lines += ['', '两项新实验相对v5原样重放：', '', '| 改动 | ΔAUC | ΔUser-GAUC | ΔLogLoss | User-GAUC差值95%区间 |', '| --- | ---: | ---: | ---: | --- |']
    base = summary['control']
    for name in ['restore_clocks', 'mask_history']:
        metric = summary[name]
        ci = audit['comparisons'][name+'_minus_control']['paired_user_bootstrap_95']
        lines.append(f"| {labels[name]} | {metric['overall.auc']-base['overall.auc']:+.6f} | {metric['user_gauc']['auc']-base['user_gauc']['auc']:+.6f} | {metric['logloss']-base['logloss']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines += ['', '## 本轮判断', '',
              f"1. 恢复时钟后AUC为{restored['overall.auc']:.6f}，比v5提高{restored['overall.auc']-summary['control']['overall.auc']:.6f}，但比Fusion低{v5['fusion']['overall.auc']-restored['overall.auc']:.6f}；User-GAUC降为{restored['user_gauc']['auc']:.6f}。直接时钟在当前协议下确实影响两种排序指标，但本次尚未检验混合威布尔或联合微调。",
              f"2. 屏蔽历史后User-GAUC为{masked['user_gauc']['auc']:.6f}，只提高{masked['user_gauc']['auc']-summary['control']['user_gauc']['auc']:.6f}；配对区间跨0，且AUC/LogLoss均变差。没有支持‘只需去掉累计代理便可回到0.60’的预期。",
              '3. 两种变体都没有同时改善两类排序指标。当前证据不足以确认内容解耦成功；不应继续以User-GAUC达到某阈值作为唯一目标。若继续路线B，下一步可单独检验混合威布尔；若主目标为内容价值，应先明确W/R/G监督和Direct-W对照。', '',
              '## 训练与评估口径', '',
              'KR 1K data/v3：train≤20220506，共1,721,745条；valid=20220507，共57,872条；test=20220508，共48,898条。54字段，Schema、切分和train物理顺序不变。',
              '三组共用v5 Ours已选中的预算预热checkpoint，逐tensor相同地初始化第二阶段：末层零权重、bias=3.5。预算塔完全冻结，三组的全部test gamma与v5逐值相同。每组只训练第二阶段1epoch、421步；包含复用预热后总训练量为1+1epoch。',
              'seed2026、CPU 4线程、batch4096、AdamW lr=1e-3、普通权重衰减1e-6，Embedding/bias/一维参数零衰减。无梯度累计、scheduler或EMA。完整valid在step0、106、211、421评估，按LogLoss选checkpoint，选定后评估完整test。',
              '对照只改变Ours内容输入掩码，不改网络维度或参数数量。恢复时钟组仅恢复elapsed_min/req_idx；历史屏蔽组在v5基础上额外屏蔽is_first_req、三种累计反馈率、累计hate次数和hist_tags，保留用户/开场上下文、上一屏反馈和当屏列表。', '',
              '## 实现与复现核对', '',
              '- 三组初始state逐tensor相同，train顺序哈希相同；完整valid阶段切换误差均为0。',
              '- v5原样重放的48,898条预测与历史v5逐值相同；预算全state冻结，三组gamma逐值相同。',
              '- 所有选中checkpoint独立重新加载，完整test预测与各自保存结果逐值相同；乘积公式最大误差0。',
              '- 新增4项测试通过，覆盖路由、共有初始化、阶段切换、内容末层梯度和预算冻结。', '',
              '## 内容贡献与选择记录', '',
              '| 变体 | valid所选步 | 内容stay均值 | 内容stay标准差 | 非回退相邻屏预测上升比例 | 训练及评估秒数 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, metric in summary.items():
        lines.append(f"| {labels[name]} | {metric['selected']['step']} | {metric['content_stay_mean']:.6f} | {metric['content_stay_std']:.6f} | {metric['prediction_increases']/metric['nondecreasing_time_pairs']:.2%} | {metric['wall_seconds']:.1f} |")
    lines += ['', '判断时应同时参考Budget-only：内容头变成常数即可保持预算排序，因此User-GAUC回升本身不证明内容价值或因果去偏。相邻预测变化是诊断，不能单独归因于隐形时钟。',
              '这是单seed、同初始化的路由对照；Fusion使用已有同数据结果，但其训练1epoch、Ours总计1+1epoch。用户配对bootstrap为2000次，反映当前test用户抽样波动，不包含重训种子波动；已有test被多轮查看，属于探索性结果。', '',
              '## 代码和日志', '',
              '- [模型](pipeline/time_budget_v6_probe.py)、[训练入口](pipeline/run_exit_v6_probe.py)、[报告生成](pipeline/report_v6_probe.py)、[测试](tests/test_v6_probe.py)。',
              '- [训练日志](runs/exit_v6_probe/logs/train.log)、[测试日志](runs/exit_v6_probe/logs/tests.log)、[smoke日志](runs/exit_v6_probe/logs/smoke.log)。',
              '- [配置](runs/exit_v6_probe/config.json)、[完整指标](runs/exit_v6_probe/summary.json)、[审计及配对区间](runs/exit_v6_probe/audit.json)。', '',
              '在仓库根目录执行：', '', '```bash',
              'uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v6_probe',
              'uv run python -m workspace.kr_time_budget.pipeline.report_v6_probe',
              '```', '', '已有完整结果直接复用。新参数或重新训练使用新的--output目录；本报告脚本默认审计exit_v6_probe。', '']
    (ROOT/'7探路实验报告.md').write_text('\n'.join(lines))
    print(json.dumps({k: {m: v[m] for m in ['overall.auc', 'user_gauc', 'logloss']} for k, v in summary.items()}, indent=2))
    print(ROOT/'7探路实验报告.md')


if __name__ == '__main__':
    main()

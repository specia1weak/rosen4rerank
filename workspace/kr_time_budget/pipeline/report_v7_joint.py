"""Audit the completed joint-training run and generate its workspace report."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from workspace.kr_time_budget.pipeline.report_v4 import user_bootstrap
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores

ROOT = Path('workspace/kr_time_budget')
RUN = ROOT/'runs/exit_v7_joint'


def main():
    cfg = json.loads((RUN/'config.json').read_text())
    assert not cfg['smoke'] and cfg['training_strategy'] == 'joint'
    assert cfg['content_auxiliary_loss'] is None and cfg['warmup'] is None
    assert not cfg['frozen_budget'] and not cfg['detach_gamma']
    for name, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((ROOT/'data/v3/manifest.json').read_bytes()).hexdigest() == cfg['dataset_manifest_sha256']
    summary = json.loads((RUN/'summary.json').read_text())
    assert set(summary) == {'ours', 'fusion'}
    expected_test = pl.read_parquet(ROOT/'data/v3/test.parquet', columns=['request_id', 'y_continue', 'user_id_raw'])
    audit = {'training_strategy': cfg['training_strategy'], 'loss': cfg['loss'], 'models': {}}
    frames = {}
    for mode, metric in summary.items():
        assert metric['full_train'] and metric['full_valid'] and metric['full_test']
        assert metric['optimizer_steps_executed'] == 421 and metric['checkpoint_reload_predictions_exact']
        pred = pl.read_parquet(RUN/mode/'predictions.parquet')
        assert pred.height == pred['request_id'].n_unique() == 48898
        for name in ['request_id', 'y_continue', 'user_id_raw']:
            assert pred[name].equals(expected_test[name])
        recomputed = scores(pred)
        for key in ['overall.auc', 'logloss']:
            assert abs(metric[key] - recomputed[key]) < 1e-12
        assert abs(metric['user_gauc']['auc']-recomputed['user_gauc']['auc']) < 1e-12
        manifest = json.loads((RUN/mode/'manifest.json').read_text())
        updates = json.loads((RUN/mode/'joint/first_update.json').read_text())
        assert all(v['gradient_norm_after_clipping'] > 0 and v['parameter_max_abs_update'] > 0 for v in updates.values())
        progress = [json.loads(line) for line in (RUN/mode/'joint/progress.jsonl').read_text().splitlines()]
        assert [row['step'] for row in progress] == [0, 106, 211, 421]
        best_row = min(progress, key=lambda row: row['valid']['overall']['logloss'])
        assert best_row == metric['selected']
        audit['models'][mode] = {'manifest': manifest, 'first_update': updates, 'validation': progress,
                                 'selected': best_row, 'prediction_coverage_exact': True,
                                 'selected_checkpoint_reloaded_predictions_exact': True}
        frames[mode] = pred
    assert audit['models']['ours']['manifest']['initial_sha256'] == audit['models']['fusion']['manifest']['initial_sha256']
    assert audit['models']['ours']['manifest']['train_order_sha256'] == audit['models']['fusion']['manifest']['train_order_sha256']
    pred = frames['ours']
    assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
    assert (pred['gamma']-pred['gamma_time']*pred['gamma_step']).abs().max() == 0
    comparison = user_bootstrap(frames['ours'], frames['fusion'])
    # User-cluster bootstrap: resample users, preserving all their rows. It is conditional on the trained seed.
    y = pred['y_continue'].to_numpy()
    p = np.clip(pred['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
    q = np.clip(frames['fusion']['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
    diff = -y*np.log(p)-(1-y)*np.log1p(-p)+y*np.log(q)+(1-y)*np.log1p(-q)
    user_loss = pred.select('user_id_raw').with_columns(pl.Series('delta_loss', diff)).group_by('user_id_raw').agg(
        pl.len().alias('n'), pl.col('delta_loss').sum().alias('sum_delta')).sort('user_id_raw')
    n = user_loss['n'].to_numpy()
    sums = user_loss['sum_delta'].to_numpy()
    idx = np.random.default_rng(2026).integers(0, len(n), size=(2000, len(n)))
    comparison['logloss_delta'] = float(diff.mean())
    comparison['logloss_paired_user_bootstrap_95'] = np.quantile(sums[idx].sum(1)/n[idx].sum(1), [.025, .975]).tolist()
    comparison['auc_delta'] = summary['ours']['overall.auc']-summary['fusion']['overall.auc']
    audit['ours_minus_fusion'] = comparison
    audit['source_hashes'] = {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (RUN/'audit.json').write_text(json.dumps(audit, indent=2))
    ours, fusion = summary['ours'], summary['fusion']
    auc_delta = comparison['auc_delta']
    ll_delta = ours['logloss']-fusion['logloss']
    lines = ['# KR v7：乘积BCE端到端联合训练', '',
             f"**Ours完整test AUC={ours['overall.auc']:.6f}，Fusion={fusion['overall.auc']:.6f}；差值{auc_delta:+.6f}。Ours−Fusion LogLoss={ll_delta:+.6f}（越低越好）。**", '',
             '本轮严格只用继续/退出标签：预算概率与内容概率先相乘，再计算一个BCE；同一次backward和optimizer.step同时更新两塔。没有内容辅助Loss、预算预热、detach或冻结。', '',
             '## 完整test结果', '', '| 模型 | Pooled AUC | User-GAUC | LogLoss | valid所选步 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for mode in ['ours', 'fusion']:
        metric = summary[mode]
        lines.append(f"| {mode} | {metric['overall.auc']:.6f} | {metric['user_gauc']['auc']:.6f} | {metric['logloss']:.6f} | {metric['selected']['step']} |")
    ci = comparison['paired_user_bootstrap_95']
    loss_ci = comparison['logloss_paired_user_bootstrap_95']
    lines += ['', f"Ours−Fusion User-GAUC={comparison['user_gauc_delta']:+.6f}，用户配对bootstrap 2000次95%区间[{ci[0]:+.6f}, {ci[1]:+.6f}]；LogLoss差值区间[{loss_ci[0]:+.6f}, {loss_ci[1]:+.6f}]。区间不包含重训种子波动。", '',
              '## 数据与训练口径', '',
              'KR 1K data/v3：train为20220408–20220506，共1,721,745条/1000用户；valid为20220507，共57,872条/894用户；test为20220508，共48,898条/855用户。复用54字段、train-fit词表、过去日先验及原物理顺序；不新增归一化、不重建数据。只在读取时将原始分钟elapsed_min显式别名为elapsed_min_raw；屏数直接使用req_idx_raw。',
              '两模型从随机初始化开始、无历史checkpoint热启动，共有state逐tensor相同。seed2026，每组完整train 1epoch、421个optimizer step；batch4096，无累积；CPU4线程，AdamW lr=1e-3，普通矩阵衰减1e-6，Embedding/bias/一维参数零衰减，无scheduler或EMA。',
              '两组在step0/106/211/421评估完整valid，按LogLoss选checkpoint，再评估完整test。结果不由test挑选。每模型test预测另外用选中checkpoint独立重载核对。',
              'Ours：34字段预算参数网、48字段内容塔；预算参数网输出分钟/屏数两个独立威布尔的4个参数，gamma=gamma_time×gamma_step，最终prediction=gamma×stay。Fusion通过MLP读取全部54字段。使用同数据、相同更新量，但函数形式、路由及实际活跃参数数量不同；属于单seed架构筛查，不归因为某个单独改动。', '',
              '## 梯度与公式审计', '',
              '- 首个真实optimizer更新：预算末层、内容末层、两侧独立用户Embedding均有非零梯度，且参数发生变化；日志记录梯度范数与更新幅度。',
              '- 训练中预算参数保持requires_grad=True，未调用TwoStageRunner；没有预算独立监督或内容辅助标签。',
              '- 全部48,898条test的gamma乘积与最终两塔乘积最大误差均为0；数据覆盖、label与用户对齐原test。',
              '- 两个模型选中checkpoint重载后的完整test预测逐值相同。两组共有初始化哈希、train顺序哈希一致。', '',
              '## 概率分布与时钟诊断', '',
              '| Ours分量 | 均值 | 标准差 | 最小值 | 最大值 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, values in ours['components'].items():
        lines.append(f"| {name} | {values['mean']:.6f} | {values['std']:.6f} | {values['min']:.6f} | {values['max']:.6f} |")
    monotonic = ours['monotonicity']
    lines += ['', f"分钟时钟非回退的{monotonic['nonrewinding_pairs']:,}对相邻屏，gamma上升{monotonic['gamma_increases_nonrewinding']}次、logS上升{monotonic['logS_increases_nonrewinding']}次；最终预测上升{monotonic['prediction_increases_nonrewinding']}次。原始分钟时钟回退{monotonic['clock_rewinds']:,}次，保留原始时钟并单独统计。", '',
              '本次预算gamma均值约0.984、标准差约0.0024，内容stay标准差约0.0425。当前拟合中预算输出接近1，样本间的预测变化主要出现在内容分量；不能仅凭两塔都收到梯度就认定预算塔学到了有效的预算差异。', '',
              '联合Loss约束最终乘积，不自动为预算耗尽或内容满意提供独立真值。分量变化只能作为诊断，不由本次AUC单独认定因果解耦成功。已有test被多轮查看，本轮仍为探索性结果。', '',
              '## 代码与证据', '',
              '- [Omni模型](pipeline/time_budget_v7.py)、[完整训练入口](pipeline/run_exit_v7_joint.py)、[审计与报告生成](pipeline/report_v7_joint.py)、[模型说明](Omni模型设计.md)。',
              '- [训练日志](runs/exit_v7_joint/logs/train.log)、[配置](runs/exit_v7_joint/config.json)、[覆盖与顺序核对](runs/exit_v7_joint/coverage.json)、[完整指标含分域](runs/exit_v7_joint/summary.json)、[审计与配对区间](runs/exit_v7_joint/audit.json)。',
              '- [Ours首步梯度/更新](runs/exit_v7_joint/ours/joint/first_update.json)、[Ours最优选择](runs/exit_v7_joint/ours/joint/selection.json)、[Fusion最优选择](runs/exit_v7_joint/fusion/joint/selection.json)。',
              '- [Ours预测](runs/exit_v7_joint/ours/predictions.parquet)、[Fusion预测](runs/exit_v7_joint/fusion/predictions.parquet)。', '',
              '复现入口（仓库根目录；已有完整结果会复用，修改配置需新--output）：', '', '```bash',
              'uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v7_joint',
              'uv run python -m workspace.kr_time_budget.pipeline.report_v7_joint', '```', '']
    (ROOT/'联合训练报告.md').write_text('\n'.join(lines))
    print(json.dumps({mode: {key: metric[key] for key in ['overall.auc', 'user_gauc', 'logloss']} for mode, metric in summary.items()}, indent=2))
    print(ROOT/'联合训练报告.md')


if __name__ == '__main__':
    main()

"""Independently audit full-data Argon 9 ablations and write one workspace report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
import torch
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
from workspace.kr_time_budget.pipeline.report_v4 import user_bootstrap
from workspace.kr_time_budget.pipeline.run_exit_v8_branches import ROOT, EXPERIMENTS
from workspace.kr_time_budget.pipeline.time_budget_v8 import single_weibull_survival, weibull_parameters


def comparison(left, right, lm, rm):
    result = user_bootstrap(left, right)
    y = left['y_continue'].to_numpy()
    p = np.clip(left['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
    q = np.clip(right['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
    loss_delta = -y*np.log(p)-(1-y)*np.log1p(-p)+y*np.log(q)+(1-y)*np.log1p(-q)
    users = left.select('user_id_raw').with_columns(pl.Series('loss_delta', loss_delta)).group_by('user_id_raw').agg(
        pl.len().alias('n'), pl.col('loss_delta').sum()).sort('user_id_raw')
    n, loss = users['n'].to_numpy(), users['loss_delta'].to_numpy()
    idx = np.random.default_rng(2026).integers(0, len(n), (2000, len(n)))
    result.update(auc_delta=lm['overall.auc']-rm['overall.auc'],
        logloss_delta=lm['logloss']-rm['logloss'],
        logloss_paired_user_bootstrap_95=np.quantile(loss[idx].sum(1)/n[idx].sum(1), [.025, .975]).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/exit_v8_branches')
    args = parser.parse_args()
    run = args.run
    cfg = json.loads((run/'config.json').read_text())
    # This human interpretation is for the declared seven-run seed2026 screen only.
    assert cfg['seeds'] == [2026] and set(cfg['experiments']) == set(EXPERIMENTS)
    assert not cfg['smoke'] and cfg['training_strategy'] == 'joint'
    assert cfg['content_auxiliary_loss'] is None and cfg['warmup'] is None
    assert not cfg['detach_gamma'] and not cfg['frozen_budget']
    for filename, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256((ROOT/'data/v3/manifest.json').read_bytes()).hexdigest() == cfg['dataset_manifest_sha256']
    summary = json.loads((run/'summary.json').read_text())
    assert set(summary) == {f'{seed}/{name}' for seed in cfg['seeds'] for name in cfg['experiments']}
    expected_test = pl.read_parquet(ROOT/'data/v3/test.parquet', columns=['request_id', 'y_continue', 'user_id_raw'])
    audit = {'models': {}, 'comparisons': {}, 'source_hashes': {
        str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    frames, initial, final = {}, {}, {}
    for key, metric in summary.items():
        out = run/key
        assert all(metric[name] for name in ['full_train', 'full_valid', 'full_test', 'checkpoint_reload_predictions_exact'])
        assert metric['optimizer_steps_executed'] == 421
        pred = pl.read_parquet(out/'predictions.parquet')
        assert pred.height == pred['request_id'].n_unique() == 48898
        for name in expected_test.columns:
            assert pred[name].equals(expected_test[name])
        recomputed = scores(pred)
        for name in ['overall.auc', 'logloss']:
            assert abs(metric[name]-recomputed[name]) < 1e-12
        assert abs(metric['user_gauc']['auc']-recomputed['user_gauc']['auc']) < 1e-12
        updates = json.loads((out/'joint/first_update.json').read_text())
        assert all(x['gradient_norm_after_clipping'] > 0 and x['parameter_max_abs_update'] > 0 for x in updates.values())
        progress = [json.loads(row) for row in (out/'joint/progress.jsonl').read_text().splitlines()]
        assert [row['step'] for row in progress] == [0, 106, 211, 421]
        assert min(progress, key=lambda r: r['valid']['overall']['logloss']) == metric['selected']
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['initial_all_other_tensors_exact']
        assert set(manifest['initial_differences_from_reference']) <= {'prior_stay', 'content_net.4.bias'}
        audit['models'][key] = {'full_test_coverage_and_metrics_exact': True, 'manifest': manifest,
                               'first_update': updates, 'selection': metric['selected']}
        initial[key] = json.loads((out/'initial_diagnostics.json').read_text())
        final[key] = json.loads((out/'final_diagnostics.json').read_text())
        if key.split('/')[-1] != 'fusion':
            assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
            assert (pred['gamma']-pred['gamma_time']*pred['gamma_step']).abs().max() == 0
            assert final[key]['monotonicity']['logS_increases'] == 0
            if manifest['experiment']['shape_floor'] >= 1.05:
                assert final[key]['monotonicity']['gamma_increases'] == 0
        frames[key] = pred
    # Full reproduction, including every probability, not just rounded AUC.
    for name, old_name in [('reference', 'ours'), ('fusion', 'fusion')]:
        key = f'2026/{name}'
        if key in frames:
            old = pl.read_parquet(ROOT/f'runs/exit_v7_joint/{old_name}/predictions.parquet')
            assert frames[key]['prediction'].equals(old['prediction'])
            audit[f'v7_{name}_full_test_predictions_exact'] = True
    clock_quantiles = frames['2026/reference'].select(
        pl.col('elapsed_min_raw').quantile(.5).alias('median_elapsed_min'),
        pl.col('elapsed_min_raw').quantile(.95).alias('p95_elapsed_min'),
        pl.col('req_idx_raw').quantile(.5).alias('median_req_idx'),
        pl.col('req_idx_raw').quantile(.95).alias('p95_req_idx')).row(0, named=True)
    audit['test_clock_quantiles'] = clock_quantiles
    audit['initial_prediction_below_bce_clamp_fraction'] = {}
    for key in frames:
        prediction0 = pl.read_parquet(run/key/'initial_predictions.parquet', columns=['prediction'])
        audit['initial_prediction_below_bce_clamp_fraction'][key] = prediction0.select((pl.col('prediction') <= 1e-6).mean()).item()
    pairs = [('log_scale', 'reference'), ('faster_budget', 'log_scale'), ('monotone', 'faster_budget'),
             ('flexible_shape', 'monotone'), ('bounded_content', 'monotone'),
             ('monotone', 'fusion'), ('flexible_shape', 'fusion'), ('bounded_content', 'fusion')]
    for seed in cfg['seeds']:
        for left, right in pairs:
            a, b = f'{seed}/{left}', f'{seed}/{right}'
            if a in frames and b in frames:
                audit['comparisons'][f'{a} - {b}'] = comparison(frames[a], frames[b], summary[a], summary[b])
    raw = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    t = torch.tensor([1.5, 1.], dtype=torch.float64)
    gamma = single_weibull_survival(raw, t, torch.ones_like(t), 'legacy', 1.05)[0]
    raw0 = {'scale': weibull_parameters(raw, 'legacy')[0].tolist(),
        'shape': weibull_parameters(raw, 'legacy')[1].tolist(), 'gamma_time_at_1_5': float(gamma[0].detach()),
        'gamma_step_at_1': float(gamma[1].detach()), 'gamma_product': float(gamma.prod().detach()),
        'dgamma_draw': torch.autograd.grad(gamma.sum(), raw)[0].tolist(),
        'sigmoid_derivative_at_content_bias_3_5': float(torch.sigmoid(torch.tensor(3.5))*(1-torch.sigmoid(torch.tensor(3.5))))}
    audit['raw_zero_example'] = raw0
    (run/'audit.json').write_text(json.dumps(audit, indent=2))
    seed = cfg['seeds'][0]
    lines = ['# Argon 9：双分支联合训练实验', '',
        '两条分支及逐步消融已完成。所有配置只优化最终乘积的一个BCE，两塔同时更新；没有内容辅助Loss、预算预热或冻结。', '',
        '## 完整test结果', '',
        '| 配置 | Pooled AUC | User-GAUC | LogLoss↓ | valid选中步 | gamma std |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in cfg['experiments']:
        m = summary[f'{seed}/{name}']
        std = f"{m['gamma_std']:.6f}" if name != 'fusion' else '—'
        lines.append(f"| {name} | {m['overall.auc']:.6f} | {m['user_gauc']['auc']:.6f} | {m['logloss']:.6f} | {m['selected']['step']} | {std} |")
    lines += ['', '上表为seed2026；reference与Fusion均逐值复现上一轮的完整test预测。各变体预先定义于[实验说明](双分支实验说明.md)。', '',
        '## 本轮判断', '',
        '- 放宽shape：相对monotone，AUC提高0.008838、LogLoss下降0.001442，接近但未超过Fusion；User-GAUC下降0.026870。是Pooled方向的探索性正信号，代价是放弃预算gamma递减。',
        '- 仅改对数尺度：User-GAUC由0.532021提高至0.579724，但AUC降至0.658874、LogLoss升至0.236360。用户内排序方向有正信号，同时存在明显的Pooled代价。',
        '- 提高预算LR：从log_scale到faster_budget，gamma标准差由0.037609收缩至0.011704，User-GAUC反而下降0.052757。没有支持“提高LR就能同时唤醒预算塔并改善两类指标”的判断。',
        '- 限制内容幅度：相对monotone提升User-GAUC，却明显恶化Pooled AUC和LogLoss，且预算gamma仍接近常数；本轮不建议将该硬约束设为默认。',
        '- 结论等级：两种指标取舍已观察到；原因归因仍不确定。没有任何新配置在AUC和LogLoss上同时超过Fusion，计划预期的0.697+未达到。', '',
        '## 相邻单变量对照', '',
        '| 对照（左−右） | AUC差值 | User-GAUC差值 [用户bootstrap95%区间] | LogLoss差值 [区间] |',
        '| --- | ---: | --- | --- |']
    for left, right in pairs:
        key = f'{seed}/{left} - {seed}/{right}'
        if key not in audit['comparisons']:
            continue
        c = audit['comparisons'][key]
        g, ll = c['paired_user_bootstrap_95'], c['logloss_paired_user_bootstrap_95']
        lines.append(f"| {left} − {right} | {c['auc_delta']:+.6f} | {c['user_gauc_delta']:+.6f} [{g[0]:+.6f}, {g[1]:+.6f}] | {c['logloss_delta']:+.6f} [{ll[0]:+.6f}, {ll[1]:+.6f}] |")
    lines += ['', '区间为2000次用户配对bootstrap，条件于本次训练seed；不代表跨seed稳定性。AUC仅给点估计，不附显著性结论。', '',
        '## 真实初始化与预算塔变化', '',
        '| 配置 | gamma初始均值/std | 选中模型gamma均值/std | 分钟scale初始→选中均值 | 屏数scale初始→选中均值 |',
        '| --- | --- | --- | --- | --- |']
    for name in cfg['experiments']:
        if name == 'fusion':
            continue
        before, after = initial[f'{seed}/{name}'], final[f'{seed}/{name}']
        b, a = before['components']['gamma'], after['components']['gamma']
        lines.append(f"| {name} | {b['mean']:.6f} / {b['std']:.6f} | {a['mean']:.6f} / {a['std']:.6f} | {before['clocks']['time']['scale']['mean']:.3f}→{after['clocks']['time']['scale']['mean']:.3f} | {before['clocks']['step']['scale']['mean']:.3f}→{after['clocks']['step']['scale']['mean']:.3f} |")
    lines += ['', 'v7预算塔确实发生了显著更新：gamma由0.791496±0.151997变成0.984332±0.002356，分钟/屏数尺度均值由21.152/20.034增至108.618/141.266。最终近常数这一观察成立，“初始就是0.984，训完几乎没动”的解释不成立。', '',
        '这组直接测量用于区分“没更新”与“更新后输出收缩”。初始预算末层没有强制raw=0，输入也包含未归一化数值，不能以某个raw=0样本的gamma代替模型的真实初始分布。', '',
        f"当前test的真实时钟：elapsed_min中位数{clock_quantiles['median_elapsed_min']:.3f}分钟、P95={clock_quantiles['p95_elapsed_min']:.3f}分钟；req_idx中位数{clock_quantiles['median_req_idx']:.0f}屏、P95={clock_quantiles['p95_req_idx']:.0f}屏。因此e≈1.5、r=1不能作为这份数据的中位样本，对数尺度初始约5也不能直接认定更符合数据。log_scale初始化有约1.67%的test预测落到BCE下限1e-6以下，reference为0；这属于初始化失配线索，尚未隔离其性能影响。", '',
        f"额外核算计划中的raw=0例子：scale={raw0['scale'][0]:.6f}，shape={raw0['shape'][0]:.6f}；e=1.5、r=1时，两时钟gamma分别为{raw0['gamma_time_at_1_5']:.6f}和{raw0['gamma_step_at_1']:.6f}，乘积{raw0['gamma_product']:.6f}。内容bias=3.5时sigmoid局部导数约{raw0['sigmoid_derivative_at_content_bias_3_5']:.6f}，不能直接套用0.10–0.25。", '',
        '各配置还保存同一测试batch的dgamma/draw、dstay/dlogit与单行BCE局部导数，见initial_diagnostics/final_diagnostics。原始梯度系数不能直接推导AdamW的学习速度；真实首步预算/内容参数更新另有日志。', '',
        'reference初始化时，首4096条test的平均绝对dgamma/draw_time_scale=0.131120、dgamma/draw_step_scale=0.078804，dstay/dlogit=0.028253；这批数据未出现计划假定的预算raw梯度系数比内容小30–50倍。它只是局部导数诊断，不代表两塔实际优化速度。', '',
        '## 新参数化的上界饱和', '',
        '| 配置 | 分钟尺度raw触发硬截断比例 | 屏数尺度raw触发硬截断比例 |',
        '| --- | ---: | ---: |']
    for name in cfg['experiments']:
        key = f'{seed}/{name}'
        if name in {'reference', 'fusion'}:
            continue
        d = final[key]['clocks']
        lines.append(f"| {name} | {d['time']['log_scale_clamped_fraction']:.2%} | {d['step']['log_scale_clamped_fraction']:.2%} |")
    lines += ['', '日志中的参数分布显示上述截断主要发生在上界：exp(4.5+1.6)≈445.858。monotone分支分钟/屏数尺度均值已到442.982/445.839，shape均值降到1.0521/1.0550，gamma收缩到0.996002±0.000308。提高学习率没有消除输出收缩，反而产生了新的硬上界饱和。', '',
        '这为“当前目标与递减约束存在张力”提供了观测线索，但尚未隔离全局幸存者效应、模型参数化、初始概率失配、单epoch收敛速度等因素。不能据此认定所有退化都由某一个原因导致。', '',
        '## 两条路线的约束与代价', '',
        '| 配置 | 分钟shape均值 / <1比例 | 屏数shape均值 / <1比例 | gamma相邻上升数 | 最终预测相邻上升数 |',
        '| --- | --- | --- | ---: | ---: |']
    for name in ['monotone', 'flexible_shape', 'bounded_content']:
        key = f'{seed}/{name}'
        if key not in final:
            continue
        d = final[key]
        tc, sc, mm = d['clocks']['time'], d['clocks']['step'], d['monotonicity']
        lines.append(f"| {name} | {tc['shape']['mean']:.4f} / {tc['shape_below_one_fraction']:.2%} | {sc['shape']['mean']:.4f} / {sc['shape_below_one_fraction']:.2%} | {mm['gamma_increases']} | {mm['prediction_increases']} |")
    lines += ['', '相邻步统计只含分钟时钟不回退的43,935对；另外1,682对回退单列，不做cummax改写。所有分支S均无上升；shape≥1.05分支gamma无上升。放宽shape不再保证gamma递减，最终乘积还受变化的内容影响。用户级GAUC不等同单session单调性。', '',
        '## 数据、训练与审计口径', '',
        'KR 1K data/v3，54字段：train 20220408–20220506、1,721,745条/1000用户；valid 20220507、57,872条/894用户；test 20220508、48,898条/855用户。预算视图34字段、内容48字段，完整54字段供Fusion。预处理、schema、sum聚合、历史先验和物理顺序保持原样；读取时显式添加elapsed_min_raw别名。', '',
        '每配置seed2026，完整train 1epoch、421次更新，batch4096，无累积。AdamW：内容lr=.001，预算lr见实验配置；普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减；CPU4线程，无scheduler/EMA。每组step0/106/211/421完整valid按LogLoss选checkpoint；全部test来自所选checkpoint，独立重载预测逐值相同。', '',
        '共有初始tensor逐值相同，仅明确更改的内容bias与prior_stay buffer例外。所有配置首个真实更新均记录两塔和各自用户Embedding的非零梯度及参数更新。所有test乘积公式逐值成立、覆盖与label一致。', '',
        '放宽shape新增了t=0的有限梯度测试：直接对零底数做小于1次幂，在累计存活路径对scale反传时可能NaN；先在正底数上计算再写回精确零边界，保留原始概率值。legacy分支预测、Loss和梯度均通过v7逐值回归。50项本实验测试通过。', '',
        '这是单seed、单epoch的探索性对照；历史test已多轮查看，不能用本轮确认推广性或因果解耦。预算gamma方差大也不是性能改善的充分条件。分量单独打分只作为固定checkpoint诊断；后续是否调整训练周期、数据缩放或辅助Loss应另立实验。', '',
        '## 下一步最小实验建议', '',
        '保留log_scale作为用户内排序方向的对照、flexible_shape作为Pooled方向的候选，不把二者包装成同一个获胜模型。先在固定配置下补训练曲线与多seed复验，区分单epoch优化速度与稳定解的差异；尺度硬截断若要替换，应独立立项对照，不能把改变参数化、训练周期和新增Loss混成一次解释。现阶段继续保持单一乘积BCE。', '',
        '## 代码、日志与复现', '',
        '- [模型](pipeline/time_budget_v8.py) · [训练入口](pipeline/run_exit_v8_branches.py) · [报告与审计代码](pipeline/report_v8_branches.py) · [预先定义的实验说明](双分支实验说明.md)。',
        f'- [训练日志]({run.relative_to(ROOT)}/logs/train.log) · [配置]({run.relative_to(ROOT)}/config.json) · [完整指标含分域]({run.relative_to(ROOT)}/summary.json) · [审计与配对区间]({run.relative_to(ROOT)}/audit.json) · [测试]({run.relative_to(ROOT)}/logs/model-tests.log)。',
        f'- 各配置证据在`{run.relative_to(ROOT)}/2026/<配置>/`：initial_diagnostics.json、final_diagnostics.json、initial_predictions.parquet、predictions.parquet、manifest.json、joint/first_update.json、joint/selection.json、joint/best.pt。', '',
        '```bash', 'uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v8_branches',
        'uv run python -m workspace.kr_time_budget.pipeline.report_v8_branches', '```', '']
    (ROOT/'双分支实验报告.md').write_text('\n'.join(lines))
    print(json.dumps({key: {k: m[k] for k in ['overall.auc', 'user_gauc', 'logloss']} for key, m in summary.items()}, indent=2))
    print(ROOT/'双分支实验报告.md')


if __name__ == '__main__':
    main()

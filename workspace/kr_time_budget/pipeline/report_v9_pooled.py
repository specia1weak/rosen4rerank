"""Audit Argon 10 full-data seed pairs; report Pooled probability metrics first."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
from workspace.kr_time_budget.pipeline.run_exit_v9_pooled import ROOT, EXPERIMENTS, calibration

REPORT_DIR = ROOT/'回复/10决定'


def paired_logloss(left, right):
    y = left['y_continue'].to_numpy()
    p = np.clip(left['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
    q = np.clip(right['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
    delta = -y*np.log(p)-(1-y)*np.log1p(-p)+y*np.log(q)+(1-y)*np.log1p(-q)
    users = left.select('user_id_raw').with_columns(pl.Series('delta', delta)).group_by('user_id_raw').agg(
        pl.len().alias('n'), pl.col('delta').sum()).sort('user_id_raw')
    n, d = users['n'].to_numpy(), users['delta'].to_numpy()
    indices = np.random.default_rng(2026).integers(0, len(n), (2000, len(n)))
    return {'float64_logloss_delta': float(delta.mean()),
        'user_cluster_bootstrap_95': np.quantile(d[indices].sum(1)/n[indices].sum(1), [.025, .975]).tolist(),
        'resamples': 2000, 'bootstrap_seed': 2026, 'conditional_on_training_seed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'runs/exit_v9_pooled')
    args = parser.parse_args()
    run = args.run
    cfg = json.loads((run/'config.json').read_text())
    assert not cfg['smoke'] and cfg['training_strategy'] == 'joint'
    assert cfg['primary_metrics'] == ['overall.auc', 'logloss'] and cfg['auxiliary_metric'] == 'intra_step_gauc'
    assert cfg['auxiliary_loss'] is None and cfg['warmup'] is None
    assert not cfg['frozen_budget'] and not cfg['detach_gamma']
    for filename, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256((ROOT/'data/v3/manifest.json').read_bytes()).hexdigest() == cfg['dataset_manifest_sha256']
    summary = json.loads((run/'summary.json').read_text())
    assert set(summary) == {f'{seed}/{name}' for seed in cfg['seeds'] for name in cfg['experiments']}
    expected_test = pl.read_parquet(ROOT/'data/v3/test.parquet', columns=['request_id', 'y_continue', 'user_id_raw'])
    frames, manifests, clocks, calibrations = {}, {}, {}, {}
    audit = {'models': {}, 'aggregated': {}, 'paired': {}, 'source_hashes': {
        str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    for key, metric in summary.items():
        out = run/key
        assert all(metric[k] for k in ['full_train', 'full_valid', 'full_test', 'checkpoint_reload_predictions_exact'])
        assert metric['optimizer_steps_executed'] == 421
        pred = pl.read_parquet(out/'predictions.parquet')
        assert pred.height == pred['request_id'].n_unique() == 48898
        for name in expected_test.columns:
            assert pred[name].equals(expected_test[name])
        recomputed = scores(pred)
        for name in ['overall.auc', 'logloss']:
            assert abs(recomputed[name]-metric[name]) < 1e-12
        for name in ['intra_step_gauc', 'user_gauc']:
            assert abs(recomputed[name]['auc']-metric[name]['auc']) < 1e-12
            for count in ['eligible_groups', 'eligible_rows', 'total_groups']:
                assert recomputed[name][count] == metric[name][count]
        cal = calibration(pred)
        assert cal == json.loads((out/'calibration.json').read_text())
        updates = json.loads((out/'joint/first_update.json').read_text())
        assert all(v['gradient_norm_after_clipping'] > 0 and v['parameter_max_abs_update'] > 0 for v in updates.values())
        progress = [json.loads(row) for row in (out/'joint/progress.jsonl').read_text().splitlines()]
        assert [r['step'] for r in progress] == [0, 106, 211, 421]
        assert min(progress, key=lambda r: r['valid']['overall']['logloss']) == metric['selected']
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['initial_all_tensors_exact']
        if key.split('/')[-1] != 'fusion':
            assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
            assert (pred['gamma']-pred['gamma_time']*pred['gamma_step']).abs().max() == 0
            clocks[key] = json.loads((out/'final_diagnostics.json').read_text())
            assert clocks[key]['clock_diagnostics']['logS_increases'] == 0
            for clock in ['time', 'step']:
                if not clocks[key]['clocks'][clock]['capped']:
                    assert clocks[key]['clocks'][clock]['hard_clamped_fraction'] == 0
        manifests[key], frames[key], calibrations[key] = manifest, pred, cal
        audit['models'][key] = {'coverage_and_metrics_exact': True, 'selection': metric['selected'],
                               'first_update': updates, 'manifest': manifest}
    for seed in cfg['seeds']:
        states = {manifests[f'{seed}/{n}']['initial_sha256'] for n in cfg['experiments']}
        orders = {manifests[f'{seed}/{n}']['train_order_sha256'] for n in cfg['experiments']}
        assert len(states) == len(orders) == 1
    for name, old_run in [('capped', 'exit_v8_branches/2026/flexible_shape'), ('fusion', 'exit_v8_branches/2026/fusion')]:
        key = f'2026/{name}'
        if key in frames:
            old = pl.read_parquet(ROOT/f'runs/{old_run}/predictions.parquet')
            assert frames[key]['prediction'].equals(old['prediction'])
            audit[f'{name}_2026_full_test_predictions_reproduce_v8_exact'] = True
    for name in cfg['experiments']:
        vals = {key: np.array([summary[f'{seed}/{name}'][key] for seed in cfg['seeds']]) for key in ['overall.auc', 'logloss']}
        vals['intra_step_gauc'] = np.array([summary[f'{seed}/{name}']['intra_step_gauc']['auc'] for seed in cfg['seeds']])
        for metric in ['brier', 'ece20_fixed_width', 'mean_prediction_minus_observed']:
            vals[metric] = np.array([calibrations[f'{seed}/{name}'][metric] for seed in cfg['seeds']])
        audit['aggregated'][name] = {k: dict(mean=float(v.mean()), sample_std=float(v.std(ddof=1)) if len(v) > 1 else None,
                                           values=v.tolist()) for k, v in vals.items()}
    pairs = [('minute_uncapped', 'capped'), ('both_uncapped', 'minute_uncapped'), ('both_uncapped', 'capped'),
             ('capped', 'fusion'), ('minute_uncapped', 'fusion'), ('both_uncapped', 'fusion')]
    for left, right in pairs:
        if left not in cfg['experiments'] or right not in cfg['experiments']:
            continue
        rows = []
        for seed in cfg['seeds']:
            a, b = f'{seed}/{left}', f'{seed}/{right}'
            result = paired_logloss(frames[a], frames[b])
            result.update(seed=seed, auc_delta=summary[a]['overall.auc']-summary[b]['overall.auc'],
                          logloss_delta=summary[a]['logloss']-summary[b]['logloss'],
                          intra_step_gauc_delta=summary[a]['intra_step_gauc']['auc']-summary[b]['intra_step_gauc']['auc'])
            rows.append(result)
        audit['paired'][f'{left} - {right}'] = rows
    (run/'audit.json').write_text(json.dumps(audit, indent=2))
    rel = run.relative_to(ROOT)
    def mean_std(name, metric):
        d = audit['aggregated'][name][metric]
        return f"{d['mean']:.6f} ± {d['sample_std']:.6f}" if d['sample_std'] is not None else f"{d['mean']:.6f}"
    lines = ['# Argon 10：Pooled AUC / LogLoss主线实验', '',
        '本轮主线改为继续概率的Pooled AUC与LogLoss，Intra-Step GAUC作为辅助。User-GAUC只保留在原始指标中，不用于Loss、调参或checkpoint选择。', '',
        f"完成{len(summary)}组完整train/valid/test：{len(cfg['experiments'])}配置×{len(cfg['seeds'])}seed，均只用一个乘积BCE联合更新两塔。", '',
        '## 跨seed结果', '', '| 配置 | Pooled AUC ↑ | LogLoss ↓ | Intra-Step GAUC ↑ |',
        '| --- | --- | --- | --- |']
    for name in cfg['experiments']:
        lines.append(f"| {name} | {mean_std(name, 'overall.auc')} | {mean_std(name, 'logloss')} | {mean_std(name, 'intra_step_gauc')} |")
    lines += ['', f"seed={cfg['seeds']}；±是seed间样本标准差，不是置信区间。未挑选最优seed；各seed模型均按完整valid LogLoss选择checkpoint。", '',
              '## 配对改动判断', '']
    for left, right in pairs:
        key = f'{left} - {right}'
        if key not in audit['paired']:
            continue
        rows = audit['paired'][key]
        da = float(np.mean([r['auc_delta'] for r in rows]))
        dl = float(np.mean([r['logloss_delta'] for r in rows]))
        improve = sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows)
        verdict = '观察到一致正信号' if improve == len(rows) else ('未观察到同时改善' if improve == 0 else 'seed间结果混合')
        lines.append(f"- {left} − {right}：平均AUC差{da:+.6f}、LogLoss差{dl:+.6f}；{improve}/{len(rows)}个seed同时改善两项，{verdict}。")
    lines += ['', '结论等级：解除尺度截断在本次三seed、单epoch口径下为负信号。保留capped作为双塔主线的现有对照，Fusion继续作为Pooled基准；不将解除截断配置替换为默认。', '',
        '上述判断限于本次固定训练口径，不将解除硬截断本身等同于性能提升，也不声称已证明分钟时钟饱和是历史差距的唯一原因。', '',
        '## 逐seed结果', '', '| seed | 配置 | AUC | LogLoss | Intra-Step GAUC | valid所选步 |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for seed in cfg['seeds']:
        for name in cfg['experiments']:
            m = summary[f'{seed}/{name}']
            lines.append(f"| {seed} | {name} | {m['overall.auc']:.6f} | {m['logloss']:.6f} | {m['intra_step_gauc']['auc']:.6f} | {m['selected']['step']} |")
    example = summary[f"{cfg['seeds'][0]}/{next(iter(cfg['experiments']))}"]['intra_step_gauc']
    lines += ['', f"Intra-Step GAUC按精确req_idx_raw分组，剔除只有一种标签的组后按组样本数加权：有效{example['eligible_groups']}组/{example['total_groups']}组，覆盖{example['eligible_rows']:,}/48,898条。各配置标签和覆盖一致，尾部缺少两类标签的组不参与。", '',
        '## 校准诊断', '', '| 配置 | Brier ↓ | ECE20 ↓ | 平均预测−真实继续率 |',
        '| --- | --- | --- | --- |']
    for name in cfg['experiments']:
        lines.append(f"| {name} | {mean_std(name, 'brier')} | {mean_std(name, 'ece20_fixed_width')} | {mean_std(name, 'mean_prediction_minus_observed')} |")
    lines += ['', 'ECE使用预先固定的20个等宽概率桶，是补充诊断；更低LogLoss/Brier是本次测试集上的概率评分结果，不能独自证明模型在所有状态上均已校准。', '',
        f"### 固定屏号桶：seed{cfg['seeds'][0]}", '', '| 屏号 | 样本数 | 真实继续率 | capped预测 | minute_uncapped预测 | both_uncapped预测 | Fusion预测 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    seed = cfg['seeds'][0]
    if set(cfg['experiments']) == set(EXPERIMENTS):
        by_step = {n: calibrations[f'{seed}/{n}']['by_step'] for n in cfg['experiments']}
        for i, row in enumerate(by_step['capped']):
            values = ' | '.join(f"{by_step[n][i]['prediction_mean']:.6f}" for n in ['capped', 'minute_uncapped', 'both_uncapped', 'fusion'])
            lines.append(f"| {row['step']} | {row['rows']} | {row['observed_rate']:.6f} | {values} |")
    lines += ['', '## 尺度与梯度：解除上界之后', '',
        '| seed | 配置 | 时钟 | lambda均值 | lambda最大值 | raw超过旧上界比例 | 超界且尺度梯度非零条数 | 硬截断比例 | gamma分量均值/std |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for key, d in clocks.items():
        seed, name = key.split('/')
        for clock in ['time', 'step']:
            c = d['clocks'][clock]
            g = d['components']['gamma_time' if clock == 'time' else 'gamma_step']
            lines.append(f"| {seed} | {name} | {clock} | {c['scale']['mean']:.3f} | {c['scale']['max']:.3f} | {c['above_old_upper_fraction']:.2%} | {c['above_old_upper_with_nonzero_derivative']} | {c['hard_clamped_fraction']:.2%} | {g['mean']:.6f} / {g['std']:.6f} |")
    lines += ['', '未截断配置未新增任何尺度上界；若学习到很大的尺度，也如实保留。超界非零梯度直接从单时钟条件gamma对raw_scale的导数测量。初始化和最终lambda/shape分布、gamma分布、时钟回退及累计S单调性均保存于各配置diagnostics文件。', '',
        '观测到的变化是：解除分钟截断后，模型继续把分钟lambda推高，三个seed的均值约为5.9万至113万分钟，而分钟gamma均值仍约0.998–0.999。它确实越过了旧上界且恢复了该区域梯度，但没有自动恢复有益的分钟时钟贡献。预算尺度本身不能解释为已识别出的真实用户预算，尤其是在只监督最终乘积时。', '',
        '## 数据与训练口径', '',
        'KR 1K data/v3：train为20220408–20220506，1,721,745条/1000用户；valid为20220507，57,872条/894用户；test为20220508，48,898条/855用户。固定54字段、schema与先验，保持train物理shuffle顺序；valid/test不shuffle；只在读取时增加原始分钟别名，不重建数据。预算Omni视图34字段、内容48字段，Fusion54字段。', '',
        '每配置每seed完整train 1epoch、421次更新，batch4096，无累积，CPU4线程。一个AdamW：预算lr=.005、内容/Fusion lr=.001，普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减，无scheduler/EMA。shape下界0.3、上界5，内容bias3.0。只有两处尺度截断开关变化，网络大小及输入路由不变。', '',
        '每seed各配置完整初始state逐tensor相同。step0/106/211/421评估完整valid，按LogLoss选checkpoint，之后完整test；首步两塔及各自用户Embedding均有非零梯度与真实参数更新；未冻结或detach预算。所有模型checkpoint独立重载，全部test预测逐值相同，乘积公式、标签及用户对齐原test。seed2026的capped/Fusion逐值复现上一轮完整test预测。', '',
        '56项实验测试通过，包含原截断模型与v8的预测/梯度逐值回归、解除上下截断后梯度恢复、全部配置共有初始化、单一联合Loss更新两塔、DFR在t=0和长时钟的有限梯度。', '',
        '用户配对bootstrap LogLoss区间（2000次）见audit.json；区间条件于各训练seed，不能代替跨seed复验。历史test已经多轮查看，本轮仍属探索，不能据此确认下游策略的反事实价值或在线收益。', '',
        '## 后续方向', '',
        '执行第10份计划提出的另一候选时，应单独验证K=2分钟混合威布尔，正确计算混合S(t+1)/S(t)，不能把两个条件gamma直接做固定权重平均；复用本轮capped与Fusion，保持相同数据、训练量和指标口径。若要区分单epoch收敛不足，应先做固定架构的训练周期对照。上述仅为下一轮建议，本轮没有把混合结构或额外训练周期混入解除截断实验。', '',
        '## 复现与证据', '',
        '- [预先定义的实验说明](Pooled主线实验说明.md) · [模型](../../pipeline/time_budget_v9.py) · [训练入口](../../pipeline/run_exit_v9_pooled.py) · [审计代码](../../pipeline/report_v9_pooled.py)。',
        f'- [训练日志](../../{rel}/logs/train.log) · [配置](../../{rel}/config.json) · [数据覆盖](../../{rel}/coverage.json) · [全部指标含诊断User-GAUC和分域](../../{rel}/summary.json) · [独立审计与配对区间](../../{rel}/audit.json) · [测试](../../{rel}/logs/model-tests.log)。',
        f'- 每个配置证据：`{rel}/<seed>/<配置>/`中的manifest.json、initial_diagnostics.json、final_diagnostics.json、calibration.json、predictions.parquet，以及joint/first_update.json、joint/selection.json、joint/best.pt。', '',
        '```bash', 'uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v9_pooled',
        'uv run python -m workspace.kr_time_budget.pipeline.report_v9_pooled', '```', '']
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR/'Pooled主线实验报告.md').write_text('\n'.join(lines))
    print(json.dumps(audit['aggregated'], indent=2))
    print(REPORT_DIR/'Pooled主线实验报告.md')


if __name__ == '__main__':
    main()

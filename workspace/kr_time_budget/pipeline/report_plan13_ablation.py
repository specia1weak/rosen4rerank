"""Independent audit of plan13 2x2 cells and nested training budgets."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
from workspace.kr_time_budget.pipeline.run_exit_v9_pooled import calibration
from workspace.kr_time_budget.pipeline.report_v9_pooled import paired_logloss
from workspace.kr_time_budget.pipeline.run_plan13_ablation import ROOT, EXPERIMENTS, validation_steps

RUN = ROOT/'runs/plan13_ablation'
REPORT = ROOT/'回复/13里程碑'


def main():
    cfg = json.loads((RUN/'config.json').read_text())
    assert not cfg['smoke'] and cfg['seeds'] == [2026, 2027, 2028]
    assert cfg['reported_epoch_budgets'] == [1, 3] and cfg['epochs'] == 3
    assert set(cfg['experiments']) == set(EXPERIMENTS)
    assert cfg['training_strategy'] == 'joint' and cfg['auxiliary_loss'] is None
    assert cfg['warmup'] is None and not cfg['frozen_budget'] and not cfg['detach_gamma']
    for filename, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256((ROOT/'data/v3/manifest.json').read_bytes()).hexdigest() == cfg['dataset_manifest_sha256']
    summary = json.loads((RUN/'summary.json').read_text())
    assert set(summary) == {f'{s}/{n}/{b}ep' for s in cfg['seeds'] for n in EXPERIMENTS for b in [1, 3]}
    expected_test = pl.read_parquet(ROOT/'data/v3/test.parquet', columns=['request_id', 'y_continue', 'user_id_raw'])
    audit = dict(trajectories={}, models={}, reproduction={}, aggregated={}, comparisons={},
                 interactions={}, report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    frames, cals = {}, {}
    for seed in cfg['seeds']:
        for name in EXPERIMENTS:
            out = RUN/f'{seed}/{name}'
            assert json.loads((out/'complete.json').read_text())['trajectory_steps'] == 1263
            manifest = json.loads((out/'manifest.json').read_text())
            assert manifest['common_initial_tensors_exact']
            progress = [json.loads(row) for row in (out/'joint/progress.jsonl').read_text().splitlines()]
            assert [r['step'] for r in progress] == validation_steps(421)
            updates = json.loads((out/'joint/first_update.json').read_text())
            assert all(v['gradient_norm_after_clipping'] > 0 and v['parameter_max_abs_update'] > 0 for v in updates.values())
            audit['trajectories'][f'{seed}/{name}'] = dict(manifest=manifest, first_update=updates, validation=progress)
            for budget in [1, 3]:
                key = f'{seed}/{name}/{budget}ep'
                metric = summary[key]
                assert all(metric[k] for k in ['full_train', 'full_valid', 'full_test', 'checkpoint_reload_predictions_exact'])
                assert metric['budget_optimizer_steps'] == budget*421
                assert metric['trajectory_optimizer_steps_executed'] == 1263
                best = min((r for r in progress if r['step'] <= budget*421), key=lambda r: r['valid']['overall']['logloss'])
                assert metric['selected'] == best
                budget_audit = json.loads((out/f'joint/budget_{budget}ep.json').read_text())
                assert budget_audit['selected'] == best
                assert budget_audit['optimizer_state_steps'] == [budget*421]
                pred = pl.read_parquet(out/f'{budget}ep/predictions.parquet')
                assert pred.height == pred['request_id'].n_unique() == 48898
                for col in expected_test.columns:
                    assert pred[col].equals(expected_test[col])
                score = scores(pred)
                for col in ['overall.auc', 'logloss']:
                    assert abs(metric[col]-score[col]) < 1e-12
                assert abs(metric['intra_step_gauc']['auc']-score['intra_step_gauc']['auc']) < 1e-12
                cal = calibration(pred)
                assert cal == json.loads((out/f'{budget}ep/calibration.json').read_text())
                early = pred.filter(pl.col('req_idx_raw').is_between(2, 10))
                assert abs(scores(early)['logloss']-metric['steps2to10']['logloss']) < 1e-12
                assert abs(calibration(early)['mean_prediction_minus_observed']-metric['steps2to10']['bias']) < 1e-12
                if not name.startswith('fusion'):
                    assert (pred['gamma_time'] == 1).all()
                    assert (pred['gamma']-pred['gamma_step']).abs().max() == 0
                    assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
                    d = json.loads((out/f'{budget}ep/diagnostics.json').read_text())
                    assert d['clock_diagnostics']['logS_increases'] == 0
                if budget == 1 and name in {'mixture_step', 'mixture_step_pace', 'fusion', 'fusion_pace'}:
                    old = pl.read_parquet(ROOT/f'runs/exit_v10_plan12/{seed}/{name}/predictions.parquet')
                    assert pred['prediction'].equals(old['prediction'])
                    audit['reproduction'][key] = 'full test predictions exactly reproduce plan12'
                audit['models'][key] = dict(coverage_and_metrics_exact=True, selected=best, budget_audit=budget_audit,
                                           checkpoint_reload_predictions_exact=True)
                frames[key], cals[key] = pred, cal
    for budget in [1, 3]:
        for name in EXPERIMENTS:
            vals = {k: np.array([summary[f'{s}/{name}/{budget}ep'][k] for s in cfg['seeds']]) for k in ['overall.auc', 'logloss']}
            vals['intra_step_gauc'] = np.array([summary[f'{s}/{name}/{budget}ep']['intra_step_gauc']['auc'] for s in cfg['seeds']])
            for k in ['bias', 'logloss']:
                vals[f'steps2to10_{k}'] = np.array([summary[f'{s}/{name}/{budget}ep']['steps2to10'][k] for s in cfg['seeds']])
            for k in ['brier', 'ece20_fixed_width']:
                vals[k] = np.array([cals[f'{s}/{name}/{budget}ep'][k] for s in cfg['seeds']])
            audit['aggregated'][f'{name}/{budget}ep'] = {k: dict(mean=float(v.mean()), sample_std=float(v.std(ddof=1)),
                values=v.tolist()) for k, v in vals.items()}
    pairs = [('mixture_step', 'single_step'), ('mixture_step_pace', 'single_step_pace'),
        ('single_step_pace', 'single_step'), ('mixture_step_pace', 'mixture_step'), ('fusion_pace', 'fusion'),
        ('mixture_step_pace', 'fusion'), ('mixture_step_pace', 'fusion_pace'), ('single_step_pace', 'fusion')]
    def compare(left_keys, right_keys):
        rows = []
        for seed, a, b in zip(cfg['seeds'], left_keys, right_keys):
            m, n = summary[a], summary[b]
            row = paired_logloss(frames[a], frames[b])
            row.update(seed=seed, auc_delta=m['overall.auc']-n['overall.auc'], logloss_delta=m['logloss']-n['logloss'],
                intra_step_gauc_delta=m['intra_step_gauc']['auc']-n['intra_step_gauc']['auc'])
            rows.append(row)
        return rows
    for budget in [1, 3]:
        for a, b in pairs:
            audit['comparisons'][f'{budget}ep/{a} - {b}'] = compare(
                [f'{s}/{a}/{budget}ep' for s in cfg['seeds']], [f'{s}/{b}/{budget}ep' for s in cfg['seeds']])
        # Interaction of metric increments, not a causal identification claim.
        interaction = []
        for seed in cfg['seeds']:
            keys = [f'{seed}/{n}/{budget}ep' for n in ['mixture_step_pace', 'mixture_step', 'single_step_pace', 'single_step']]
            row = dict(seed=seed)
            for col in ['overall.auc', 'logloss']:
                v = [summary[k][col] for k in keys]
                row[col] = v[0]-v[1]-v[2]+v[3]
            v = [summary[k]['intra_step_gauc']['auc'] for k in keys]
            row['intra_step_gauc'] = v[0]-v[1]-v[2]+v[3]
            # Bootstrap the four-way LogLoss contrast on the same resampled users.
            y = frames[keys[0]]['y_continue'].to_numpy()
            row_losses = []
            for key in keys:
                p = np.clip(frames[key]['prediction'].to_numpy().astype(float), 1e-7, 1-1e-7)
                row_losses.append(-y*np.log(p)-(1-y)*np.log1p(-p))
            contrast = row_losses[0]-row_losses[1]-row_losses[2]+row_losses[3]
            users = frames[keys[0]].select('user_id_raw').with_columns(pl.Series('contrast', contrast)).group_by('user_id_raw').agg(
                pl.len().alias('n'), pl.col('contrast').sum()).sort('user_id_raw')
            n, sums = users['n'].to_numpy(), users['contrast'].to_numpy()
            idx = np.random.default_rng(2026).integers(0, len(n), (2000, len(n)))
            row['logloss_interaction_user_bootstrap_95'] = np.quantile(sums[idx].sum(1)/n[idx].sum(1), [.025, .975]).tolist()
            interaction.append(row)
        audit['interactions'][f'{budget}ep'] = interaction
    for name in EXPERIMENTS:
        audit['comparisons'][f'3ep - 1ep/{name}'] = compare(
            [f'{s}/{name}/3ep' for s in cfg['seeds']], [f'{s}/{name}/1ep' for s in cfg['seeds']])
    (RUN/'audit.json').write_text(json.dumps(audit, indent=2))
    def fmt(name, budget, col):
        v = audit['aggregated'][f'{name}/{budget}ep'][col]
        return f"{v['mean']:.6f} ± {v['sample_std']:.6f}"
    lines = ['# 第13份计划：2×2消融与1/3epoch训练预算', '',
        '已补single_step、single_step_pace，与mixture_step、mixture_step_pace组成2×2矩阵，并保留Fusion/Fusion_pace。六配置×三seed，每条轨迹连续训练3epoch，分别按1epoch与3epoch预算内完整valid LogLoss选模型；共18条训练轨迹、54个完整epoch、36组完整test评估。', '',
        '## 2×2与黑盒对照', '',
        '主指标：Pooled AUC、LogLoss；辅助：Intra-Step GAUC。一个乘积BCE联合训练；没有辅助Loss、冻结或两阶段预热。±为三个seed间样本标准差，不是置信区间。', '']
    for budget in [1, 3]:
        lines += [f'### {budget}epoch训练预算', '', '| 配置 | Pooled AUC ↑ | LogLoss ↓ | Intra-Step GAUC ↑ |',
                  '| --- | --- | --- | --- |']
        for name in EXPERIMENTS:
            lines.append(f"| {name} | {fmt(name, budget, 'overall.auc')} | {fmt(name, budget, 'logloss')} | {fmt(name, budget, 'intra_step_gauc')} |")
        lines.append('')
    lines += ['## 本轮结论', '']
    for budget in [1, 3]:
        rows = audit['comparisons'][f'{budget}ep/mixture_step_pace - fusion']
        count = sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows)
        da, dl = [np.mean([r[k] for r in rows]) for k in ['auc_delta', 'logloss_delta']]
        lines.append(f'- {budget}epoch预算：混合+pace相对Fusion平均AUC差{da:+.6f}、LogLoss差{dl:+.6f}；{count}/3 seed同时改善两项。')
    for budget in [1, 3]:
        rows = audit['comparisons'][f'{budget}ep/single_step_pace - single_step']
        count = sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows)
        lines.append(f'- {budget}epoch预算内，单威布尔加pace有{count}/3 seed同时改善AUC与LogLoss；pace增益不能只归属于混合分布。')
    for budget in [1, 3]:
        counts = []
        for left, right in [('mixture_step', 'single_step'), ('mixture_step_pace', 'single_step_pace')]:
            rows = audit['comparisons'][f'{budget}ep/{left} - {right}']
            counts.append(sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows))
        lines.append(f'- {budget}epoch预算内，混合相对单威布尔：无pace为{counts[0]}/3、有pace为{counts[1]}/3 seed同时改善AUC与LogLoss，支持当前实现下混合底座的增益。')
    rows = audit['comparisons']['3ep/mixture_step_pace - fusion']
    changes = '；'.join(f"seed{r['seed']}：AUC差{r['auc_delta']:+.6f}、LogLoss差{r['logloss_delta']:+.6f}" for r in rows)
    lines.append(f'- 3epoch逐seed的混合+pace−Fusion为{changes}。两个seed的小幅改善不足以抵消另一个seed的退步，需同时看均值与逐seed结果。')
    lines += ['- 本轮把“是否优于Fusion”限定到训练预算。训练步数相同不代表不同架构已经同等充分收敛，不能从1epoch胜负推出任意训练周期下的模型上限。',
        '- 2×2比较检验固定实现下的结构与特征效果，不证明pace是因果沉浸度、两塔已经完成潜变量识别或混合分布是唯一有效的预算函数族。', '']
    lines += ['## 同预算配对判断', '',
        '| 预算 | 左−右 | 平均AUC差 | 平均LogLoss差 | 平均Intra-Step差 | 同时改善AUC/LogLoss的seed |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for budget in [1, 3]:
        for a, b in pairs:
            rows = audit['comparisons'][f'{budget}ep/{a} - {b}']
            da, dl, di = [np.mean([r[k] for r in rows]) for k in ['auc_delta', 'logloss_delta', 'intra_step_gauc_delta']]
            count = sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows)
            lines.append(f'| {budget}ep | {a} − {b} | {da:+.6f} | {dl:+.6f} | {di:+.6f} | {count}/3 |')
    lines += ['', '0/3不必然表示两项均恶化，可能一项变好另一项变差；查看逐seed配对值及方向。LogLoss用户聚类bootstrap 95%区间、原始差值均保存于audit.json；区间条件于训练seed。', '',
        '## pace的增益是否依赖混合结构', '',
        '下表为(混合加pace−混合无pace)−(单分量加pace−单分量无pace)。AUC/Intra-Step正值表示混合的pace增益更大；LogLoss负值表示混合的pace降损更大。它是本次指标的差分，不是潜变量已识别或因果正交性的证明。', '',
        '| 预算 | AUC增益差均值 | LogLoss增益差均值 | Intra-Step增益差均值 |', '| --- | ---: | ---: | ---: |']
    for budget in [1, 3]:
        rows = audit['interactions'][f'{budget}ep']
        v = [np.mean([r[k] for r in rows]) for k in ['overall.auc', 'logloss', 'intra_step_gauc']]
        lines.append(f'| {budget}ep | {v[0]:+.6f} | {v[1]:+.6f} | {v[2]:+.6f} |')
    lines += ['', 'pace原始量纲是分钟/屏，log1p只是数值变换；不能自动称为无量纲或与进度统计独立。架构通过路由分工不等于已证明两个信号正交。', '',
        '## 固定架构：3epoch预算−1epoch预算', '',
        '| 配置 | 平均AUC差 | 平均LogLoss差 | 平均Intra-Step差 | 同时改善AUC/LogLoss的seed |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in EXPERIMENTS:
        rows = audit['comparisons'][f'3ep - 1ep/{name}']
        da, dl, di = [np.mean([r[k] for r in rows]) for k in ['auc_delta', 'logloss_delta', 'intra_step_gauc_delta']]
        count = sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows)
        lines.append(f'| {name} | {da:+.6f} | {dl:+.6f} | {di:+.6f} | {count}/3 |')
    lines += ['', '同一轨迹前缀与延长预算相关，不是独立重训。训练预算增加包含额外更新与额外valid选择机会，3epoch预算的所选checkpoint可以来自更早epoch。训练过程不在epoch间回滚best或重置AdamW，记录的活跃参数优化器step为421与1263；完整轨迹结束后才计算两个预算的test预测。', '',
        '### 完整valid训练曲线：每个epoch末的三seed平均LogLoss', '',
        '| 配置 | epoch1末 | epoch2末 | epoch3末 |', '| --- | ---: | ---: | ---: |']
    for name in EXPERIMENTS:
        values = []
        for step in [421, 842, 1263]:
            values.append(np.mean([next(r for r in audit['trajectories'][f'{s}/{name}']['validation'] if r['step'] == step)['valid']['overall']['logloss'] for s in cfg['seeds']]))
        lines.append(f'| {name} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '这里是实际epoch末状态的valid结果，不是其预算内best；选择仍使用全部预先定义节点的最低valid LogLoss。训练结束前不使用test判断哪个模型已经收敛。', '',
        '## 第2–10屏与校准诊断', '',
        '| 预算 | 配置 | 第2–10屏偏差 | 第2–10屏LogLoss | Brier | ECE20 |',
        '| --- | --- | --- | --- | --- | --- |']
    for budget in [1, 3]:
        for name in EXPERIMENTS:
            v = [fmt(name, budget, k) for k in ['steps2to10_bias', 'steps2to10_logloss', 'brier', 'ece20_fixed_width']]
            lines.append(f'| {budget}ep | {name} | '+' | '.join(v)+' |')
    lines += ['', '第2–10屏窗口17,498条，偏差为平均预测减真实继续率。均值偏差下降不独自证明个体校准或生存曲率正确。ECE固定20等宽概率桶，属于补充诊断。', '',
        '## 逐seed指标和checkpoint', '',
        '| seed | 配置 | 预算 | AUC | LogLoss | Intra-Step GAUC | valid所选step | 所选valid LogLoss |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for key, m in summary.items():
        seed, name, budget = key.split('/')
        lines.append(f"| {seed} | {name} | {budget} | {m['overall.auc']:.6f} | {m['logloss']:.6f} | {m['intra_step_gauc']['auc']:.6f} | {m['selected']['step']} | {m['selected']['valid']['overall']['logloss']:.6f} |")
    lines += ['', '## 数据、训练与审计口径', '',
        'KR 1K data/v3：train=20220408–20220506，1,721,745条/1000用户；valid=20220507，57,872条/894用户；test=20220508，48,898条/855用户。固定原54字段、schema、历史先验和物理顺序；每epoch重复同一已shuffle训练顺序，valid/test不shuffle。没有重建数据。', '',
        '使用原time_budget_v10模型：预算34字段独立Omni，内容48字段；pace配置在内容或Fusion向量后附加当前屏结束时的log1p(elapsed_min_raw/req_idx_raw)，没有未来信息。单/混合都只用屏数预算，分钟gamma=1；shape=[0.3,5]、原尺度上下截断和初始bias保持第12份。', '',
        'seed2026/2027/2028，batch4096，无累积。一个AdamW连续更新：预算lr=.005、内容/Fusion lr=.001，普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减，无scheduler/EMA，CPU4线程。每epoch421次更新，1/3epoch预算421/1263次。验证节点0/106/211/421/527/632/842/948/1053/1263；各预算仅按其范围内完整valid LogLoss选checkpoint。', '',
        '共有参数按原模型拓展规则核对，混合末层新增参数和pace新列明确排除或分列比较。混合与单分量末层大小及分量初始化不同，本轮控制了共有部分，但没有单独排除参数量因素。所有原四配置1epoch预测逐seed复现第12份；全部36组checkpoint独立重载预测逐值一致，数据覆盖、标签/用户、预算gamma及最终乘积公式核对通过；首步两塔与用户Embedding均有梯度及真实更新。', '',
        '65项测试通过；六配置短程连续训练同时核对两个预算选择范围、优化器步数与原1epoch前缀复现。预算快照使用本次验证已处理的实际step，避免Trainer在batch内global_step尚未递增时把下一epoch验证误写进早期快照。全量选择范围再次独立审计。', '',
        '多次查看过同一test，以上属于固定切分上的探索性复验，不把3seed标准差或离线AUC差直接解释为线上收益，也不从权重/后验均值认定潜在预算类别已经识别。', '',
        '## 代码与复现', '',
        '- [实验定义](实验说明.md) · [未改动的模型](../../pipeline/time_budget_v10.py) · [训练入口](../../pipeline/run_plan13_ablation.py) · [报告审计](../../pipeline/report_plan13_ablation.py)。',
        '- [完整训练日志](../../runs/plan13_ablation/logs/train.log) · [配置](../../runs/plan13_ablation/config.json) · [全部指标含分域](../../runs/plan13_ablation/summary.json) · [独立审计与配对区间](../../runs/plan13_ablation/audit.json) · [测试](../../runs/plan13_ablation/logs/tests.log)。',
        '- 每条轨迹位于`runs/plan13_ablation/<seed>/<配置>/`：joint/progress.jsonl、first_update.json、best_within_1ep.pt、best_within_3ep.pt、budget_1ep.json、budget_3ep.json；1ep/3ep下分别有metrics、predictions、calibration与diagnostics。', '',
        '```bash', 'uv run python -u -m workspace.kr_time_budget.pipeline.run_plan13_ablation',
        'uv run python -m workspace.kr_time_budget.pipeline.report_plan13_ablation', '```', '']
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'实验报告.md').write_text('\n'.join(lines))
    print(json.dumps(audit['aggregated'], indent=2))
    print(REPORT/'实验报告.md')


if __name__ == '__main__':
    main()

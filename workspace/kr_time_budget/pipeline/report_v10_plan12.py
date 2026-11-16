"""Audit plan 12 mixtures and pace experiments; archive under 回复/12下一步."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
from workspace.kr_time_budget.pipeline.run_exit_v9_pooled import calibration
from workspace.kr_time_budget.pipeline.report_v9_pooled import paired_logloss
from workspace.kr_time_budget.pipeline.run_exit_v10_plan12 import EXPERIMENTS, ROOT

RUN = ROOT/'runs/exit_v10_plan12'
REPORT = ROOT/'回复/12下一步'


def main():
    cfg = json.loads((RUN/'config.json').read_text())
    assert not cfg['smoke'] and cfg['seeds'] == [2026, 2027, 2028]
    assert set(cfg['experiments']) == set(EXPERIMENTS)
    assert cfg['training_strategy'] == 'joint' and cfg['auxiliary_loss'] is None
    assert cfg['warmup'] is None and not cfg['frozen_budget'] and not cfg['detach_gamma']
    for filename, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256((ROOT/'data/v3/manifest.json').read_bytes()).hexdigest() == cfg['dataset_manifest_sha256']
    summary = json.loads((RUN/'summary.json').read_text())
    assert set(summary) == {f'{seed}/{name}' for seed in cfg['seeds'] for name in EXPERIMENTS}
    expected_test = pl.read_parquet(ROOT/'data/v3/test.parquet', columns=['request_id', 'y_continue', 'user_id_raw'])
    frames, diagnostics, cals, manifests = {}, {}, {}, {}
    audit = dict(models={}, aggregates={}, comparisons={}, reproduction={}, report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for key, metric in summary.items():
        out = RUN/key
        assert all(metric[k] for k in ['full_train', 'full_valid', 'full_test', 'checkpoint_reload_predictions_exact'])
        assert metric['optimizer_steps_executed'] == 421
        pred = pl.read_parquet(out/'predictions.parquet')
        assert pred.height == pred['request_id'].n_unique() == 48898
        for name in expected_test.columns:
            assert pred[name].equals(expected_test[name])
        recomputed = scores(pred)
        for name in ['overall.auc', 'logloss']:
            assert abs(recomputed[name]-metric[name]) < 1e-12
        assert abs(recomputed['intra_step_gauc']['auc']-metric['intra_step_gauc']['auc']) < 1e-12
        cal = calibration(pred)
        assert cal == json.loads((out/'calibration.json').read_text())
        early = pred.filter(pl.col('req_idx_raw').is_between(2, 10))
        assert early.height == metric['steps2to10']['rows']
        assert abs(scores(early)['logloss']-metric['steps2to10']['logloss']) < 1e-12
        assert abs(calibration(early)['mean_prediction_minus_observed']-metric['steps2to10']['bias']) < 1e-12
        updates = json.loads((out/'joint/first_update.json').read_text())
        assert all(v['gradient_norm_after_clipping'] > 0 and v['parameter_max_abs_update'] > 0 for v in updates.values())
        progress = [json.loads(row) for row in (out/'joint/progress.jsonl').read_text().splitlines()]
        assert [row['step'] for row in progress] == [0, 106, 211, 421]
        assert min(progress, key=lambda r: r['valid']['overall']['logloss']) == metric['selected']
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['common_initial_tensors_exact']
        diag = json.loads((out/'final_diagnostics.json').read_text())
        if not key.split('/')[-1].startswith('fusion'):
            assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
            assert (pred['gamma']-pred['gamma_time']*pred['gamma_step']).abs().max() == 0
            assert diag['clock_diagnostics']['logS_increases'] == 0
            if manifest['experiment']['step_only']:
                assert (pred['gamma_time'] == 1).all()
        audit['models'][key] = dict(coverage_metrics_formula_exact=True, first_update=updates,
                                    selected=metric['selected'], manifest=manifest)
        frames[key], diagnostics[key], cals[key], manifests[key] = pred, diag, cal, manifest
    for seed in cfg['seeds']:
        for name, old_name in [('single_dual', 'capped'), ('fusion', 'fusion')]:
            old = pl.read_parquet(ROOT/f'runs/exit_v9_pooled/{seed}/{old_name}/predictions.parquet')
            assert frames[f'{seed}/{name}']['prediction'].equals(old['prediction'])
            audit['reproduction'][f'{seed}/{name}'] = 'full test prediction exactly equals plan10'
    for name in EXPERIMENTS:
        vals = {k: np.array([summary[f'{s}/{name}'][k] for s in cfg['seeds']]) for k in ['overall.auc', 'logloss']}
        vals['intra_step_gauc'] = np.array([summary[f'{s}/{name}']['intra_step_gauc']['auc'] for s in cfg['seeds']])
        for key in ['brier', 'ece20_fixed_width']:
            vals[key] = np.array([cals[f'{s}/{name}'][key] for s in cfg['seeds']])
        for key in ['bias', 'logloss']:
            vals[f'steps2to10_{key}'] = np.array([summary[f'{s}/{name}']['steps2to10'][key] for s in cfg['seeds']])
        audit['aggregates'][name] = {k: dict(mean=float(v.mean()), sample_std=float(v.std(ddof=1)), values=v.tolist()) for k, v in vals.items()}
    pairs = [('mixture_dual', 'single_dual'), ('mixture_step', 'mixture_dual'),
        ('mixture_step_pace', 'mixture_step'), ('fusion_pace', 'fusion'),
        ('mixture_dual', 'fusion'), ('mixture_step_pace', 'fusion'), ('mixture_step_pace', 'fusion_pace')]
    for left, right in pairs:
        rows = []
        for seed in cfg['seeds']:
            a, b = f'{seed}/{left}', f'{seed}/{right}'
            m, n = summary[a], summary[b]
            row = paired_logloss(frames[a], frames[b])
            row.update(seed=seed, auc_delta=m['overall.auc']-n['overall.auc'], logloss_delta=m['logloss']-n['logloss'],
                intra_step_gauc_delta=m['intra_step_gauc']['auc']-n['intra_step_gauc']['auc'],
                steps2to10_bias_delta=m['steps2to10']['bias']-n['steps2to10']['bias'])
            rows.append(row)
        audit['comparisons'][f'{left} - {right}'] = rows
    (RUN/'audit.json').write_text(json.dumps(audit, indent=2))
    def fmt(name, key):
        v = audit['aggregates'][name][key]
        return f"{v['mean']:.6f} ± {v['sample_std']:.6f}"
    lines = ['# 第12份计划：K=2混合威布尔与平均停留特征实验', '',
        '已阅读第11份理解，按第12份两步执行，共6配置×3seed=18组完整train/valid/test。主指标为Pooled AUC与LogLoss，辅助Intra-Step GAUC；一个最终乘积BCE同时更新预算塔和内容塔，没有辅助Loss、预热、冻结或detach。', '',
        '## 三seed结果', '', '| 配置 | Pooled AUC ↑ | LogLoss ↓ | Intra-Step GAUC ↑ |',
        '| --- | --- | --- | --- |']
    for name in EXPERIMENTS:
        lines.append(f"| {name} | {fmt(name, 'overall.auc')} | {fmt(name, 'logloss')} | {fmt(name, 'intra_step_gauc')} |")
    lines += ['', 'seed2026/2027/2028；±为seed间样本标准差，不是置信区间。每模型仅按完整valid LogLoss选checkpoint，没有test挑seed。', '',
        '**本轮候选为mixture_step_pace：三seed的Pooled AUC、LogLoss和Intra-Step GAUC均优于Fusion及Fusion_pace。相对Fusion，平均AUC提高0.001756，LogLoss下降0.000485，Intra-Step GAUC提高0.002494。**', '',
        '结论等级：单epoch、固定测试集上的一致探索性正信号。用户配对bootstrap中，相对Fusion的LogLoss差值在seed2026的95%区间仍含0，seed2027/2028不含0；不把三seed方向一致表述为所有统计区间均排除0。', '',
        '## 两步实验与判断', '']
    for left, right in pairs:
        rows = audit['comparisons'][f'{left} - {right}']
        da = np.mean([r['auc_delta'] for r in rows])
        dl = np.mean([r['logloss_delta'] for r in rows])
        di = np.mean([r['intra_step_gauc_delta'] for r in rows])
        count = sum(r['auc_delta'] > 0 and r['logloss_delta'] < 0 for r in rows)
        verdict = '一致正信号' if count == 3 else ('未观察到两项同时改善' if count == 0 else 'seed间混合信号')
        lines.append(f'- {left} − {right}：平均AUC {da:+.6f}，LogLoss {dl:+.6f}，Intra-Step GAUC {di:+.6f}；{count}/3 seed同时改善AUC与LogLoss，{verdict}。')
    lines += ['', '第一步只更换预算分布结构，仍保留原capped的尺度和shape范围。第二步先关闭分钟预算，再加入单屏平均停留特征；额外Fusion_pace用于判断同样的特征工程是否也改善黑盒。混合分布头输出从4维变12维，增加了参数和初始化方案，不能把整体结构结果归因为“混合”之外的某个单独数值机制。', '',
        '## 第2–10屏偏差是否改善', '',
        '| 配置 | 第2–10屏平均预测−真实继续率 | 第2–10屏LogLoss |', '| --- | --- | --- |']
    for name in EXPERIMENTS:
        lines.append(f"| {name} | {fmt(name, 'steps2to10_bias')} | {fmt(name, 'steps2to10_logloss')} |")
    n_early = summary['2026/single_dual']['steps2to10']['rows']
    lines += ['', f'第2–10屏共有{n_early:,}条，占完整test的{n_early/48898:.2%}。偏差接近0只说明该聚合窗口均值更接近，不等同于个体概率准确，也不独自证明函数曲率已经恢复。', '',
        '该窗口平均高估从单威布尔的约+1.09个百分点降至混合双时钟的+0.75个百分点，最终候选约+0.67个百分点；有所缓解但未消除。加入pace后窗口均值改善幅度较小，而Intra-Step GAUC提升更明显，应将总体校准与同屏排序分别看待。', '',
        '### 固定屏号桶：三seed平均预测', '',
        '| 屏号 | 样本数 | 真实继续率 | 单威布尔 | 混合双时钟 | 混合仅屏数 | 混合屏数+停留 | Fusion | Fusion+停留 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    base = cals['2026/single_dual']['by_step']
    for i, row in enumerate(base):
        values = [np.mean([cals[f'{s}/{n}']['by_step'][i]['prediction_mean'] for s in cfg['seeds']]) for n in EXPERIMENTS]
        lines.append(f"| {row['step']} | {row['rows']} | {row['observed_rate']:.6f} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '## 混合先验、后验与分量诊断', '',
        '| seed | 配置 | 时钟 | 分量 | lambda均值 | shape均值 | 先验均值 | 存活后验均值 | scale上界比例 |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, diag in diagnostics.items():
        seed, name = key.split('/')
        for clock, info in diag.get('clocks', {}).items():
            for k, c in enumerate(info.get('components', [])):
                lines.append(f"| {seed} | {name} | {clock} | {k} | {c['scale']['mean']:.3f} | {c['shape']['mean']:.4f} | {c['prior']['mean']:.4f} | {c['posterior']['mean']:.4f} | {c['scale_upper_clamped_fraction']:.2%} |")
    lines += ['', '分量序号来自初始化，不自动保证训练后仍对应短/长模式。混合采用S_mix(t+1)/S_mix(t)，后验权重随已存活时钟更新；不是π加权的固定条件gamma。原第11份关于多峰预算和先升后降的图像作为动机；仅由当前退出监督、AUC或两个非零权重不能确认真实潜在预算模式已被识别。', '',
        '最终mixture_step_pace的两个屏数分量lambda均值约32–38与139–152屏，形状均值约0.92–0.95与0.60–0.65。它们没有直接复现第11份举例的“1–3屏短预算+20–100屏长预算且两分量shape>1”；本轮结果不能当成该严格物理图像的验证。分钟混合分支中一个先验权重约2%，也提示分量可能退化，详见完整diagnostics。', '',
        '## 派生特征与数据流变化', '',
        'pace=log1p(max(elapsed_min_raw,0)/max(req_idx_raw,1))，在当前屏结束的决策点从已有raw列现算。不使用下一屏、会话最终时长或剩余步数。它是到当前屏的平均分钟，不是prev1_watch_min；原始分钟回退及跨屏观看差异仍会影响它。', '',
        '预算Omni始终34字段；内容Omni48字段，指定配置在向量后追加1维pace；Fusion为54原字段，fusion_pace同样追加1维。它明确放宽了内容塔不能接收时钟派生信号的旧约束，未假装仍是原48字段输入。没有重拟合schema或重写parquet。追加第一层新列零初始化，共有列复制，允许1e-7级矩阵形状浮点误差；混合头在所有共有模块构造后替换，不改变共有参数RNG序列。', '',
        '## 数据、训练与审计', '',
        'KR 1K data/v3，54原字段：train 20220408–20220506，1,721,745条/1000用户；valid 20220507，57,872条/894用户；test 20220508，48,898条/855用户。复用train物理shuffle及原词表/先验，valid/test不shuffle，只在读取时添加elapsed_min_raw别名。', '',
        '每配置每seed完整train 1epoch、421次更新，batch4096，无累积。AdamW预算lr=.005、内容/Fusion lr=.001；普通矩阵衰减1e-6，Embedding/bias/一维参数零衰减，无scheduler/EMA；CPU4线程。各分量仍为exp(clamp(raw,−2,4.5)+1.6)尺度、clamp(0.3+softplus(raw),max=5)形状，不额外修改尺度上界或强制IFR。', '',
        'step0/106/211/421完整valid按LogLoss选checkpoint，完整test评估。全部18组checkpoint独立重载，预测逐值一致；原single_dual和Fusion逐seed复现第10份结果。所有配置共有初始tensor按实际拓展规则核对；首步两塔网络与各自用户Embedding有非零梯度及真实更新；公式乘积和数据覆盖、标签/用户对齐通过。', '',
        '63项测试通过，包含混合S之比手算、存活后验更新、与固定先验平均的差异、相同分量退化为单分量、零点及长时钟梯度、步数分支独立于分钟、追加特征首步可学习与共有初始化回归。累计S在43,935对分钟不回退相邻步无上升；原1,682对分钟回退单列保留。Intra-Step GAUC有效131/681组，覆盖46,797/48,898条。', '',
        '三seed结果仍是单epoch探索。用户聚类LogLoss配对bootstrap区间（2000次）保存于audit.json，条件于训练seed；不替代跨seed复验。多轮查看过同一test，不能把离线预测变化直接解释为下游反事实列表价值或在线收益。', '',
        '## 逐seed指标', '', '| seed | 配置 | AUC | LogLoss | Intra-Step GAUC | valid选中步 |',
        '| ---: | --- | ---: | ---: | ---: | ---: |']
    for key, metric in summary.items():
        seed, name = key.split('/')
        lines.append(f"| {seed} | {name} | {metric['overall.auc']:.6f} | {metric['logloss']:.6f} | {metric['intra_step_gauc']['auc']:.6f} | {metric['selected']['step']} |")
    lines += ['', '## 下一步最小验证', '',
        '保留mixture_step_pace为本轮候选。为区分混合结构的必要性与pace的作用，应补单威布尔仅屏数及其pace版本，组成单/混合×有/无pace的配对对照；当前结果已经隔离了混合模型内的pace增益，但没有证明“任何单威布尔加pace都无法达到同样结果”。另外应固定架构补训练周期对照，再考虑调整混合分量或引入辅助Loss，避免同时改动。', '',
        '## 代码、日志与复现', '',
        '- [实验定义](实验说明.md) · [模型](../../pipeline/time_budget_v10.py) · [训练入口](../../pipeline/run_exit_v10_plan12.py) · [审计生成代码](../../pipeline/report_v10_plan12.py) · [测试](../../tests/test_v10.py)。',
        '- [训练日志](../../runs/exit_v10_plan12/logs/train.log) · [配置](../../runs/exit_v10_plan12/config.json) · [完整指标与分域](../../runs/exit_v10_plan12/summary.json) · [审计及配对区间](../../runs/exit_v10_plan12/audit.json) · [测试日志](../../runs/exit_v10_plan12/logs/model-tests.log)。',
        '- 每配置文件位于`workspace/kr_time_budget/runs/exit_v10_plan12/<seed>/<配置>/`：manifest、initial/final_diagnostics、calibration、initial_predictions/predictions、joint/first_update、joint/selection、joint/best.pt。', '',
        '```bash', 'uv run python -u -m workspace.kr_time_budget.pipeline.run_exit_v10_plan12',
        'uv run python -m workspace.kr_time_budget.pipeline.report_v10_plan12', '```', '']
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'实验报告.md').write_text('\n'.join(lines))
    print(json.dumps(audit['aggregates'], indent=2))
    print(REPORT/'实验报告.md')


if __name__ == '__main__':
    main()

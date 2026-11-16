"""Audit finished v2 folds and write a research handoff without retraining."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from workspace.kr_time_budget.pipeline.run_exit_v2 import scores


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    root=Path('workspace/kr_time_budget/runs/exit_v2')
    data=Path('workspace/kr_time_budget/data/v2')
    raw=pl.read_parquet('workspace/kr_time_budget/data/v1/requests.parquet',columns=['request_id','fold','y_continue','user_id','session_id','elapsed_end_s_DIAGNOSTIC'])
    summary={};fold_audit=[]
    for fold in range(5):
        metadata=json.loads((data/f'fold_{fold}/manifest.json').read_text())
        sets={k:set(v['users']) for k,v in metadata['splits'].items()}
        assert not sets['train']&sets['valid'] and not sets['train']&sets['test'] and not sets['test']&sets['valid']
        manifests=[json.loads((root/f'fold_{fold}'/m/'manifest.json').read_text()) for m in ['content','budget','fusion','ours']]
        assert len({m['initial_sha256'] for m in manifests})==1
        assert len({m['train_order_sha256'] for m in manifests})==1
        assert all(m['no_item_id_schema'] for m in manifests)
        switch=json.loads((root/f'fold_{fold}/ours/stage_switch.json').read_text())
        assert switch['gamma_max_change_after_content_training']==0
        assert switch['budget_hash_before_stage2']==switch['budget_hash_after_stage2']
        schema=data/f'fold_{fold}/schema'
        for split in ['train','valid','test']:
            deltas=pl.scan_parquet(data/f'fold_{fold}'/f'{split}.parquet').select(pl.col('delta_minutes').unique()).collect()['delta_minutes'].to_list()
            assert deltas==[1.],deltas
        fold_audit.append({'fold':fold,'metadata':metadata,'initial_sha256':manifests[0]['initial_sha256'],'same_initialization_all_four':True,'same_train_order_all_four':True,'train_order_sha256':manifests[0]['train_order_sha256'],'stage_switch':switch,'schema_files_sha256':{str(p.relative_to(schema)):sha(p) for p in sorted(schema.rglob('*')) if p.is_file()},'dataset_files_sha256':{split:sha(data/f'fold_{fold}'/f'{split}.parquet') for split in ['train','valid','test']}})
    for mode in ['content','budget','fusion','ours']:
        paths=[root/f'fold_{fold}'/mode/'predictions.parquet' for fold in range(5)]
        p=pl.read_parquet(paths)
        assert p.height==raw.height and p['request_id'].n_unique()==raw.height
        matched=p.join(raw,on='request_id',suffix='_source',validate='1:1')
        for key in ['fold','y_continue','user_id','session_id']:
            assert (matched[key]==matched[key+'_source']).all()
        expected_elapsed=(matched['elapsed_end_s_DIAGNOSTIC']/60).cast(pl.Float32)
        assert np.array_equal(matched['elapsed_minutes'].to_numpy(),expected_elapsed.to_numpy())
        m=scores(p);m['completed_folds']=5;m['each_request_tested_once']=True
        folds=[json.loads(path.with_name('metrics.json').read_text()) for path in paths]
        assert all(x['full_train'] and x['full_test'] for x in folds)
        m['fold_metrics']=folds
        if mode in ['ours','budget']:
            assert all(x['survival_diagnostics']['log_s_increase_when_clock_nondecreasing']==0 for x in folds)
        if mode=='ours':
            assert all(x['product_max_abs_error']==0 for x in folds)
            m['tower_diagnostics']=p.select(pl.col('gamma').mean().alias('gamma_mean'),pl.col('content_stay').mean().alias('stay_mean'),(pl.col('content_stay')>.99).mean().alias('stay_over_99_fraction'),(pl.col('gamma')-pl.col('prediction')).mean().alias('mean_gamma_minus_product')).to_dicts()[0]
        summary[mode]=m
    # Paired user AUCs from pooled OOF, bootstrapped by user (not by request).
    def per_user(mode):
        p=pl.read_parquet([root/f'fold_{f}'/mode/'predictions.parquet' for f in range(5)],columns=['user_id','y_continue','prediction'])
        p=p.with_columns(pl.col('prediction').rank('average').over('user_id').alias('rank'))
        return p.group_by('user_id').agg(pl.len().alias('n'),pl.col('y_continue').sum().alias('pos'),(pl.col('rank')*pl.col('y_continue')).sum().alias('rank_sum')).with_columns(((pl.col('rank_sum')-pl.col('pos')*(pl.col('pos')+1)/2)/(pl.col('pos')*(pl.col('n')-pl.col('pos')))).alias('auc')).select('user_id','n','auc')
    paired=per_user('ours').join(per_user('fusion'),on='user_id',suffix='_fusion').sort('user_id')
    delta=(paired['auc']-paired['auc_fusion']).to_numpy();weight=paired['n'].to_numpy()
    rng=np.random.default_rng(2026);samples=[]
    for _ in range(2000):
        idx=rng.integers(0,len(delta),len(delta));samples.append(float(np.sum(delta[idx]*weight[idx])/weight[idx].sum()))
    paired_result={'comparison':'ours minus fusion User-GAUC','delta':float(np.sum(delta*weight)/weight.sum()),'bootstrap_unit':'user','replicates':2000,'seed':2026,'ci95_percentile':np.quantile(samples,[.025,.975]).tolist(),'interpretation':'single-seed paired predictive difference, not a mechanism attribution'}
    v1=json.loads(Path('workspace/kr_time_budget/runs/exit_v1/summary.json').read_text())
    code=[Path('workspace/kr_time_budget/pipeline/time_budget_v2.py')]+list(Path('workspace/kr_time_budget/pipeline').glob('*v2.py'))
    constants=[]
    for fold in range(5):
        n=pl.scan_parquet(data/f'fold_{fold}/train.parquet').select(pl.col('y_continue').mean()).collect().item()
        part=pl.read_parquet(root/f'fold_{fold}/ours/predictions.parquet').with_columns(pl.lit(n).alias('prediction'))
        constants.append(part)
    constant=scores(pl.concat(constants))
    audit={'summary':summary,'folds':fold_audit,'paired_user_comparison':paired_result,'train_prevalence_constant':constant,'end_clock_verified_all_oof':True,'delta_one_verified_all_splits':True,'source_hashes':{str(p):sha(p) for p in code},'config':json.loads((root/'config.json').read_text()),'tests_log':'workspace/kr_time_budget/data/v2/logs/tests.log'}
    (root/'audit.json').write_text(json.dumps(audit,indent=2))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    names={'content':'Content / tag-only','budget':'Budget / fixed context','fusion':'Fusion / tag-only','ours':'Survival × Content'}
    table=['| 模型 | User-GAUC ↑ | Pooled-AUC ↑ | LogLoss ↓ | Intra-Step GAUC ↑ |','|---|---:|---:|---:|---:|']
    for mode,m in summary.items():
        table.append(f"| {names[mode]} | {m['user_gauc']['auc']:.6f} | {m['overall.auc']:.6f} | {m['logloss']:.6f} | {m['intra_step_gauc']['auc']:.6f} |")
    old=['| 模型 | v1 User-GAUC | v2 User-GAUC | v1 LogLoss | v2 LogLoss |','|---|---:|---:|---:|---:|']
    for mode,m in summary.items():
        old.append(f"| {names[mode]} | {v1[mode]['user_gauc']['auc']:.6f} | {m['user_gauc']['auc']:.6f} | {v1[mode]['logloss']:.6f} | {m['logloss']:.6f} |")
    per_fold=['| 折 | 模型 | 选中epoch /训练步 | valid LogLoss | test User-GAUC | test LogLoss |','|---|---|---:|---:|---:|---:|']
    for fold in range(5):
        for mode,m in summary.items():
            x=m['fold_metrics'][fold]
            per_fold.append(f"| {fold} | {mode} | {x['selected_epoch']} / {x['selected_step']} | {x['selected_valid']['logloss']:.6f} | {x['user_gauc']['auc']:.6f} | {x['logloss']:.6f} |")
    diag=summary['ours']['tower_diagnostics']
    switches=[x['stage_switch'] for x in fold_audit]
    ratios=[s['product_to_gamma_mean_ratio_first_valid_batch'] for s in switches]
    m=summary['ours']
    verdict='相乘模型的User-GAUC高于Fusion。' if m['user_gauc']['auc']>summary['fusion']['user_gauc']['auc'] else '相乘模型的User-GAUC仍低于Fusion，本轮没有证实冻结预算×内容残差的优势。'
    note=f'''# KR 时间预算 v2：Argon 反馈执行结果

依据：[3反馈.md](../../../Argon-plans/3反馈.md)。本轮按用户/Leader口径执行屏末挂钟与固定1分钟步长，纠正v1的测试用户先验失配、多epoch高基数ID过拟合风险、两阶段冲击和预算编码器漂移。**结论：{verdict}**

## 五折全量结果

1000用户，118,041场，1,856,774条request；外层五折用户分组，每条request仅在其测试折预测一次。每折从外层训练用户中留出10%作为valid，schema和全局/物品先验只用inner train。三个完整epoch均执行，按valid LogLoss选择checkpoint，仅测试选中模型。没有用test调参。

{chr(10).join(table)}

仅使用各折inner train继续比例的常数预测，合并OOF LogLoss={constant['logloss']:.6f}。

Intra-Step GAUC按精确req_idx分组、样本数加权，排除单类组。Ours覆盖 {m['intra_step_gauc']['eligible_rows']:,}/{m['rows']:,} request、{m['intra_step_gauc']['eligible_groups']}/{m['intra_step_gauc']['total_groups']} 组。额外的(user_active_degree,req_idx_bin) GAUC={m['active_step_gauc']['auc']:.6f}，详见JSON。它们是固定进度的预测诊断，不能单靠分桶宣称机制识别。

Ours−Fusion User-GAUC差值 {paired_result['delta']:+.6f}；以用户为单位、2000次配对bootstrap的95%区间 [{paired_result['ci95_percentile'][0]:+.6f}, {paired_result['ci95_percentile'][1]:+.6f}]。这是单seed预测差异，不是机制因果归因。

## 相较v1

{chr(10).join(old)}

v2同时修改多项已指定协议，并新增inner valid；不能把总体变化全部归因于去除item-ID，或把v1失败解释为时间预算思路失败。v1仅用训练用户历史兜底测试用户，产生了不必要的先验失配；本轮所有用户都使用自身严格过去的已完成会话，再以训练总体先验m=5平滑。测试用户的行为历史是当前时刻已知信息，与将未来监督目标放入优化器不同。

## 六条修改的落实

1. 不建立slate_items/history_items的Embedding，仅tag、片长、过去日物品质量先验；本轮选择反馈允许的默认去ID方案，没有运行可选保留ID/频次50/dropout分支。
2. 自身会话历史对train/valid/test一致，查询时刻固定到当前场开始日期。历史会话结束日期严格早于当前场开始日期；不再额外滞后一天。train总体过去日先验以5场伪计数平滑。
3. 预算参数只看场开始固定背景。B2时钟/序号、动态B3、B4及序列均掩码；保留开场hour/weekend/tab/前序当日消费。仅掩码B2/B4而允许B3动态变化仍不能保证固定分布，本轮选择严格静态版本。
4. elapsed使用本屏末条timestamp减场首timestamp，加本屏末条clipped播放量；delta=1分钟。片长留在内容/Fusion输入。当前实验为屏末观察协议，不再套用v1的屏前时钟解释。
5. 内容头bias=+2.6。Stage1只训练gamma和独立预算编码器，固定一个完整epoch；从该epoch的valid检查点选warmup，然后Stage2冻结预算编码器和budget_net，并gamma.detach，训练两完整epoch内容残差。
6. 每epoch的25%/50%/100%做full valid。核心指标User-GAUC/Pooled-AUC/LogLoss，增加精确进度Intra-Step和活跃度×进度分桶GAUC，不再把退化的Session-GAUC当主要证据。

## 实现验证与机制诊断

- 每折四模型初始全state hash完全一致，同一折train物理顺序、batch4096、最大3epoch一致。AdamW lr=1e-3，矩阵decay=1e-6，Embedding/bias/一维参数零衰减；本轮无item-ID，不适用保留ID分支的Embedding衰减例外。
- 两塔字段接口一致，预算参数路径按反馈约束显式路由，激活字段不同。预算与内容编码器初始tensor一致、可训练参数独立。Stage2后预算hash和probe gamma逐元素精确不变；所有五折通过。
- 切换时乘积/gamma均值比范围 {min(ratios):.6f}–{max(ratios):.6f}，不再接近0.5；预算输出无训练漂移；概率乘积误差为0。
- OOF gamma均值 {diag['gamma_mean']:.6f}，content_stay均值 {diag['stay_mean']:.6f}，stay>0.99比例 {diag['stay_over_99_fraction']:.2%}，gamma−乘积均值 {diag['mean_gamma_minus_product']:.6f}。这显示内容乘子趋近1，冻结后对预算预测的修正较小。预热gamma已经直接拟合完整继续标签，而乘子只能下调gamma；这是需要下一步检查的约束，不预先断定因果。
- 固定参数时，人工递增e的S单调测试通过；全量OOF中“原时钟未回退但S上升”的次数为0。屏末原公式在少量相邻chunk上会回退；按要求原样保留并单独统计，不用cummax悄悄改口径。
- 原始KR约束和v1观测保留为背景，本轮不以未经对齐的时间戳推断拒绝Leader的实验定义。固定1分钟生存比作为继续预测的归纳偏置，不将下一屏标签定义等同于已校准的1分钟生存事件。

## Checkpoint与更新谱系

{chr(10).join(per_fold)}

选中epoch内的检查点可能为0.25/0.5epoch，不能一概称训练满3epoch的模型。Ours Stage1虽执行完整1epoch，但回选最佳warmup检查点再开始Stage2；日志的step为实际执行步编号，实际保留谱系为warmup选中步数+已选Stage2更新步数。Stage2起点也参与候选，避免将内容未适应的初始乘积强制作为改进。所有最终测试都加载本run的best.pt，checkpoint含stage和valid指标。

## 下一步建议

先与用户讨论两个有区分力的对照：预算预热是否需要给内容留出可学习空间；预算参数只看场初背景与允许历史状态更新各回答什么问题。后一方案不应继续声称跨步S属于固定分布。若要确认ID过拟合的归因，再单独做同协议tag-only与保留频次≥50 ID+指定正则的配对消融；不从本轮多修复结果直接下因果结论。

## 产物

- 方案：`workspace/kr_time_budget/history/20261116_kr-time-budget-v2.md`
- 说明：`workspace/kr_time_budget/history/README-v2.md`
- 数据/折内schema：`workspace/kr_time_budget/data/v2/fold_*/`
- 模型：`workspace/kr_time_budget/pipeline/time_budget_v2.py`
- 训练入口：`workspace/kr_time_budget/pipeline/run_exit_v2.py`
- 全日志：`workspace/kr_time_budget/runs/exit_v2/run.log`
- 指标/审计：`workspace/kr_time_budget/runs/exit_v2/summary.json`、`audit.json`
- OOF/checkpoint/valid记录：`workspace/kr_time_budget/runs/exit_v2/fold_*/{{content,budget,fusion,ours}}/`
- 测试日志：`workspace/kr_time_budget/data/v2/logs/tests.log`

本实验后续代码、报告与产物统一放workspace/kr_time_budget。
'''
    report=Path('workspace/kr_time_budget/history/20261116_kr-time-budget-v2-report.md')
    report.write_text(note)
    print(report)
    print(json.dumps({mode:{k:m[k] for k in ['user_gauc','overall.auc','logloss','intra_step_gauc']} for mode,m in summary.items()},indent=2))

if __name__=='__main__':
    main()

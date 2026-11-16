"""弱辅助复核和报告：复用同一指标/曲线审计，另逐步检查10%上限与梯度。"""
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

from workspace.kr_time_budget.pipeline.exit_training import WORKSPACE, EXPERIMENTS, load_batches, state_hash
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_uid_zero import zero_uid_inputs
from workspace.kr_time_budget.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel, cap_auxiliary
from workspace.kr_time_budget.pipeline.report_auxiliary_exit import analyze, digest, VARIANTS, COLORS, LABELS


def check_caps(run):
    config=json.loads((run/'config.json').read_text())
    strong=Path(config['previous_strong_run'])
    assert digest(strong/'config.json')==config['previous_strong_config_sha256']
    assert config['max_auxiliary_main_ratio']==.1
    old_audit=json.loads((strong/'analysis_audit.json').read_text())
    assert digest(Path(__file__).with_name('report_auxiliary_exit.py'))==old_audit['analysis_source_sha256']
    for p,h in old_audit['source_hashes'].items():
        assert digest(Path(p))==h,p
    old_metrics_hash=digest(strong/'analysis_metrics.parquet')
    all_steps=[];summaries=[];hashes={}
    for seed in config['seeds']:
        for model in config['experiments']:
            for variant,weights in config['variants'].items():
                key=f'{seed}/{model}/{variant}'
                for epoch in range(1,6):
                    path=run/key/f'loss_steps_epoch_{epoch:02d}.parquet'
                    frame=pl.read_parquet(path)
                    assert frame.height==421
                    assert frame['rows'].sum()==1721745
                    assert frame['step'].to_list()==list(range((epoch-1)*421+1,epoch*421+1))
                    a,b=weights['left_weight'],weights['right_weight']
                    assert 0<=a<=.1 and 0<=b<=.1
                    j=frame['joint_bce'].to_numpy()
                    n,c=frame['left_session_nll'].to_numpy(),frame['right_bce'].to_numpy()
                    before=a*n+b*c
                    scale=np.minimum(1.,.1*j/np.maximum(before,np.finfo(float).tiny))
                    weighted=before*scale
                    for actual,expected in [('auxiliary_before_cap',before),('cap_scale',scale),
                        ('auxiliary_weighted',weighted),('effective_left_weight',a*scale),
                        ('effective_right_weight',b*scale),('total',j+weighted),('auxiliary_main_ratio',weighted/j)]:
                        np.testing.assert_allclose(frame[actual].to_numpy(),expected,rtol=1e-6,atol=1e-10)
                    assert frame['auxiliary_main_ratio'].max()<=.1+1e-12
                    assert frame['effective_left_weight'].max()<=.1
                    assert frame['effective_right_weight'].max()<=.1
                    row=dict(seed=seed,model=model,variant=variant,epoch=epoch,
                        joint_mean=float(np.average(j,weights=frame['rows'])),
                        weighted_aux_mean=float(np.average(weighted,weights=frame['rows'])),
                        maximum_ratio=float(frame['auxiliary_main_ratio'].max()),
                        mean_effective_left=float(np.average(a*scale,weights=frame['rows'])),
                        mean_effective_right=float(np.average(b*scale,weights=frame['rows'])),
                        cap_active_fraction=float(np.mean(scale<1)))
                    summaries.append(row)
                    all_steps.append(frame.with_columns(pl.lit(seed).alias('seed'),pl.lit(model).alias('model'),
                        pl.lit(variant).alias('variant'),pl.lit(epoch).alias('epoch')))
                    hashes[str(path)]=digest(path)
    pl.concat(all_steps).write_parquet(run/'loss_steps_all.parquet')
    pl.DataFrame(summaries).write_parquet(run/'loss_cap_summary.parquet')
    steps=sum(f.height for f in all_steps)
    result=dict(training_steps_checked=steps,all_steps_within_main_loss_10_percent=True,
        maximum_ratio=max(r['maximum_ratio'] for r in summaries),coefficient_maximum=.1,
        independent_numpy_cap_reconstruction=True,strong_run_source_artifacts_unchanged=True,
        loss_step_sha256=hashes,strong_metrics_sha256=old_metrics_hash,source_sha256=digest(Path(__file__)))
    (run/'loss_cap_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


def gradient_diagnostic(run):
    """固定第5轮checkpoint+首个训练batch，无step；区分数值上限与梯度比例。"""
    config=json.loads((run/'config.json').read_text())
    manager=request_manager(WORKSPACE/'data/v3/schema',load=True)
    batch=zero_uid_inputs(load_batches(WORKSPACE/'data/v3',manager,
        {'train':4096,'valid':1,'test':1}))['train'][0]
    rows=[]
    for seed in config['seeds']:
        for name in config['experiments']:
            for variant,weights in config['variants'].items():
                checkpoint=run/f'{seed}/{name}/{variant}/epoch_05/checkpoint.pt'
                model=LimitedAuxiliaryExitModel(manager,**EXPERIMENTS[name],**weights,
                    max_aux_ratio=.1,requests_per_session=config['left_normalization']).eval()
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model.load_state_dict(payload['model_state_dict'],strict=True)
                before=state_hash(model)
                rng=torch.get_rng_state().clone()
                terms=model.loss_terms(batch)
                aux=weights['left_weight']*terms['left_session_nll']+weights['right_weight']*terms['right_bce']
                weighted,scale=cap_auxiliary(terms['joint_bce'],aux)
                parameters=list(model.parameters())
                main_grad=torch.autograd.grad(terms['joint_bce'],parameters,retain_graph=True,allow_unused=True)
                aux_grad=torch.autograd.grad(weighted,parameters,allow_unused=True)
                budget_ids={id(p) for m in model.budget_modules for p in m.parameters()}
                for scope in ['all','budget','content']:
                    indexes=[i for i,p in enumerate(parameters) if scope=='all' or
                        ((id(p) in budget_ids)==(scope=='budget'))]
                    main_sq=aux_sq=dot=0.
                    for i in indexes:
                        g,h=main_grad[i],aux_grad[i]
                        if g is not None:main_sq+=float(g.double().square().sum())
                        if h is not None:aux_sq+=float(h.double().square().sum())
                        if g is not None and h is not None:dot+=float((g.double()*h.double()).sum())
                    rows.append(dict(seed=seed,model=name,variant=variant,scope=scope,
                        epoch=5,training_batch=1,rows=4096,main_gradient_norm=main_sq**.5,
                        auxiliary_gradient_norm=aux_sq**.5,
                        auxiliary_main_gradient_ratio=(aux_sq/main_sq)**.5 if main_sq else None,
                        gradient_cosine=dot/(main_sq*aux_sq)**.5 if main_sq*aux_sq else None,
                        loss_value_ratio=float(weighted.detach()/terms['joint_bce'].detach()),
                        cap_scale=float(scale),optimizer_step_performed=False))
                assert state_hash(model)==before and torch.equal(torch.get_rng_state(),rng)
    pl.DataFrame(rows).write_parquet(run/'gradient_diagnostic.parquet')


def generate(run,report):
    report.mkdir(parents=True,exist_ok=True)
    config=json.loads((run/'config.json').read_text())
    strong=Path(config['previous_strong_run'])
    frame=pl.read_parquet(run/'analysis_metrics.parquet')
    aggregates=pl.read_parquet(run/'analysis_aggregates.parquet')
    old=pl.read_parquet(strong/'analysis_aggregates.parquet')
    curves=pl.read_parquet(run/'analysis_curves.parquet')
    caps=pl.read_parquet(run/'loss_cap_summary.parquet')
    gradients=pl.read_parquet(run/'gradient_diagnostic.parquet')
    models=sorted(frame['model'].unique().to_list(),reverse=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,m,title in zip(axes.flat,['auc','user_auc','budget_rmse50','product_rmse50'],
        ['Test Pooled AUC','Test User-GAUC','Budget S RMSE (r=0..50)','Final risk-product S RMSE (r=0..50)']):
        for model in models:
            for v in VARIANTS:
                selected=aggregates.filter((pl.col('model')==model)&(pl.col('variant')==v)).sort('epoch')
                ax.plot(selected['epoch'],selected[m+'_mean'],color=COLORS[v],linestyle='--' if model.startswith('single') else '-',
                    label=f'{model.split("_")[0]} {LABELS[v]}')
        ax.set(title=title,xlabel='Epoch',xticks=[1,2,3,4,5]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.savefig(report/'弱辅助逐epoch.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,model in zip(axes,models):
        selected=curves.filter((pl.col('model')==model)&(pl.col('epoch')==5)&(pl.col('r')<=50))
        truth=selected.filter(pl.col('variant')=='baseline').group_by('r').agg(pl.col('survival').mean()).sort('r')
        ax.plot(truth['r'],truth['survival'],color='black',linewidth=2,label='Observed S')
        for variant in VARIANTS:
            c=selected.filter(pl.col('variant')==variant).group_by('r').agg(pl.col('budget').mean(),pl.col('product').mean()).sort('r')
            ax.plot(c['r'],c['budget'],color=COLORS[variant],linestyle='--',label=LABELS[variant]+' budget')
            ax.plot(c['r'],c['product'],color=COLORS[variant],label=LABELS[variant]+' final')
        ax.set(title=model,xlabel='Completed screens r',ylabel='Survival',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(report/'弱辅助生存曲线.png',dpi=170);plt.close(fig)
    lines=['# 第21份补充：弱辅助Loss复验','',
        '**辅助合计每步最多为主Loss的10%，每项有效系数也≤0.1。** 主Loss保持原乘积BCE，不改变forward。旧权重1实验保留；本报告是按用户要求降低强度后的新实验。','',
        '完整定义见[弱辅助实验说明](弱辅助实验说明.md)。沿用KR v3、编码UID置0、单/混合+pace、3seed，每轮末测完整valid/test，连续5轮，每条2105次更新。新增18条训练轨迹、90次完整test；旧无辅助和旧强辅助直接复用。','',
        '## 核心结果','']
    for model in models:
        selection=(pl.col('model')==model)&(pl.col('epoch')==5)
        baseline=aggregates.filter(selection&(pl.col('variant')=='baseline')).row(0,named=True)
        both=aggregates.filter(selection&(pl.col('variant')=='both')).row(0,named=True)
        previous=old.filter(selection&(pl.col('variant')=='both')).row(0,named=True)
        lines += [f'{model}双辅助：AUC={both["auc_mean"]:.6f}（无辅助{baseline["auc_mean"]:.6f}，旧强辅助{previous["auc_mean"]:.6f}）；'
            f'User-AUC={both["user_auc_mean"]:.6f}（无辅助{baseline["user_auc_mean"]:.6f}，旧强辅助{previous["user_auc_mean"]:.6f}）。'
            f'左塔S RMSE由无辅助{baseline["budget_rmse50_mean"]:.4f}变为{both["budget_rmse50_mean"]:.4f}，'
            f'最终曲线RMSE由{baseline["product_rmse50_mean"]:.4f}变为{both["product_rmse50_mean"]:.4f}。','']
    lines += [
        '## 固定第5轮完整test结果','',
        '| 结构 | 变体 | Pooled AUC | LogLoss | User-AUC | Intra-Step GAUC | 左塔会话NLL | 左塔S RMSE | 最终曲线RMSE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    metric_names=['auc','logloss','user_auc','intra_step_gauc','session_nll','budget_rmse50','product_rmse50']
    for model in models:
        for v in VARIANTS:
            row=aggregates.filter((pl.col('model')==model)&(pl.col('variant')==v)&(pl.col('epoch')==5)).row(0,named=True)
            lines.append('| '+' | '.join([model,LABELS[v],*[f'{row[m+"_mean"]:.6f} ± {row[m+"_std"]:.6f}' for m in metric_names]])+' |')
    lines+=['','均值±seed标准差，固定第5轮末权重；未以test选点。User-AUC是请求数加权的用户内AUC，排除单类别用户后为823用户、48,855请求。Pooled AUC使用全部48,898请求。','',
        '## 相同seed下的变化','',
        '| 结构 | 变体 | ΔAUC vs无辅助 | ΔUser-AUC vs无辅助 | ΔLogLoss vs无辅助 | ΔAUC vs旧强辅助 | ΔUser-AUC vs旧强辅助 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for model in models:
        reference=aggregates.filter((pl.col('model')==model)&(pl.col('variant')=='baseline')&(pl.col('epoch')==5)).row(0,named=True)
        for v in VARIANTS[1:]:
            row=aggregates.filter((pl.col('model')==model)&(pl.col('variant')==v)&(pl.col('epoch')==5)).row(0,named=True)
            previous=old.filter((pl.col('model')==model)&(pl.col('variant')==v)&(pl.col('epoch')==5)).row(0,named=True)
            differences=[row[m+'_mean']-reference[m+'_mean'] for m in ['auc','user_auc','logloss']]
            differences += [row[m+'_mean']-previous[m+'_mean'] for m in ['auc','user_auc']]
            lines.append('| '+' | '.join([model,LABELS[v],*[f'{x:+.6f}' for x in differences]])+' |')
    lines+=['','## 每步Loss上限核验','',
        '原始辅助系数a,b从1降到0.1；再令`c=min(1,0.1*J.detach()/A.detach())`，`A=a*N+b*C`。最终优化`J+c*A`。c停止梯度，主任务梯度系数仍为1；两个辅助项共用合计上限。','',
        '| 结构 | 变体 | epoch5主Loss均值 | epoch5加权辅助均值 | 左项有效系数均值 | 右项有效系数均值 | 全5轮最大辅助/主比 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for model in models:
        for v in VARIANTS[1:]:
            group=caps.filter((pl.col('model')==model)&(pl.col('variant')==v))
            last=group.filter(pl.col('epoch')==5)
            lines.append('| '+' | '.join([model,LABELS[v],
                *[f'{last[m].mean():.6f}' for m in ['joint_mean','weighted_aux_mean','mean_effective_left','mean_effective_right']],
                f'{group["maximum_ratio"].max():.12f}'])+' |')
    lines+=['','每场NLL与每请求BCE的数值尺度不同，左项实际有效系数通常远小于0.1，这是执行“辅助合计最多主Loss10%”的结果。合计上限相同时，左/右/双辅助分配给两个分支的比例仍不同。每步占比、系数、主Loss与原始辅助值全部留档，并用独立NumPy公式重算。','',
        '![弱辅助逐epoch](弱辅助逐epoch.png)','','![弱辅助生存曲线](弱辅助生存曲线.png)','','## 固定checkpoint梯度诊断','',
        '固定第5轮模型，在第一个4096行训练batch上只计算梯度，不执行optimizer.step。下表为三seed均值；比例是裁剪前的辅助梯度范数/主任务梯度范数，不能与Loss数值占比混淆。','',
        '| 结构 | 变体 | 全模型梯度比 | 预算分支梯度比 | 内容分支梯度比 | 全模型梯度夹角余弦 |',
        '| --- | --- | --- | --- | --- | --- |']
    for model in models:
        for v in VARIANTS[1:]:
            group=gradients.filter((pl.col('model')==model)&(pl.col('variant')==v))
            ratios=[group.filter(pl.col('scope')==scope)['auxiliary_main_gradient_ratio'].mean() for scope in ['all','budget','content']]
            cosine=group.filter(pl.col('scope')=='all')['gradient_cosine'].mean()
            lines.append('| '+' | '.join([model,LABELS[v],*[f'{x:.6f}' for x in ratios],f'{cosine:.6f}'])+' |')
    lines+=['','该诊断是一个固定训练batch，不代表整轮梯度统计；训练仍使用原clip=10。10%的数值上限本身不能承诺梯度范数≤10%。','',
        '## 每轮完整test均值','',
        '| 结构 | 变体 | epoch | AUC | LogLoss | User-AUC | 左塔NLL | 左塔S RMSE | 最终曲线RMSE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for model in models:
        for v in VARIANTS:
            for row in aggregates.filter((pl.col('model')==model)&(pl.col('variant')==v)).sort('epoch').iter_rows(named=True):
                lines.append('| '+' | '.join([model,LABELS[v],str(row['epoch']),
                    *[f'{row[m+"_mean"]:.6f}' for m in ['auc','logloss','user_auc','session_nll','budget_rmse50','product_rmse50']]])+' |')
    lines+=['','## 结论边界和证据','',
        '这次只降低辅助强度，未搜索多个上限；不以test结果进一步调节。三seed均值和波动不替代显著性检验。左塔仍拟合观测总停止长度，右塔仍使用总继续标签；强度变化不构成内容满意度监督或潜在预算识别。','',
        '左塔S曲线为所有首屏预算分布的平均；最终曲线为已观察风险集平均继续概率的连乘。RMSE都在r=0..50衡量，二者预测对象不同。KR v3沿用完整会话依赖的分屏和屏末pace，不声称屏前在线可用。','',
        '数据：train 2022-04-08至05-06的1,721,745请求/109,368场，valid 05-07的57,872请求/3,822场，test 05-08的48,898请求/3,281场。初始化hash、optimizer分组、物理请求顺序与已有对照一致，编码UID置0、其他输入保持原样。AdamW预算lr=.005、其他lr=.001、普通权重decay1e-6、embedding/bias/1D为0，batch4096、clip10，无冻结、预热、scheduler或EMA。','',
        '- [训练日志](../../runs/plan21_auxiliary_cap10/logs/train.log) · [配置](../../runs/plan21_auxiliary_cap10/config.json) · [测试日志](../../runs/plan21_auxiliary_cap10/logs/tests.log)',
        '- [指标与checkpoint审计](../../runs/plan21_auxiliary_cap10/analysis_audit.json) · [Loss上限审计](../../runs/plan21_auxiliary_cap10/loss_cap_audit.json)',
        '- [全部指标](../../runs/plan21_auxiliary_cap10/analysis_metrics.parquet) · [每步Loss](../../runs/plan21_auxiliary_cap10/loss_steps_all.parquet) · [梯度诊断](../../runs/plan21_auxiliary_cap10/gradient_diagnostic.parquet)',
        '- [弱辅助模型](../../pipeline/limited_auxiliary_exit.py) · [训练入口](../../pipeline/run_limited_auxiliary.py) · [报告入口](../../pipeline/report_limited_auxiliary.py)',
        '- [旧权重1报告](实验报告.md) · [弱辅助实验说明](弱辅助实验说明.md)','']
    (report/'弱辅助实验报告.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=WORKSPACE/'runs/plan21_auxiliary_cap10')
    parser.add_argument('--report',type=Path,default=WORKSPACE/'回复/21新 Loss')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if not args.report_only:
        analyze(args.run)
        check_caps(args.run)
        gradient_diagnostic(args.run)
    generate(args.run,args.report)
    print('REPORT',args.report/'弱辅助实验报告.md',flush=True)


if __name__=='__main__':
    main()

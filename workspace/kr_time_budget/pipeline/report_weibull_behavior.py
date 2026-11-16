"""Plan19只读汇报：经验离散Weibull拟合、预算分支与总继续概率的差别。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from workspace.kr_time_budget.pipeline.check_weibull_behavior import log_survival, MODELS
from workspace.kr_time_budget.pipeline.exit_training import WORKSPACE


def generate(run, report):
    manifest=json.loads((run/'manifest.json').read_text())
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    assert manifest['models_extracted']==72 and manifest['sessions']==3281
    for run_name in ['epochwise5','epochwise5_uid_zero']:
        config=json.loads((WORKSPACE/'runs'/run_name/'config.json').read_text())
        for filename,digest in config['source_hashes'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    fits=json.loads((run/'empirical_fits.json').read_text())
    empirical=pl.read_parquet(run/'empirical_global.parquet')
    groups=pl.read_parquet(run/'group_fits.parquet')
    curves=pl.read_parquet(run/'curves.parquet')
    risk=pl.read_parquet(run/'risk_curves.parquet')
    params=pl.read_parquet(run/'model_session_parameters.parquet')
    correlations=pl.DataFrame(json.loads((run/'group_correlations.json').read_text()))
    primary=groups.filter((pl.col('epoch')==5)&(pl.col('grouping')=='global'))
    calibration=[]
    for key,group in risk.filter((pl.col('grouping')=='global')&(pl.col('req_idx_raw')<=50)).partition_by(
            'uid','seed','epoch','model',as_dict=True).items():
        weights=group['at_risk'].to_numpy();observed=group['observed_continue'].to_numpy()
        row=dict(zip(['uid','seed','epoch','model'],key))
        for field in ['mean_prediction','mean_gamma']:
            if group[field].null_count()==group.height:continue
            error=group[field].to_numpy()-observed
            row[field+'_weighted_rmse50']=float(np.sqrt(np.average(error**2,weights=weights)))
        calibration.append(row)
    (run/'risk_calibration_summary.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2)+'\n')
    report.mkdir(parents=True,exist_ok=True)
    def select(frame,state,model,epoch=5,grouping='global'):
        return frame.filter((pl.col('uid')==state)&(pl.col('model')==model)&
                            (pl.col('epoch')==epoch)&(pl.col('grouping')==grouping))
    def average(frame,field,key='r'):
        return frame.group_by(key).agg(pl.col(field).mean()).sort(key)
    grid=np.arange(101)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    actual=empirical.filter(pl.col('r')<=100)
    for axis,state in zip(axes,['uid_zero','uid']):
        axis.plot(actual['r'],actual['survival'],color='black',label='Empirical all-cause survival',linewidth=2)
        for fit_name,label in [('discrete_single_mle','Empirical single MLE'),('discrete_mixture_mle','Empirical K2 MLE')]:
            fit=fits[fit_name]
            curve=np.exp(log_survival(grid,np.array(fit['scale'])[None,:],np.array(fit['shape'])[None,:],np.array(fit['prior'])[None,:]))[0]
            axis.plot(grid,curve,linestyle=':',label=label)
        for model in ['single_step_pace','mixture_step_pace']:
            selected=select(curves,state,model).filter(pl.col('r')<=100)
            mean=average(selected,'mean_budget_survival')
            line,=axis.plot(mean['r'],mean['mean_budget_survival'],linestyle='--',label=model+' budget S')
            total=average(selected,'observed_risk_product_survival')
            axis.plot(total['r'],total['observed_risk_product_survival'],color=line.get_color(),label=model+' risk product')
        axis.set(xlabel='r: screens passed',ylabel='Survival',title=state+'; epoch5, 3-seed mean',ylim=(0,1.02))
        axis.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=7)
    fig.savefig(report/'全局生存曲线.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,state in enumerate(['uid_zero','uid']):
        view=empirical.filter((pl.col('r')>=1)&(pl.col('r')<=50))
        axes[0,col].plot(view['r'],view['continuation'],color='black',label='Observed continue')
        single=fits['discrete_single_mle'];lam,z=single['scale'][0],single['shape'][0]
        t=view['r'].to_numpy();gamma=np.exp(((t-1)/lam)**z-(t/lam)**z)
        axes[0,col].plot(t,gamma,linestyle=':',label='Empirical single MLE conditional')
        for model in ['single_step_pace','mixture_step_pace','fusion']:
            selected=select(risk,state,model).filter(pl.col('req_idx_raw')<=50)
            mean=average(selected,'mean_prediction','req_idx_raw')
            line,=axes[0,col].plot(mean['req_idx_raw'],mean['mean_prediction'],label=model+' total')
            if model=='mixture_step_pace':
                b=average(selected,'mean_gamma','req_idx_raw')
                axes[0,col].plot(b['req_idx_raw'],b['mean_gamma'],linestyle='--',color=line.get_color(),label='mixture budget gamma')
        axes[0,col].set(title=state+'; observed risk-set probabilities',xlabel='Request t',ylabel='Continue probability')
        axes[0,col].legend(fontsize=7);axes[0,col].grid(alpha=.2)
        tail=empirical.filter((pl.col('r')>=1)&(pl.col('survival')>0))
        x=np.log(tail['r'].to_numpy());y=np.log(-np.log(tail['survival'].to_numpy()))
        axes[1,col].plot(x,y,color='black',label='Empirical double-log')
        for end in ['10','50','100','680']:
            fit=fits['double_log_ols'][end];use=tail.filter(pl.col('r')<=int(end))
            xx=np.log(use['r'].to_numpy())
            axes[1,col].plot(xx,fit['shape']*(xx-np.log(fit['scale'])),label='OLS through '+end)
        axes[1,col].set(xlabel='log r',ylabel='log(-log S)',title='Same empirical data; fitting range sensitivity')
        axes[1,col].legend(fontsize=7);axes[1,col].grid(alpha=.2)
    fig.savefig(report/'继续率与尾部形态.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,grouping in enumerate(['history_decile','model_scale_decile']):
        selected=select(groups,'uid_zero','mixture_step_pace',grouping=grouping)
        stats=selected.with_columns(pl.col('group').cast(pl.Int32).alias('bucket')).group_by('bucket').agg(
            pl.col('model_effective_scale_mean').mean(),pl.col('empirical_scale').mean(),
            pl.col('empirical_shape').mean(),pl.col('length_mean').mean()).sort('bucket')
        axes[0,col].plot(stats['bucket'],stats['model_effective_scale_mean'],marker='o',label='Mean budget e^-1 crossing')
        axes[0,col].plot(stats['bucket'],stats['empirical_scale'],marker='o',label='Empirical single MLE scale')
        axes[0,col].plot(stats['bucket'],stats['length_mean'],marker='o',label='Observed mean L')
        axes[0,col].set(xlabel='Bucket',ylabel='Screens',title=grouping+'; UID zero, mixture+pace')
        axes[0,col].legend(fontsize=8);axes[0,col].grid(alpha=.2)
        for bucket in ['1','5','10']:
            chosen=select(curves,'uid_zero','mixture_step_pace',grouping=grouping).filter((pl.col('group')==bucket)&(pl.col('r')<=50))
            if chosen.is_empty():continue
            em=average(chosen,'survival');bu=average(chosen,'mean_budget_survival')
            line,=axes[1,col].plot(em['r'],em['survival'],label='Bucket '+bucket+' empirical')
            axes[1,col].plot(bu['r'],bu['mean_budget_survival'],linestyle='--',color=line.get_color(),label='Bucket '+bucket+' budget')
        axes[1,col].set(xlabel='r',ylabel='Survival',ylim=(0,1.02),title='Solid empirical vs dashed budget')
        axes[1,col].legend(fontsize=7);axes[1,col].grid(alpha=.2)
    fig.savefig(report/'分桶核对.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    parameter_summary=[]
    for col,state in enumerate(['uid_zero','uid']):
        for model in MODELS:
            chosen=params.filter((pl.col('uid')==state)&(pl.col('epoch')==5)&(pl.col('model')==model))
            from workspace.kr_time_budget.pipeline.check_weibull_behavior import parameter_arrays,effective_scale
            a,b,p=parameter_arrays(chosen,model.startswith('mixture'))
            scales=effective_scale(a,b,p)
            if model.endswith('pace'):
                axes[0,col].hist(scales,bins=35,density=True,histtype='step',label=model)
                for component in range(b.shape[1]):
                    axes[1,col].hist(b[:,component],bins=35,density=True,histtype='step',
                                     label=model+f' shape{component}')
            for field,array in [('e_minus1_scale',scales),*[(f'scale{k}',a[:,k]) for k in range(a.shape[1])],
                                *[(f'shape{k}',b[:,k]) for k in range(b.shape[1])],
                                *[(f'prior{k}',p[:,k]) for k in range(p.shape[1])]]:
                parameter_summary.append(dict(uid=state,model=model,field=field,
                    mean=float(array.mean()),p10=float(np.quantile(array,.1)),
                    median=float(np.median(array)),p90=float(np.quantile(array,.9))))
        axes[0,col].axvline(fits['discrete_single_mle']['scale'][0],color='black',linestyle=':',label='Empirical global scale')
        axes[1,col].axvline(fits['discrete_single_mle']['shape'][0],color='black',linestyle=':',label='Empirical global shape')
        axes[0,col].set(xlabel='Individual budget e^-1 crossing (screens)',ylabel='Density',title=state+'; epoch5')
        axes[1,col].set(xlabel='Individual component shape',ylabel='Density',title='Parameter summaries, not population fits')
        for axis in axes[:,col]:axis.legend(fontsize=7);axis.grid(alpha=.2)
    fig.savefig(report/'参数分布.png',dpi=180);plt.close(fig)
    (run/'parameter_distribution_summary.json').write_text(json.dumps(parameter_summary,ensure_ascii=False,indent=2)+'\n')
    def fmt(frame,col):
        values=frame[col].to_numpy();return f'{np.mean(values):.6f} ± {np.std(values,ddof=1):.6f}'
    lines=['# 测试集行为威布尔拟合与模型核对', '',
        '**本轮没有重训。** 直接统计KR v3完整test的3,281场、48,898请求；复用既有预测，并对72个已有checkpoint各运行一次首屏预算参数读取。当前UID置0和原UID均覆盖，主看第5轮，第1/3轮作敏感性。', '',
        '结论：真实总停止屏数可被单/混合Weibull近似描述；预算分支尺度与桶级真实长度有正向关系，但预算S本身与总退出曲线存在明显差距。最终继续概率在已观察风险集上的累计曲线接近经验曲线，不能把它升级为外生预算被识别或未观察轨迹已校准的结论。', '',
        '## 1. 口径与要回答的问题', '',
        '真实停止屏数L取每场最大req_idx；每场一个观测，不能对单条请求或单场直接估计“真实lambda和shape”。经验S(r)=P(L>r)，r=0时为1；第t屏之后继续的经验概率是S(t)/S(t−1)。模型用r=t−1输入预算，gamma=S_budget(t)/S_budget(t−1)。', '',
        '必须分开比较：①总退出分布是否近似Weibull；②预算分支的群体曲线是否等于总退出曲线；③最终继续概率在真实风险集上是否贴近观察。②在乘积分解下本就不是与①同一个目标，数值不同不能直接判为实现错误，也不能自动判为识别成功。', '',
        '## 2. 真实停止屏数的经验拟合', '',
        '| 拟合 | scale（屏） | shape | 权重 | 0–50屏生存RMSE | 0–50屏R² |',
        '| --- | --- | --- | --- | ---: | ---: |']
    for key,label in [('discrete_single_mle','离散单Weibull MLE'),('discrete_mixture_mle','离散K=2 MLE')]:
        fit=fits[key]
        lines.append('| '+label+' | '+', '.join(f'{v:.6f}' for v in fit['scale'])+' | '+', '.join(f'{v:.6f}' for v in fit['shape'])+
            ' | '+', '.join(f'{v:.4f}' for v in fit['prior'])+f' | {fit["in_sample_errors50"]["rmse"]:.6f} | {fit["in_sample_errors50"]["r2"]:.6f} |')
    lines += ['', 'MLE使用离散概率P(L=l)=S(l−1)−S(l)，包含L=1，不把整数屏数当连续密度。这是对test本身的描述性拟合，R²是同样本、累计曲线点的拟合指标，不是独立验证，更不是分布检验的通过概率。K=2额外参数更多，拟合误差较小也不能直接当作预测泛化收益。', '',
        '| 双对数OLS拟合范围 | scale | shape | 点数 |', '| --- | ---: | ---: | ---: |']
    for end,fit in fits['double_log_ols'].items():lines.append(f'| 1–{end}屏 | {fit["scale"]:.6f} | {fit["shape"]:.6f} | {fit["points"]} |')
    mean=float(pl.read_parquet(run/'sessions.parquet')['L'].mean())
    lines += ['', f'平均停止屏数为{mean:.6f}，最大681屏。前50屏OLS确实给出约0.72，但随拟合范围变化，shape明显改变；不能把局部斜率与网络参数均值相近写成“全分布已高度吻合”的铁证。', '',
        '## 3. 全局曲线：预算S与总继续概率分开', '', '![全局生存曲线](全局生存曲线.png)', '',
        '| UID | 模型 | 经验MLE scale | 模型预算e^-1交点均值 | 模型第0分量shape均值 | 预算S RMSE₀₋₅₀ | 总概率风险集连乘 RMSE₀₋₅₀ |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for state in ['uid_zero','uid']:
        for model in MODELS:
            group=primary.filter((pl.col('uid')==state)&(pl.col('model')==model))
            lines.append(f'| {state} | {model} | {fits["discrete_single_mle"]["scale"][0]:.6f} | '+
                ' | '.join(fmt(group,field) for field in ['model_effective_scale_mean','model_shape_component0_mean','budget_rmse50','product_rmse50'])+' |')
    lines += ['', '预算曲线按每场首屏的参数计算S_s(r)，对所有3,281场等权平均；不是把平均参数代回Weibull。混合模型用S_s(r)=Σπ_sk exp(−(r/λ_sk)^z_sk)，报告S_s(t)=e^-1的数值交点作尺度摘要；第0分量按每场lambda较小者排序，仅作成分描述，不冒充混合分布shape。', '',
        '“总概率风险集连乘”在每个真实仍在场的屏号上平均保存的prediction，再连乘。已退出场缺少之后的内容历史及列表，因此它是沿既有行为风险集的校准诊断，不是为所有开场用户生成的完整未来轨迹、无偏生存预测或候选策略反事实评估。', '',
        '## 4. 每屏继续率与长尾敏感性', '', '![继续率及尾部](继续率与尾部形态.png)', '',
        '| 已经过r屏 | 经验累计S(r) | 在第r屏的风险集场数 | 第r屏继续率 |',
        '| ---: | ---: | ---: | ---: |']
    for r in [1,2,5,10,20,50,100,200,500,681]:
        row=empirical.filter(pl.col('r')==r).row(0,named=True)
        lines.append(f'| {r} | {row["survival"]:.6f} | {row["at_risk"]} | {row["continuation"]:.6f} |')
    lines += ['', '长尾每点的风险集可能很小。累计点不独立，不能把高R²当作大量独立证据；该test还经过会话首尾剔除和完整6条分屏选择，不是未经选择的真实全部会话总体。', '',
        'UID置0、混合+pace第5轮，部分屏号的三个seed平均概率如下。这里直接比较条件继续率，不将预算gamma冒充最终概率：', '',
        '| 第t屏 | 真实继续率 | 模型gamma均值 | 模型最终prediction均值 |',
        '| ---: | ---: | ---: | ---: |']
    mainrisk=select(risk,'uid_zero','mixture_step_pace')
    for t in [1,2,5,10,20,50]:
        subset=mainrisk.filter(pl.col('req_idx_raw')==t)
        lines.append(f'| {t} | {subset["observed_continue"].mean():.6f} | {subset["mean_gamma"].mean():.6f} | {subset["mean_prediction"].mean():.6f} |')
    lines += ['', '按各屏风险集请求数加权的前50屏继续率RMSE见[条件概率摘要](../../runs/plan19_weibull_check/risk_calibration_summary.json)，包含Fusion对照及1/3/5轮。', '',
        '## 5. 分桶趋势核对', '', '![分桶核对](分桶核对.png)', '',
        '历史场均屏数分位桶是全模型共同的开场先验分组；模型尺度桶按每场预算e^-1交点分位，桶随模型/seed而变。相同取值不强行拆开，实际桶覆盖在原始表中。另保留tab分组。每桶单独估计经验单Weibull，不能把组内异质模型的参数均值当真实参数。', '',
        '| UID | 模型 | 分桶 | 预测尺度均值 vs 经验尺度ρ | 预测尺度均值 vs 实际平均L ρ |',
        '| --- | --- | --- | ---: | ---: |']
    for state in ['uid_zero','uid']:
        for model in MODELS:
            for grouping in ['history_decile','model_scale_decile']:
                rows=select(correlations,state,model,grouping=grouping)
                a=rows.filter(pl.col('target')=='empirical_scale')
                b=rows.filter(pl.col('target')=='length_mean')
                lines.append(f'| {state} | {model} | {grouping} | {fmt(a,"spearman")} | {fmt(b,"spearman")} |')
    lines += ['', 'ρ是每个seed内桶级Spearman，再列三seed均值±标准差。桶数约10且同一用户可跨场出现，未做独立性或显著性检验；不能由正ρ直接推断潜在预算被识别。完整逐桶样本数、参数、拟合误差以及tab在[分桶数据](../../runs/plan19_weibull_check/group_fits.parquet)。', '',
        'UID置0、混合+pace第5轮的桶覆盖与经验拟合质量：', '',
        '| 分组 | 每桶场数范围 | 三seed合计桶拟合数 | R²>0.98数 | 最低R² |',
        '| --- | --- | ---: | ---: | ---: |']
    for grouping in ['history_decile','model_scale_decile','tab_raw']:
        chosen=select(groups,'uid_zero','mixture_step_pace',grouping=grouping)
        lines.append(f'| {grouping} | {chosen["sessions"].min()}–{chosen["sessions"].max()} | {chosen.height} | '+
                     f'{int((chosen["empirical_fit_r2_50"]>.98).sum())} | {chosen["empirical_fit_r2_50"].min():.6f} |')
    lines += ['', '桶内误差使用r=0到min(50,该桶最大L)的已观察进度，不对退出后缺失的risk-set概率外推；不同桶的误差支持范围可能不同。桶内单Weibull的高拟合R²支持描述性近似，但小tab分组未全部达到0.98；重复seed共享同一真实test，合计次数不是独立数据集数。', '',
        '## 6. 参数分布与epoch敏感性', '', '![模型参数分布](参数分布.png)', '',
        '直方图汇集三seed每场参数，仅描述分布，不把三次预测当成三倍独立会话。经验全局拟合参数的竖线与个体分量参数不是同一估计对象；K=2各分量按尺度从小到大标记。', '',
        '| UID | epoch | 模型 | 预算e^-1交点均值 | 第0分量shape均值 | 预算RMSE₀₋₅₀ | 总概率连乘RMSE₀₋₅₀ |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for state in ['uid_zero','uid']:
        for epoch in [1,3,5]:
            for model in MODELS:
                frame=groups.filter((pl.col('uid')==state)&(pl.col('epoch')==epoch)&(pl.col('model')==model)&(pl.col('grouping')=='global'))
                lines.append(f'| {state} | {epoch} | {model} | '+' | '.join(fmt(frame,f) for f in
                    ['model_effective_scale_mean','model_shape_component0_mean','budget_rmse50','product_rmse50'])+' |')
    lines += ['', '完整每场分量scale/shape/prior与分桶标签见[参数数据](../../runs/plan19_weibull_check/model_session_parameters.parquet)，分布均值及P10/P50/P90见[参数摘要](../../runs/plan19_weibull_check/parameter_distribution_summary.json)。其均值只是摘要，不能代表总体Weibull参数。', '',
        '## 7. 纠正计划中的固定倍率解释', '',
        '若右塔每步继续率恰为常数q，则S_total(r)=S_budget(r)q^r，所以总累计风险H_total(r)=(r/λ)^z−r log q。额外的是线性风险；z≠1时通常不再是相同shape的Weibull，风险比例也随r变化。只有额外风险与r^z成固定比例等特殊条件，才有固定尺度倍率；常数q不能保证这一条件。实际stay还随内容历史和列表变化，不能用其请求均值构造“扣除分摊后完全对齐”的lambda_eff。', '',
        '因此本轮不提供未经识别的固定alpha校正，也不把左塔尺度偏大判作数学错误。分别展示真实曲线、预算曲线和最终继续概率诊断，匹配程度由数据说话。', '',
        '第19份对UID消融的解释同样需要收窄：Fusion在固定5轮协议下去UID的AUC下降更大，是字段依赖的证据；不能推出它的追赶“全靠死记硬背”、双塔“几乎零依赖”，也不能用两种AUC降幅的比值定义通用抗冲击能力。当前是同批用户的时间切分，不是冷启动用户泛化实验。', '',
        '## 8. 已验证的范围与复现', '',
        f'- 首屏34个预算输入逐场固定；每场一次参数读取。独立NumPy公式在72组完整test的已观察进度上复算log S，最大绝对误差{manifest["maximum_independent_log_survival_error"]:.3g}；没有进行优化器更新。',
        f'- 读取前后{len(manifest["source_hashes"])}份来源hash不变。无新模型、无数据重建、无辅助Loss、无test选点。',
        '- 模型来源：已有六配置×三seed连续5轮实验，batch4096、421步/轮，一个AdamW，预算LR .005、内容/Fusion LR .001，decay1e-6及Embedding/bias/一维参数零decay；本轮只分析其中4种预算模型与Fusion概率对照。',
        '- [实验说明](实验说明.md) · [核对日志](../../runs/plan19_weibull_check/logs/check.log) · [来源与参数读取审计](../../runs/plan19_weibull_check/manifest.json) · [经验拟合](../../runs/plan19_weibull_check/empirical_fits.json)。',
        '- [独立优化器及原始事件核对](../../runs/plan19_weibull_check/independent_checks.json) · [6项测试](../../runs/plan19_weibull_check/logs/tests.log)。',
        '- [每屏risk曲线](../../runs/plan19_weibull_check/risk_curves.parquet) · [全局与分桶曲线](../../runs/plan19_weibull_check/curves.parquet) · [桶级相关](../../runs/plan19_weibull_check/group_correlations.json)。',
        '- [统计与参数读取脚本](../../pipeline/check_weibull_behavior.py) · [报告脚本](../../pipeline/report_weibull_behavior.py)。', '',
        '```bash','uv run python -m workspace.kr_time_budget.pipeline.check_weibull_behavior --output workspace/kr_time_budget/runs/plan19_weibull_check_new',
        'uv run python -m workspace.kr_time_budget.pipeline.report_weibull_behavior','```','']
    import os
    relative=os.path.relpath(run,report)
    (report/'核对报告.md').write_text('\n'.join(lines).replace('../../runs/plan19_weibull_check',relative))
    print('REPORT',report/'核对报告.md')
    print('GLOBAL',primary.select('uid','model','seed','model_effective_scale_mean','model_shape_component0_mean','budget_rmse50','product_rmse50'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=WORKSPACE/'runs/plan19_weibull_check')
    args=parser.parse_args();generate(args.run.resolve(),WORKSPACE/'回复/19韦伯')


if __name__=='__main__':main()

"""Audit full v4 coverage and generate the current workspace report."""
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from workspace.kr_time_budget.pipeline.run_exit import grouped_auc
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores

ROOT=Path('workspace/kr_time_budget')


def user_bootstrap(left,right,seed=2026):
    joined=left.join(right.select('request_id',pl.col('prediction').alias('other')),on='request_id',validate='1:1')
    ranked=joined.with_columns(pl.col('prediction').rank('average').over('user_id_raw').alias('r1'),pl.col('other').rank('average').over('user_id_raw').alias('r2'))
    g=ranked.group_by('user_id_raw').agg(pl.len().alias('n'),pl.col('y_continue').sum().alias('p'),
        (pl.col('r1')*pl.col('y_continue')).sum().alias('r1'),(pl.col('r2')*pl.col('y_continue')).sum().alias('r2'))
    g=g.filter((pl.col('p')>0)&(pl.col('p')<pl.col('n'))).sort('user_id_raw')
    delta=((g['r1']-g['r2'])/(g['p']*(g['n']-g['p']))).to_numpy();w=g['n'].to_numpy()
    idx=np.random.default_rng(seed).integers(0,len(w),size=(2000,len(w)))
    values=(delta[idx]*w[idx]).sum(1)/w[idx].sum(1)
    return {'user_gauc_delta':float((delta*w).sum()/w.sum()),'paired_user_bootstrap_95':np.quantile(values,[.025,.975]).tolist(),'resamples':2000,'seed':seed}


def main():
    out=ROOT/'runs/exit_v4';summary=json.loads((out/'summary.json').read_text());old=json.loads((ROOT/'runs/exit_v3/summary.json').read_text())
    test=pl.read_parquet(ROOT/'data/v3/features_raw.parquet').filter(pl.col('split')=='test').select('request_id','user_id','y_continue')
    audit={'models':{},'comparisons':{}};frames={}
    for mode,metric in summary.items():
        assert metric['full_train'] and metric['full_test']
        p=pl.read_parquet(out/mode/'predictions.parquet');frames[mode]=p
        assert p.height==p['request_id'].n_unique()==test.height
        joined=p.join(test.rename({'user_id':'expected_user','y_continue':'expected_y'}),on='request_id',validate='1:1')
        assert (joined['y_continue']==joined['expected_y']).all() and (joined['user_id_raw']==joined['expected_user']).all()
        check=scores(p)
        assert abs(check['overall.auc']-metric['overall.auc'])<1e-12
        initial=json.loads((out/mode/'manifest.json').read_text())
        assert initial['common_v3_state_exact']
        selections=[{'phase':f.parent.name,**json.loads(f.read_text())} for f in (out/mode).glob('*/selection.json')]
        audit['models'][mode]={'rows':p.height,'initial':initial,'selection':selections}
    assert len({m['initial']['initial_sha256'] for m in audit['models'].values()})==1
    assert len({m['initial']['train_order_sha256'] for m in audit['models'].values()})==1
    audit['comparisons']['v4_ours_minus_fusion']=user_bootstrap(frames['ours'],frames['fusion'])
    audit['comparisons']['v4_ours_minus_v3_ours']=user_bootstrap(frames['ours'],pl.read_parquet(ROOT/'runs/exit_v3/ours/predictions.parquet'))
    mechanism=json.loads((out/'mechanism_audit.json').read_text());switch=json.loads((out/'ours/stage_switch.json').read_text())
    assert switch['transition_full_valid_rows']==57872 and switch['transition_prediction_max_abs_error']==switch['transition_logloss_difference']==switch['gamma_max_change']==0
    for mode in ['budget','ours']:
        assert summary[mode]['survival_diagnostics']['gamma_increase_with_nondecreasing_time']==0
        assert summary[mode]['survival_diagnostics']['S_increase_with_nondecreasing_time']==0
    audit['mechanism']=mechanism;audit['stage_switch']=switch
    audit['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pipeline').glob('*v4.py')}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    m3,m4=mechanism['v3_ours'],mechanism['v4_ours']
    delta=audit['comparisons']['v4_ours_minus_fusion']
    lines=['# KR v4：Argon 5 反馈执行报告','',
        '**两处修改均已落实：相乘模型的AUC/LogLoss略优于Fusion，但User-GAUC仍低于Fusion。内容饱和显著缓解；反馈所预测的User-GAUC达到0.61+没有实现。**','',
        '依据：[5反馈 5.md](../../Argon-plans/5反馈%205.md)。本次复用原v3数据，不重新预处理、不改特征/切分；旧报告见 [v3执行报告](history/20261116_kr-time-budget-v3-report.md)。','',
        '## 本次改动','',
        '1. 预算改为单威布尔：scale=softplus(raw_scale)×30+0.5，shape=clamp(1.05+softplus(raw_shape),max=5)。原始输出从6维改2维。累计风险差用float64与expm1/log1p稳定计算，避免大数相减造成gamma回弹。',
        '2. Budget模式及Ours预热均预测gamma×s0，s0=sigmoid(2.6)=0.9308615923。进入Ours第二阶段时，将未参与预热的内容头最后一层置零权重、bias=2.6，冻结预算分支。',
        '3. Content/Fusion初始化和训练完全保留v3；零头只在Ours阶段转换时设置。四模型同名共有参数与v3逐tensor核对（排除预算末层），Content/Fusion完整测试预测与v3逐值相同，最大误差0。',
        '4. 沿用rosen BaseTrainer、EarlyStopper、TwoStageRunner；新增全valid零跳变检查、gamma单调断言和参数分布。所有改动位于本workspace。','',
        '## 数据与训练','',
        'KR 1K；train=20220408–20220506，1,721,745条/1000用户；valid=20220507，57,872条/894用户；test=20220508，48,898条/855用户。沿用54字段、Session整场不交叉、训练过去日先验、评估0506快照。数据：[manifest](data/v3/manifest.json)。',
        '基线各1epoch、Ours预热1epoch+内容1epoch；batch4096，seed2026，AdamW lr=1e-3，普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减；无scheduler/EMA/累积。每epoch 25%/50%/100%完整valid，以LogLoss选checkpoint；Ours还比较stage2起点。最终只评选中checkpoint的完整test。基线执行421步，Ours421+421步，本轮选中均为阶段第421步。',
        '本轮同数据、同顺序、共有初始化相同，但同时修改IFR、预热目标和阶段衔接，是修正版结构对照；不能将变化单独归因给其中一项。Ours比单阶段模型多一轮训练，不宣称等训练量因果比较。','',
        '## 完整测试结果','',
        '| 模型 | v3 AUC | v4 AUC | v3 User-GAUC | v4 User-GAUC | v3 LogLoss | v4 LogLoss |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,m in summary.items():
        o=old[mode];lines.append(f"| {mode} | {o['overall.auc']:.6f} | {m['overall.auc']:.6f} | {o['user_gauc']['auc']:.6f} | {m['user_gauc']['auc']:.6f} | {o['logloss']:.6f} | {m['logloss']:.6f} |")
    lines += ['',f"v4 Ours−Fusion：AUC {summary['ours']['overall.auc']-summary['fusion']['overall.auc']:+.6f}，LogLoss {summary['ours']['logloss']-summary['fusion']['logloss']:+.6f}；User-GAUC {delta['user_gauc_delta']:+.6f}，用户配对bootstrap 95%区间 [{delta['paired_user_bootstrap_95'][0]:.6f}, {delta['paired_user_bootstrap_95'][1]:.6f}]。单seed不作机制因果结论。",'',
        '| 模型 | v4 Intra-Step GAUC |','| --- | ---: |']
    for mode,m in summary.items():lines.append(f"| {mode} | {m['intra_step_gauc']['auc']:.6f} |")
    lines += ['', 'User-GAUC为用户内AUC、有效样本数加权，排除单类用户；覆盖823/855用户、48,855/48,898条。Intra-Step按精确req_idx分组，覆盖131组、46,797条。分域指标在summary.json。',
        '无训练排序诊断沿用同一个test：-elapsed_min User-GAUC=0.605780，-req_idx=0.592362。Budget达到0.591388，仍未超过负时钟；Ours为0.473160，仍未解决最终预测的用户内排序。','',
        '## 机制核对','',
        f"- 原v3并非只是假设shape<1：Ours两个分量shape<1的样本分量占比实测为{mechanism['v3_ours']['component_shape_below_one_fraction']:.2%}。其测试中43,935对时间不回退的相邻Request里，gamma上升{m3['adjacent_gamma_increases']:,}次。该证据支持检查递减风险率，但不证明所有排序差异都由它造成。",
        '- v4在同一43,935对相邻Request上，S上升0次、gamma上升0次；原时钟回退仍为1,682次，未改公式。另以256个固定背景、0/1/5/10/20/60/120分钟网格核对gamma，无违规。',
        '- 完整57,872条valid在预热最优状态→stage2第0步的预测逐样本精确一致；最大误差0、LogLoss差0。第二阶段结束预算全部state与snapshot逐tensor相同，probe gamma漂移0。乘积误差0。',
        f"- Ours测试stay均值：v3 {m3['stay_mean']:.6f} → v4 {m4['stay_mean']:.6f}；stay>0.99：{m3['stay_over_099']:.2%} → {m4['stay_over_099']:.2%}；stay<0.85：{m4['stay_below_085']:.2%}。已释放上下调节空间。",
        f"- v4测试gamma均值{m4['gamma_mean']:.6f}；Budget预测均值{mechanism['v4_budget']['prediction_mean']:.6f}，Ours预测均值{m4['prediction_mean']:.6f}，真实测试继续率0.932901。",
        '- 22项相关测试通过；包括数值公式、极端时间、IFR双单调、零跳变、冻结、同初始化和原数据协议。','',
        '## 与反馈预测不同的地方','',
        's0=0.930862低于训练继续率0.936478，因此预热输出gamma×s0≤s0，存在显式上限。实际gamma升到约0.99546，而不是反馈估计的0.97～0.98。我们没有静默更换s0或提前调参。该约束使预热倾向提高gamma；不能把这个gamma直接视为已识别的纯时间预算概率。',
        'IFR保证固定Session背景下gamma随时间非增，不保证同一用户跨不同Session的排序；跨场参数不同，右塔也可改变最终排序。v4 Ours时间不回退的相邻Request中，最终乘积仍上升24,723次，因为内容特征随屏改变。这符合当前模型定义，不是gamma单调实现失败。',
        'Pooled与User-GAUC方向差异、IFR约束以及单侧乘积限制都值得诊断，但不能据此推导改IFR后User-GAUC必为1−旧值。当前数据不支持“已彻底消除过拟合”或“保证打穿全部指标”的结论。','',
        '## 入口','',
        '- [v4模型](pipeline/time_budget_v4.py)、[v4训练](pipeline/run_exit_v4.py)、[v4报告生成](pipeline/report_v4.py)、[v4测试](tests/test_v4.py)。',
        '- [全日志](runs/exit_v4/run.log)、[汇总指标](runs/exit_v4/summary.json)、[审计](runs/exit_v4/audit.json)、[机制核对](runs/exit_v4/mechanism_audit.json)、[零跳变与冻结](runs/exit_v4/ours/stage_switch.json)。',
        '- [复现说明](复现说明.md)、[总入口](README.md)。','']
    # Markdown links may contain spaces; keep the filesystem spelling for local navigation.
    note='\n'.join(lines).replace('5反馈%205.md','5反馈 5.md')
    (ROOT/'history/20261116_kr-time-budget-v4-report.md').write_text(__import__('re').sub(r'\]\(([^)]+)\)',lambda m: ']('+('../' if '://' not in m.group(1) else '')+m.group(1)+')',note))
    print(ROOT/'history/20261116_kr-time-budget-v4-report.md')
    print(json.dumps(audit['comparisons'],indent=2))

if __name__=='__main__':main()

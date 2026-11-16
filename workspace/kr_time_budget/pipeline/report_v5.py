"""Full-coverage audit and Argon 6 execution report, without retraining."""
import hashlib
import json
from pathlib import Path
import polars as pl
from workspace.kr_time_budget.pipeline.report_v4 import user_bootstrap
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores

ROOT=Path('workspace/kr_time_budget')


def main():
    out=ROOT/'runs/exit_v5'
    summary=json.loads((out/'summary.json').read_text())
    previous=json.loads((ROOT/'runs/exit_v4/summary.json').read_text())
    mechanisms=json.loads((out/'mechanism_audit.json').read_text())
    switch=json.loads((out/'ours/stage_switch.json').read_text())
    expected=pl.read_parquet(ROOT/'data/v3/test.parquet',columns=['request_id','user_id_raw','y_continue'])
    audit={'models':{},'comparisons':{},'stage_switch':switch,'mechanisms':mechanisms}
    predictions={}
    for mode,metric in summary.items():
        p=pl.read_parquet(out/mode/'predictions.parquet');predictions[mode]=p
        assert metric['full_train'] and metric['full_test']
        assert p.height==p['request_id'].n_unique()==expected.height
        joined=p.join(expected.rename({'user_id_raw':'expected_user','y_continue':'expected_y'}),on='request_id',validate='1:1')
        assert (joined['y_continue']==joined['expected_y']).all() and (joined['user_id_raw']==joined['expected_user']).all()
        recalculated=scores(p)
        for name in ['overall.auc','logloss']:assert abs(recalculated[name]-metric[name])<1e-12
        initial=json.loads((out/mode/'manifest.json').read_text())
        old=json.loads((ROOT/f'runs/exit_v4/{mode}/manifest.json').read_text())
        assert initial['train_order_sha256']==old['train_order_sha256'] and initial['common_v3_state_exact']
        selection=[{'phase':f.parent.name,**json.loads(f.read_text())} for f in sorted((out/mode).glob('*/selection.json'))]
        audit['models'][mode]={'rows':p.height,'initial':initial,'selection':selection}
        if mode in ['content','fusion']:
            ref=pl.read_parquet(ROOT/f'runs/exit_v4/{mode}/predictions.parquet')
            assert p['request_id'].equals(ref['request_id']) and p['prediction'].equals(ref['prediction'])
        if mode in ['budget','ours']:
            assert metric['survival_diagnostics']['gamma_increase_with_nondecreasing_time']==0
            assert metric['survival_diagnostics']['S_increase_with_nondecreasing_time']==0
    assert len({a['initial']['initial_sha256'] for a in audit['models'].values()})==1
    assert switch['transition_full_valid_rows']==57872
    assert switch['transition_prediction_max_abs_error']==switch['transition_logloss_difference']==switch['gamma_max_change']==0
    audit['comparisons']['ours_minus_fusion']=user_bootstrap(predictions['ours'],predictions['fusion'])
    audit['comparisons']['ours_minus_v4_ours']=user_bootstrap(predictions['ours'],pl.read_parquet(ROOT/'runs/exit_v4/ours/predictions.parquet'))
    cfg=json.loads((out/'config.json').read_text())
    assert cfg['dataset_manifest_sha256']==json.loads((ROOT/'runs/exit_v4/config.json').read_text())['dataset_manifest_sha256']
    audit['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pipeline').glob('*v5.py')}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    own,fusion=summary['ours'],summary['fusion'];mech=mechanisms['v5_ours'];oldmech=mechanisms['v4_ours']
    better={name:own[name]>fusion[name] if name=='overall.auc' else own[name]<fusion[name] for name in ['overall.auc','logloss']}
    gauc_better=own['user_gauc']['auc']>fusion['user_gauc']['auc']
    lines=['# KR v5：Argon 6 执行报告','',
           f"**已完成指定两处改动及四模型全量训练。相对Fusion，Ours的AUC{'更高' if better['overall.auc'] else '更低'}、LogLoss{'更低' if better['logloss'] else '更高'}、User-GAUC{'更高' if gauc_better else '更低'}；具体数值如下。**",'',
           '依据：[Argon-plans/6.md](../../Argon-plans/6.md)。复用data/v3；旧版见 [v4执行报告](history/20261116_kr-time-budget-v4-report.md)。','',
           '## 只改了什么','',
           '1. PRIOR_STAY_LOGIT由2.6改为3.5，s0=sigmoid(3.5)=0.970687747；Budget及Ours预热预测gamma×s0。Ours进入第二阶段时，内容末层仍为零权重，bias同步为3.5，以保持切换逐样本一致。',
           '2. 只在Ours内容塔内将elapsed_min、req_idx对应表示置零。54字段仍完整读取；elapsed_min走左塔生存公式。Fusion继续全字段，Content继续原B3/B4路由。其他字段不额外屏蔽。',
           '单威布尔IFR、数值公式、预算静态路由及冻结、特征编码、切分、数据顺序和训练协议全部沿用v4。Content/Fusion重新训练后的48,898条test预测与v4逐值相同。','',
           '## 数据、训练与评估口径','',
           'KR 1K：train为20220408–20220506，1,721,745条/1000用户；valid为20220507，57,872条/894用户；test为20220508，48,898条/855用户。沿用54字段、完整Session切分、过去日训练先验与0506评估快照，未重新构建数据，manifest哈希与v4相同。',
           '基线1epoch；Ours预算预热1epoch、冻结预算后内容1epoch。batch4096、seed2026、AdamW lr=1e-3；普通矩阵weight_decay=1e-6，Embedding/bias/一维参数零衰减；无累积、scheduler、EMA。每epoch的25%/50%/100%完整valid，以LogLoss选checkpoint，Ours额外比较stage2起点。只测试选中checkpoint；实际基线421步，Ours421+421步。',
           '四模型同初始全state hash、同train物理顺序；与v4相比，除prior_stay常数外共有参数初始化不变。本轮同时改变先验和内容路由，属于组合修改对照，不能将增益单独归因其中一项。Ours多一个预热epoch，不宣称与基线等训练量因果比较。','',
           '## 完整测试结果','',
           '| 模型 | v4 AUC | v5 AUC | v4 User-GAUC | v5 User-GAUC | v4 LogLoss | v5 LogLoss |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,m in summary.items():
        prev=previous[mode]
        lines.append(f"| {mode} | {prev['overall.auc']:.6f} | {m['overall.auc']:.6f} | {prev['user_gauc']['auc']:.6f} | {m['user_gauc']['auc']:.6f} | {prev['logloss']:.6f} | {m['logloss']:.6f} |")
    comp=audit['comparisons']['ours_minus_fusion'];ci=comp['paired_user_bootstrap_95']
    lines += ['',f"Ours−Fusion：AUC {own['overall.auc']-fusion['overall.auc']:+.6f}，LogLoss {own['logloss']-fusion['logloss']:+.6f}，User-GAUC {comp['user_gauc_delta']:+.6f}（用户配对bootstrap 2000次95%区间 [{ci[0]:.6f}, {ci[1]:.6f}]；单seed对照）。",'',
              '| 模型 | v5 Intra-Step GAUC |','| --- | ---: |']
    for mode,m in summary.items():lines.append(f"| {mode} | {m['intra_step_gauc']['auc']:.6f} |")
    lines += ['', 'User-GAUC按用户内AUC、有效样本数加权；排除单类用户，覆盖823/855用户、48,855/48,898条。Intra-Step按精确req_idx分组，覆盖详见metrics。分域结果位于summary.json。无训练负时钟基线仍为User-GAUC=0.605780，不随版本改变。','',
              '## 机制核对','',
              f"- 完整valid的{switch['transition_full_valid_rows']:,}条样本切换误差0、LogLoss差0；预算全state精确冻结，probe gamma漂移0，最终乘积误差0。",
              '- v5 Ours完整test上，分别把elapsed_min加10分钟、req_idx加5，内容输入最大变化0、stay最大变化0。由此验证两字段确已屏蔽；Fusion仍保留时钟。',
              f"- v5测试Ours gamma均值{mech['gamma_mean']:.6f}、标准差{mech['gamma_std']:.6f}、范围[{mech['gamma_min']:.6f},{mech['gamma_max']:.6f}]；v4均值{oldmech['gamma_mean']:.6f}、标准差{oldmech['gamma_std']:.6f}。预热最优状态在完整valid的gamma均值为{switch['transition_gamma_mean']:.6f}。",
              f"- Ours内容stay均值{mech['stay_mean']:.6f}；stay>0.99占{mech['stay_over_099']:.2%}，stay<0.85占{mech['stay_below_085']:.2%}；最终预测均值{mech['prediction_mean']:.6f}，真实test继续率{mech['label_mean']:.6f}。",
              f"- 时间不回退的{mech['adjacent_nondecreasing_time_pairs']:,}对相邻屏中，gamma上升{mech['gamma_increases']}次；最终乘积上升从v4的{oldmech['prediction_increases']:,}次变为v5的{mech['prediction_increases']:,}次。原屏末时钟回退1,682次，按原公式保留。",
              '- 最终选中checkpoint重新加载后，完整test预测与已保存结果逐值一致。25项相关测试通过，包含初始化、时钟掩码、零跳变、冻结与原数据协议。','',
              '## 解释边界','',
              '已按要求屏蔽两个直接时钟字段。前序反馈、累计统计、历史标签等仍保留，可能间接表达进度；当屏内容也逐屏变化。因此“屏蔽时钟”不等于保证整个内容塔或最终乘积随时间单调。相邻乘积上升本身也不是对劣化原因的因果证明。',
              's0提高后的gamma应根据实测分布解释；没有仅凭先验设定将其称为已识别的纯时间预算。未额外修改路由、调参或扩展实验，以免偏离第6份的两处最小改动。','',
              '## 代码与产物','',
              '- [v5模型代码](pipeline/time_budget_v5.py)、[训练入口](pipeline/run_exit_v5.py)、[机制审计](pipeline/audit_v5.py)、[报告生成](pipeline/report_v5.py)、[测试](tests/test_v5.py)。',
              '- [训练日志](runs/exit_v5/run.log)、[指标](runs/exit_v5/summary.json)、[逐项审计](runs/exit_v5/audit.json)、[机制数据](runs/exit_v5/mechanism_audit.json)、[零跳变/冻结](runs/exit_v5/ours/stage_switch.json)、[测试日志](runs/exit_v5/logs/tests.log)。',
              '- [总入口](README.md)、[复现说明](复现说明.md)。','']
    (ROOT/'执行报告.md').write_text('\n'.join(lines))
    print(ROOT/'执行报告.md');print(json.dumps(audit['comparisons'],indent=2))

if __name__=='__main__':main()

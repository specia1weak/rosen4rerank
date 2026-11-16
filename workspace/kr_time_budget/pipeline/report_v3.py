"""Read-only audit and human report for the Argon 4 experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss
from workspace.kr_time_budget.pipeline.run_exit import grouped_auc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('workspace/kr_time_budget/data/v3'))
    ap.add_argument('--output',type=Path,default=Path('workspace/kr_time_budget/runs/exit_v3'))
    args=ap.parse_args();root,out=args.data,args.output
    r=pl.read_parquet(root/'features_raw.parquet');test=r.filter(pl.col('split')=='test');train=r.filter(pl.col('split')=='train')
    source=json.loads((root/'manifest.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    audit={'constant_train_rate':train['y_continue'].mean(),
           'constant_test_logloss':float(log_loss(test['y_continue'],np.full(test.height,train['y_continue'].mean()),labels=[0,1]))}
    for col in ['req_idx','elapsed_min']:
        p=-test[col].to_numpy();y=test['y_continue'].to_numpy()
        audit['negative_'+col]={'auc':float(roc_auc_score(y,p)),'user_gauc':grouped_auc(y,p,test['user_id'].to_numpy())}
    audit['models']={}
    for mode,metric in summary.items():
        pred=pl.read_parquet(out/mode/'predictions.parquet')
        assert pred['request_id'].n_unique()==test.height==pred.height
        joined=pred.join(test.select('request_id',pl.col('y_continue').alias('expected_y'),pl.col('user_id').alias('expected_user')),on='request_id',validate='1:1')
        assert (joined['y_continue']==joined['expected_y']).all()
        assert (joined['user_id_raw']==joined['expected_user']).all()
        assert metric['full_train'] and metric['full_test']
        phases=[{'phase':p.parent.name,**json.loads(p.read_text())} for p in sorted((out/mode).glob('*/selection.json'))]
        audit['models'][mode]={'prediction_rows':pred.height,'unique_requests':pred['request_id'].n_unique(),
                             'selected':phases,'initial':json.loads((out/mode/'manifest.json').read_text())}
        if mode=='ours':audit['models'][mode]['components']={'gamma_mean':pred['gamma'].mean(),'content_stay_mean':pred['content_stay'].mean(),'stay_over_099':(pred['content_stay']>.99).mean()}
    assert len({v['initial']['initial_sha256'] for v in audit['models'].values()})==1
    assert len({v['initial']['train_order_sha256'] for v in audit['models'].values()})==1
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# KR v3：Argon 4 特征集执行结果','',
           '已按 `Argon-plans/4.特征集敲定.md` 构建54字段日期切分数据，并完成四模型 full train/full test。',
           '**本轮相乘模型未超过 Fusion；所有模型 User-GAUC 还低于简单负时钟基线，不能仅凭整体 AUC 宣称方案有效。**','',
           '## 数据与训练口径','',
           '| 切分 | Request | 用户 | 日期 |','| --- | ---: | ---: | --- |']
    for split,date in [('train','20220408–20220506'),('valid','20220507'),('test','20220508')]:
        s=source['splits'][split];lines.append(f"| {split} | {s['rows']:,} | {s['users']} | {date} |")
    lines += ['', '完整日志先切场、倒切M=6，再过滤训练范围Top5 Session场景；主tab取首条。完整保留在单个切分内的Session，跨边界268场单独排除。每用户首尾场排除；不足一屏头部余数参与历史。',
              '训练先验只用严格过去日；验证/测试固定使用截至5月6日快照。用户场均先验m=5平滑，物品CTR/long_view先验α=20。正反馈为每视频任一like/comment/follow/forward。tag保留重复、按时间拼接、原生MultiSparse sum及head截断。',
              '基线各1epoch；Ours预热1epoch、冻结整条预算路径后再1epoch。batch4096、AdamW lr=1e-3，普通矩阵decay=1e-6，Embedding/bias/一维参数零衰减。固定seed2026，无EMA、无scheduler。每epoch 25%/50%/100% full valid，以LogLoss选checkpoint，最后完整test一次。',
              '使用原生BaseTrainer、EvaluatorManager、EarlyStopper、TwoStageRunner；四模型初始全state hash与train物理顺序一致。Ours多一个预热epoch且路由/目标不同，本轮属于结构筛查，不是训练量相同的单变量因果比较。','',
              '## 测试结果','', '| 模型 | Pooled AUC | User-GAUC | LogLoss | Intra-Step GAUC |', '| --- | ---: | ---: | ---: | ---: |']
    for mode,m in summary.items():lines.append(f"| {mode} | {m['overall.auc']:.6f} | {m['user_gauc']['auc']:.6f} | {m['logloss']:.6f} | {m['intra_step_gauc']['auc']:.6f} |")
    lines += ['',f"训练继续率常数预测的测试LogLoss：{audit['constant_test_logloss']:.6f}。User-GAUC排除单类用户，覆盖823/855用户、48,855/48,898条Request，按有效样本数加权。Intra-Step按精确req_idx分组，也排除单类组，覆盖见JSON。",'',
              '| 无训练排序诊断 | Pooled AUC | User-GAUC |','| --- | ---: | ---: |']
    for col in ['req_idx','elapsed_min']:
        a=audit['negative_'+col];lines.append(f"| -{col} | {a['auc']:.6f} | {a['user_gauc']['auc']:.6f} |")
    lines += ['', '负时钟在整体AUC和用户内排序上的方向不同，说明只看跨用户的Pooled AUC会掩盖用户内排序不足。该诊断无需训练，也没有用它重新选择模型。新日期切分与v2五折用户切分不同，不能把两版数字直接当成特征增益。','',
              '## 实现核对','',
              '- 完整54字段，无裸video/author/music ID Embedding；标签通过原生MultiSparse计算，不再将整串标签当单个类别。',
              '- 测试预测与原始request_id/user_id/label逐条核对，48,898条完整且唯一；Session无跨切分，输入无空值。',
              '- 三个单阶段基线实际各执行421步；Ours执行421+421步。所有本轮选中checkpoint都在对应阶段第421步，详见selection.json。',
              '- 每条有效Request恰好6个事件，事件无重复归属；验证/测试用户都有自身过去场历史。测试slate事件中约61.48%没有过去日物品记录，质量先验回退为总体率。',
              '- Ours第二阶段预算全部state与第一阶段snapshot逐tensor一致，probe gamma最大变化0；预测乘积误差0。',
              '- 原始测试屏末时钟回退1,682次；时钟未回退却S上升的次数为0。未静默改时钟。',
              '- 18项测试通过，包括过去日先验、前屏历史、标签拆分、参数独立所有权、同初始化和冻结验证。','']
    comp=audit['models']['ours']['components']
    lines += [f"Ours测试gamma均值{comp['gamma_mean']:.6f}，content_stay均值{comp['content_stay_mean']:.6f}，stay>0.99占{comp['stay_over_099']:.2%}。内容乘子仍接近1；本次未证明冻结预算×内容残差有效。",'',
              '## 产物与复现','',
              '- 数据：[manifest](data/v3/manifest.json)、[数据审计](data/v3/data_audit.json)。',
              '- 代码：[预处理](pipeline/build_requests_v3.py)、[Schema](pipeline/kr_request_schema.py)。',
              '- 代码：[模型](pipeline/time_budget_v3.py)、[训练](pipeline/run_exit_v3.py)。',
              '- 训练：[全日志](runs/exit_v3/run.log)、[指标](runs/exit_v3/summary.json)、[审计](runs/exit_v3/audit.json)。',
              '- 各模型：`workspace/kr_time_budget/runs/exit_v3/{content,budget,fusion,ours}/`；含checkpoint、预测、选择记录、分域结果。',
              '- 数据日志：`workspace/kr_time_budget/data/v3/logs/build.log`；测试：同目录tests-all.log。',
              '- 配置与命令：[复现说明](复现说明.md)。','']
    report=Path('workspace/kr_time_budget/history/20261116_kr-time-budget-v3-report.md');report.parent.mkdir(parents=True,exist_ok=True);report.write_text(__import__('re').sub(r'\]\(([^)]+)\)', lambda m: '](' + ('../' if '://' not in m.group(1) else '') + m.group(1) + ')', '\n'.join(lines)))
    print(report)

if __name__=='__main__':main()

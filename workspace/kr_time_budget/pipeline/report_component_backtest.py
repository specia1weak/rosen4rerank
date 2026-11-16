"""Read checked plan16 statistics; report components versus observed current feedback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import polars as pl
from scipy.stats import rankdata
from workspace.kr_time_budget.pipeline.backtest_components import pearson

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/plan16_component_backtest_checked'
REPORT=ROOT/'回复/16 回测'
SEEDS=[2026,2027,2028]
STAYS=['single_step/content_stay','single_step_pace/content_stay','mixture_step/content_stay','mixture_step_pace/content_stay']
COMPARE=STAYS+['fusion/prediction','fusion_pace/prediction']
LABELS={
    'single_step/content_stay':'单威布尔stay', 'single_step_pace/content_stay':'单威布尔＋pace stay',
    'mixture_step/content_stay':'混合威布尔stay','mixture_step_pace/content_stay':'混合＋pace stay',
    'fusion/prediction':'Fusion预测','fusion_pace/prediction':'Fusion＋pace预测',
    'mixture_step_pace/gamma':'混合＋pace gamma','mixture_step_pace/log_survival':'混合＋pace log S',
    'mixture_step_pace/prediction':'混合＋pace最终乘积',
    'baseline/step':'屏号','baseline/elapsed':'累计分钟','baseline/pace':'pace',
    'baseline/prior_ctr':'输入物品CTR先验','baseline/prior_longview':'输入物品长播先验'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    args=parser.parse_args()
    run=args.run
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='PASS'
    assert manifest['feedback_audit']['rows']==48898
    assert manifest['feedback_audit']['previous_screen_feedback_crosscheck_rows']==45617
    assert hashlib.sha256((ROOT/'pipeline/backtest_components.py').read_bytes()).hexdigest()==manifest['code_sha256']
    metrics=pl.read_parquet(run/'metrics.parquet')
    aggregates=pl.read_parquet(run/'aggregates.parquet')
    assert metrics.height==manifest['metric_rows']
    lookup={(row['budget'],row['scope'],row['score'],row['target'],row['metric']):row for row in aggregates.to_dicts()}
    raw={(row['budget'],row['seed'],row['scope'],row['score'],row['target'],row['metric']):row for row in metrics.to_dicts()}

    def get(score,target='curr_longview_rate',scope='all',method='spearman',budget=3):
        return lookup[(budget,scope,score,target,method)]

    def value(score,target='curr_longview_rate',scope='all',method='spearman',budget=3):
        return get(score,target,scope,method,budget)['mean']

    def fmt(score,target='curr_longview_rate',scope='all',method='spearman',budget=3):
        row=get(score,target,scope,method,budget)
        if row['mean'] is None: return '无定义'
        return f"{row['mean']:.6f} ± {row['seed_std']:.6f}"

    # 原则检查：同屏号内的屏号是常量，不能产生任何有限相关。
    invalid=metrics.filter((pl.col('score')=='baseline/step')&pl.col('metric').is_in(['within_step','within_user_step'])&pl.col('value').is_not_null())
    assert invalid.is_empty()
    high='mixture_step_pace/content_stay'
    fusion='fusion/prediction'
    lines=['# 第16份回测：拆出的组件能否代表当屏参与度？', '',
        '**结论：本次回测不支持计划中“右塔必胜、content_stay能直接代表当屏内容质量”的预期。** 三seed、最终3epoch预算下，右塔stay与当屏长播率、点击率、观看时长均为负相关；只看退出屏、同屏号或同用户内，主要负方向仍在。不能把“比Fusion稍少负相关”写成“正向内容质量识别成功”。', '',
        f"主目标当屏长播率：全集混合＋pace stay的Spearman为**{fmt(high)}**，Fusion为**{fmt(fusion)}**；退出屏则分别为**{fmt(high,scope='exit')}**、**{fmt(fusion,scope='exit')}**。",
        '', '## 1. 实验范围与已有模型口径', '',
        '复用第13份的全部36组已保存预测：6配置×3seed×1/3epoch预算；3epoch为主，1epoch为对照。原模型完整train=1,721,745、valid=57,872、test=48,898；KR v3按日期切分，test为20220508、855用户。原轨迹连续3epoch/1263步，按预算内完整valid LogLoss选checkpoint。', '',
        '本次**没有训练、改变checkpoint选择、重新模型推理、修改特征或重建训练集**。只读关联现有test预测与当屏事件，并计算相关/AUC诊断。使用到的45份原数据/预测/配置文件做了前后内容哈希核对。没有把下游Generator/Evaluator训练混入这次回测。', '',
        '| 子集 | 请求数 | 含义 |','| --- | ---: | --- |']
    scope_labels={'all':'全部test','exit':'退出屏，y_continue=0','continue':'继续屏，y_continue=1',
                  'first_or_30plus':'第1屏与≥30屏混合','first':'仅第1屏','30plus':'仅≥30屏'}
    for scope,rows in manifest['scopes'].items(): lines.append(f'| {scope} | {rows:,} | {scope_labels[scope]} |')
    lines += ['', '## 2. 当屏反馈从哪里来，关联是否正确', '',
        '计划假设requests_base包含当屏计数，但实际仅保留前序反馈。我们使用`request_events.event_id`回连`events`，再按request_id聚合当屏事件；没有把`prev1_*`误当本屏真值。', '',
        '- 48,898个请求恰好对应293,388个唯一事件，每屏6条；chunk0不产生本屏监督样本。',
        '- 当屏6个video_id按event_idx排序后，逐请求与requests_base.slate_items一致。',
        '- 用户、session、屏号、label匹配；以末事件时间及其截断播放重算累计分钟，与原请求表逐值对应。',
        '- 独立时序交叉检查：本屏click/longview计数，等于下一屏prev1字段；45,617条可检查请求全部通过。', '',
        '| 目标 | 定义 | 用途 |','| --- | --- | --- |',
        '| curr_longview_rate | 当屏long_view标志之和/6 | 主目标 |',
        '| curr_click_rate | 当屏is_click之和/6 | 辅助目标 |',
        '| curr_watch_min | 当屏非负原始play_time_ms之和/60000 | 观看时长 |',
        '| curr_watch_clip_min | 当屏play_clip之和/60000，单视频最多5分钟 | 重播/极值敏感性 |',
        '| curr_mean_watch_fraction | 逐视频min(非负播放/截断视频时长,1)的均值 | 内容时长归一化敏感性 |',
        '| curr_any_longview / curr_any_click | 当屏是否至少有1次长播/点击 | 二元AUC与User-AUC |', '',
        '这些是当屏**观察参与度**代理，不是已经独立标注的满意度/内容质量。当前反馈没有直接作为对应请求的输入字段或loss标签，但pace使用当前屏结束时的累计耗时，和观看目标共享来源；用户画像、视频时长、采样筛选也影响反馈，因此不能称为概率意义上的“外部独立真值”。', '',
        '## 3. 指标、方向与分层', '',
        '预期方向固定为“分数高→当屏参与度高”，不因负结果改成1−stay、退出概率或对相关取绝对值。Spearman使用平均rank处理ties。', '',
        '分层相关先在每组内计算Spearman，再按有效组请求数加权：同屏号、同用户、同用户×屏号；分数或目标为常量的组无定义，显式排除并记录覆盖。它不等于因果调整或线性偏相关。分层覆盖随目标/分数变化，尤其退出屏下用户×屏号组较稀疏，须看有效样本数。', '',
        'User-AUC为当屏二元目标的用户内AUC，剔除单标签用户，按请求数加权；与之前继续预测的User-AUC是不同任务，不能相互替代。±均为三个训练seed的样本标准差，不是显著性区间。本轮没有新增bootstrap，也未做因果/等效性检验。', '',
        '## 4. 3epoch主结果：完整test', '',
        '| 分数 | 长播率ρ | 原始观看分钟ρ | 点击率ρ |','| --- | --- | --- | --- |']
    main_scores=COMPARE+['mixture_step_pace/gamma','mixture_step_pace/log_survival','mixture_step_pace/prediction']
    for score in main_scores:
        lines.append('| '+LABELS[score]+' | '+' | '.join(fmt(score,target) for target in ['curr_longview_rate','curr_watch_min','curr_click_rate'])+' |')
    lines += ['',
        '全集上双塔stay确实可能比Fusion“少负一些”，但上述stay和继续预测分数仍与观察参与度反向，不能据此声称右塔已经正向表示内容好坏。最终乘积、gamma和stay都受到继续预测任务的监督；本次发现的是语义代理没有按预期成立。log S的相关方向不同，不能把它与条件gamma混为一个分数。', '',
        '## 5. 退出屏专项：3,281条', '',
        '| 分数 | 长播率ρ | 原始观看分钟ρ | 点击率ρ |','| --- | --- | --- | --- |']
    for score in COMPARE+['mixture_step_pace/gamma','mixture_step_pace/prediction']:
        lines.append('| '+LABELS[score]+' | '+' | '.join(fmt(score,target,scope='exit') for target in ['curr_longview_rate','curr_watch_min','curr_click_rate'])+' |')
    lines += ['',
        '退出屏上的混合＋pace stay比Fusion呈现更强负相关，直接反驳了“退出屏右塔必然更准确反映当屏参与度”的本次预期。退出条件本身存在选择效应，不能把这一结果反过来写成已证明某个真实退出原因。', '',
        '## 6. 控制屏号/用户后的长播相关', '',
        '| 子集 | 分数 | 同屏号内ρ | 同用户内ρ | 同用户×屏号内ρ | 用户×屏号有效请求范围 |',
        '| --- | --- | --- | --- | --- | ---: |']
    for scope in ['all','exit']:
        for score in COMPARE:
            coverage=get(score,scope=scope,method='within_user_step')
            lines.append(f'| {scope} | '+LABELS[score]+' | '+' | '.join(fmt(score,scope=scope,method=method) for method in ['within_step','within_user','within_user_step'])+f" | {coverage['eligible_rows_min']:,}–{coverage['eligible_rows_max']:,} |")
    lines += ['',
        '同屏号内的比较降低了显式屏号差异的影响，同用户内比较降低了固定用户差异；两者合并后仍为负，说明不能只用“跨进度混在一起”解释全部负方向。但稀疏分层、剩余场景差异、历史反馈和退出选择仍存在，不能把它当作严格因果净效应。', '',
        '### 第1屏/深屏混合及各自子集', '',
        '| 子集 | stay与长播ρ | Fusion与长播ρ | stay与观看分钟ρ | Fusion与观看分钟ρ |',
        '| --- | --- | --- | --- | --- |']
    for scope in ['first_or_30plus','first','30plus','continue']:
        lines.append(f'| {scope} | '+ ' | '.join([fmt(high,scope=scope),fmt(fusion,scope=scope),fmt(high,'curr_watch_min',scope),fmt(fusion,'curr_watch_min',scope)])+' |')
    lines += ['',
        '## 7. 输入/时钟正对照与真实反馈深度分布', '',
        '| 分数/基线 | 全集长播ρ | 退出屏长播ρ | 全集观看分钟ρ |','| --- | --- | --- | --- |']
    for score in ['baseline/step','baseline/elapsed','baseline/pace','baseline/prior_ctr','baseline/prior_longview']:
        lines.append('| '+LABELS[score]+' | '+' | '.join([fmt(score),fmt(score,scope='exit'),fmt(score,'curr_watch_min')])+' |')
    lines += ['',
        'pace和物品长播先验与实际长播正相关，说明本次反馈并非对所有可用信号都毫无辨识度；它们是模型可见的正对照，不是新的独立真值。尤其pace与观看共享信息，正相关不能作为解耦成功证据。', '',
        '| 屏号桶 | 请求数 | 真实继续率 | 长播率 | 点击率 | 观看分钟均值 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    buckets=pl.read_parquet(run/'depth_feedback.parquet')
    for name in ['1','2','3','4–5','6–10','11–20','21–50','51+']:
        row=buckets.filter(pl.col('step_bucket')==name).row(0,named=True)
        lines.append(f"| {name} | {row['rows']:,} | {row['continue_rate']:.6f} | {row['curr_longview_rate']:.6f} | {row['curr_click_rate']:.6f} | {row['curr_watch_min']:.6f} |")
    lines += ['',
        '观察到：越深的屏号桶，继续率越高，但长播率、点击率和当屏观看分钟均值反而更低。这解释了为何一个继续概率预测器可以与当屏参与度负相关的一部分；然而同屏号/同用户分层后仍负，不能把所有现象都归给这个总体趋势。', '',
        '## 8. 当屏二元反馈的AUC与User-AUC', '',
        '下面AUC的label是“当前屏至少1个视频长播”，不是“下一屏继续”。分数不做反向转换。0.5以下表示当前分数方向与这个代理目标不一致。', '',
        '| 子集 | 分数 | 当屏长播AUC | 当屏长播User-AUC | User-AUC有效请求范围 |',
        '| --- | --- | --- | --- | ---: |']
    for scope in ['all','exit']:
        for score in COMPARE:
            row=get(score,'curr_any_longview',scope,'user_auc')
            lines.append(f'| {scope} | '+LABELS[score]+' | '+fmt(score,'curr_any_longview',scope,'auc')+' | '+fmt(score,'curr_any_longview',scope,'user_auc')+f" | {row['eligible_rows_min']:,}–{row['eligible_rows_max']:,} |")
    lines += ['', '### 当屏至少一次点击', '',
        '| 子集 | 分数 | 当屏点击AUC | 当屏点击User-AUC |', '| --- | --- | --- | --- |']
    for scope in ['all','exit']:
        for score in [STAYS[1],high,fusion]:
            lines.append(f'| {scope} | '+LABELS[score]+' | '+fmt(score,'curr_any_click',scope,'auc')+' | '+fmt(score,'curr_any_click',scope,'user_auc')+' |')
    lines += ['',
        '全部五个连续目标及每组覆盖，也全部保存在metrics.parquet中；没有只保留长播较有利的结果。原退出预测的User-AUC结论未变，本回测回答的是不同语义。', '',
        '## 9. 1epoch对照与时长定义敏感性', '',
        '| 预算 | 子集 | 混合＋pace stay长播ρ | Fusion长播ρ | 混合＋pace stay观看ρ | Fusion观看ρ |',
        '| --- | --- | --- | --- | --- | --- |']
    for budget in [1,3]:
        for scope in ['all','exit']:
            lines.append(f'| {budget}ep | {scope} | '+' | '.join([fmt(high,scope=scope,budget=budget),fmt(fusion,scope=scope,budget=budget),fmt(high,'curr_watch_min',scope,budget=budget),fmt(fusion,'curr_watch_min',scope,budget=budget)])+' |')
    lines += ['',
        '| 子集 | 分数 | 原始观看ρ | 截断观看ρ | 归一化观看比例ρ |', '| --- | --- | --- | --- | --- |']
    for scope in ['all','exit']:
        for score in [high,'mixture_step/content_stay',fusion]:
            lines.append(f'| {scope} | '+LABELS[score]+' | '+' | '.join(fmt(score,target,scope) for target in ['curr_watch_min','curr_watch_clip_min','curr_mean_watch_fraction'])+' |')
    lines += ['',
        '有无pace、1/3epoch、原始/截断/归一化时长均提供了敏感性检查。这里不把换目标、翻转符号或选择单个seed当作挽救“组件纯度”的方法。', '',
        '## 10. gamma、S与会话进度：结构与统计相关分开看', '',
        '以下直接读三个seed的已有3epoch组件，不做模型推理，报告与屏号/累计分钟的Spearman。log S与S在非下溢时排序相同，使用log S避免浮点下溢把长尾全部压成0。', '',
        '| 分数 | 与屏号ρ | 与累计分钟ρ |','| --- | --- | --- |']
    clock_records=[]
    for score in [high,'mixture_step_pace/gamma','mixture_step_pace/log_survival','mixture_step_pace/prediction',fusion]:
        model,column=score.split('/')
        groups=[]
        for target in ['req_idx_raw','elapsed_min_raw']:
            values=[]
            for seed in SEEDS:
                pred=pl.read_parquet(ROOT/f'runs/plan13_ablation/{seed}/{model}/3ep/predictions.parquet',columns=[column,target])
                rho=pearson(rankdata(pred[column].to_numpy()),rankdata(pred[target].to_numpy()))
                clock_records.append(dict(seed=seed,score=score,target=target,spearman=rho))
                values.append(rho)
            groups.append(f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}')
        lines.append('| '+LABELS[score]+' | '+' | '.join(groups)+' |')
    lines += ['',
        '预算参数由开场34字段决定，不读当前内容，已有路由/梯度测试已覆盖这个结构性质；固定同一开场参数和屏号时更换列表不会进入预算参数公式。它不意味着跨请求样本里gamma/S必须与参与度零相关，因为用户、场景、进度与内容会共同变化。', '',
        '右塔仍有首屏指示、前序反馈、累计hate、sum标签模长以及pace；不能因没有直接读req_idx就断言分数与进度无关。当前实验亦未证明S是无偏的存活权重或可直接用于逆概率校正。', '',
        '## 11. 逐seed核心差异：stay−Fusion', '',
        '| 预算 | 子集 | seed | stay长播ρ | Fusion长播ρ | 差值 |','| --- | --- | ---: | ---: | ---: | ---: |']
    paired=[]
    for budget in [3,1]:
        for scope in ['all','exit']:
            for seed in SEEDS:
                left=raw[(budget,seed,scope,high,'curr_longview_rate','spearman')]['value']
                right=raw[(budget,seed,scope,fusion,'curr_longview_rate','spearman')]['value']
                lines.append(f'| {budget}ep | {scope} | {seed} | {left:.6f} | {right:.6f} | {left-right:+.6f} |')
                paired.append(dict(budget=budget,scope=scope,seed=seed,stay=left,fusion=right,delta=left-right))
    lines += ['',
        '## 12. 结果解释与当前结论', '',
        '1. **没有得到计划预言的正向组件纯度证据。** 全集、退出屏、分层及单/混合等多个检查都没有支持“高stay代表高当屏参与度”。退出屏中右塔甚至比Fusion更负。',
        '2. **这不推翻原退出预测AUC。** 训练目标是gamma×stay预测下一屏继续，不是直接预测当屏长播/观看。当前数据里继续率与当屏参与度的关系也不是计划默认的简单正向关系；总分预测有效并不自动赋予每个乘因子另一项语义。',
        '3. **也不能把负结果直接当成已识别某种退出原因。** 离线屏重建、存活/退出选择、时长机会成本、用户差异、既有特征代理等都可能参与，但本轮没有对它们做因果干预。',
        '4. **当前不应直接将content_stay称为内容质量分、将S称为无偏权重。** 如需这些下游语义，应另行明确监督目标或设计可证伪的下游验证；本次不为得到正结论自动修改模型、加辅助loss或进入下游训练。', '',
        '## 13. 实现核查、数值修正与证据', '',
        '新增5项小样本测试通过，覆盖事件归属/视频顺序、原始与截断时长、重复事件拒绝、ties及常量组、分层与总体排序反转、按请求数加权User-AUC。', '',
        '另外用SciPy独立复算seed2026/3epoch的真实全集及退出屏Spearman，并逐个精确屏号独立复算相关后加权，与统计文件误差小于1e−12；核对同屏号内的屏号基线全部为无定义，没有留下常量伪相关。见final_checks.json。', '',
        '在分组相关初稿中发现：大常量组的Polars Pearson有时产生有限浮点伪相关，仅以is_finite过滤不够。已明确要求分数与目标组内max−min>0，并增加大常量组回归测试；全部本报告结果来自修正后重算的只读统计。旧初稿标记SUPERSEDED并留档，没有改动原模型或第13份预测。', '',
        '- [实验说明](实验说明.md) · [统计脚本](../../pipeline/backtest_components.py) · [报告生成](../../pipeline/report_component_backtest.py) · [新增测试](../../tests/test_component_backtest.py)。',
        '- [关联与来源manifest](../../runs/plan16_component_backtest_checked/manifest.json) · [完整运行日志](../../runs/plan16_component_backtest_checked/logs/run.log) · [测试日志](../../runs/plan16_component_backtest/logs/tests.log)。',
        '- [最终来源与独立数值核查](../../runs/plan16_component_backtest_checked/final_checks.json)。',
        '- [逐seed全部指标](../../runs/plan16_component_backtest_checked/metrics.parquet) · [三seed汇总](../../runs/plan16_component_backtest_checked/aggregates.parquet) · [当屏反馈](../../runs/plan16_component_backtest_checked/feedback.parquet) · [2026逐请求对齐表](../../runs/plan16_component_backtest_checked/aligned_2026_3ep.parquet)。',
        '- [组件与时钟相关](组件时钟相关.json) · [核心配对差异](核心配对差异.json)。', '',
        '复现仅计算统计，不训练或模型推理；输出目录必须是新目录，以免混入旧产物：', '',
        '```bash',
        'uv run python -m workspace.kr_time_budget.pipeline.backtest_components --output workspace/kr_time_budget/runs/plan16_component_backtest_fresh',
        'uv run python -m workspace.kr_time_budget.pipeline.report_component_backtest --run workspace/kr_time_budget/runs/plan16_component_backtest_fresh',
        '```', '',
        '报告生成默认读取核定的checked目录；--run显式选择统计输入。本报告的证据链接以当前输入目录生成。', '']
    # 切换--run时，全部产物链接同步指向该次统计，避免报告读新指标却链接旧文件。
    relative_run=Path(os.path.relpath(run,REPORT)).as_posix()
    lines=[line.replace('../../runs/plan16_component_backtest_checked',relative_run) for line in lines]
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'组件时钟相关.json').write_text(json.dumps(clock_records,ensure_ascii=False,indent=2))
    (REPORT/'核心配对差异.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2))
    (REPORT/'回测报告.md').write_text('\n'.join(lines))
    print(REPORT/'回测报告.md')


if __name__=='__main__':
    main()

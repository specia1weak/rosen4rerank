"""只读汇总定稿口径：复用plan13指标/审计，不训练、不推理、不重新bootstrap。"""
import hashlib
import json
from pathlib import Path
import statistics

WORKSPACE = Path(__file__).resolve().parents[1]
RUN = WORKSPACE/'runs/plan13_ablation'
REPORT = WORKSPACE/'回复/14定稿'
SEEDS = [2026, 2027, 2028]
NAMES = ['single_step', 'single_step_pace', 'mixture_step', 'mixture_step_pace', 'fusion', 'fusion_pace']
LABELS = dict(single_step='单威布尔', single_step_pace='单威布尔＋pace',
              mixture_step='K=2混合威布尔', mixture_step_pace='K=2混合威布尔＋pace',
              fusion='Fusion', fusion_pace='Fusion＋pace')
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(WORKSPACE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def metric(row, name):
    return row[name]['auc'] if name in {'user_gauc', 'intra_step_gauc'} else row[name]


def mean_std(values):
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values), values=values)


def display(values):
    value = mean_std(values)
    return f"{value['mean']:.6f} ± {value['sample_std']:.6f}"


def main():
    config = read(RUN/'config.json')
    summary = read(RUN/'summary.json')
    audit = read(RUN/'audit.json')
    data = read(WORKSPACE/'data/v3/manifest.json')
    diagnostic_path = WORKSPACE/'runs/diagnostics/plan14_final_mixture_step_pace_2026_3ep/summary.json'
    diagnostic = read(diagnostic_path)
    assert config['seeds'] == SEEDS and config['reported_epoch_budgets'] == [1, 3]
    assert set(config['experiments']) == set(NAMES) and not config['smoke']
    assert config['training_strategy'] == 'joint' and config['auxiliary_loss'] is None
    assert config['epochs'] == 3 and not config['frozen_budget'] and not config['detach_gamma']
    assert len(audit['models']) == 36 and len(audit['trajectories']) == 18
    assert len(audit['reproduction']) == 12 and diagnostic['status'] == 'PASS'
    assert hashlib.sha256((WORKSPACE/'data/v3/manifest.json').read_bytes()).hexdigest() == config['dataset_manifest_sha256']
    for filename, digest in config['source_hashes'].items():
        path = Path(filename)
        if not path.is_absolute():
            path = WORKSPACE.parents[1]/path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, filename
    columns = ['overall.auc', 'logloss', 'user_gauc', 'intra_step_gauc']
    aggregate, calibrations, diagnostics = {}, {}, {}
    for name in NAMES:
        for budget in [1, 3]:
            rows = []
            for seed in SEEDS:
                key = f'{seed}/{name}/{budget}ep'
                row = summary[key]
                path = RUN/key
                assert read(path/'metrics.json') == row
                assert all(row[field] for field in ['full_train', 'full_valid', 'full_test', 'checkpoint_reload_predictions_exact'])
                assert row['rows'] == 48898 and row['budget_optimizer_steps'] == budget*421
                assert row['trajectory_optimizer_steps_executed'] == 1263
                assert row['selected']['step'] <= budget*421
                assert row['user_gauc']['eligible_groups'] == 823 and row['user_gauc']['eligible_rows'] == 48855
                assert row['user_gauc']['total_groups'] == 855
                assert row['intra_step_gauc']['eligible_groups'] == 131 and row['intra_step_gauc']['eligible_rows'] == 46797
                assert audit['models'][key]['coverage_and_metrics_exact']
                assert audit['models'][key]['selected'] == row['selected']
                calibrations[key] = read(path/'calibration.json')
                diagnostics[key] = read(path/'diagnostics.json')
                rows.append(row)
            aggregate[f'{name}/{budget}ep'] = {field: mean_std([metric(row, field) for row in rows]) for field in columns}
            for field in ['overall.auc', 'logloss', 'intra_step_gauc']:
                existing = audit['aggregated'][f'{name}/{budget}ep'][field]
                assert abs(existing['mean']-aggregate[f'{name}/{budget}ep'][field]['mean']) < 1e-12

    def values(name, budget, field):
        return [metric(summary[f'{seed}/{name}/{budget}ep'], field) for seed in SEEDS]

    def avg(name, budget, field):
        return statistics.mean(values(name, budget, field))

    def table(budget):
        result = ['| 配置 | Pooled AUC ↑ | LogLoss ↓ | User-AUC ↑ | Intra-Step GAUC ↑ |',
                  '| --- | --- | --- | --- | --- |']
        for name in NAMES:
            result.append('| '+LABELS[name]+' | '+' | '.join(display(values(name, budget, col)) for col in columns)+' |')
        return result

    lines = ['# KR离开预测：定稿版本完整实验报告', '',
        '本报告只采用第13/14份确定的最终实验口径：KR 1K v3、单乘积BCE联合训练、三个随机种子、单/混合威布尔×有无pace，并保留Fusion及Fusion＋pace。以3epoch预算为主，1epoch为训练周期对照。User-AUC与其他指标一并完整报告。', '',
        '**全部结果来自已有18条连续训练轨迹及36组完整test评估。本次只读汇总JSON和现有审计，没有训练、重建数据、重跑模型推理或重新bootstrap。**', '',
        '## 1. 定稿结论', '',
        f"- 3epoch预算：混合＋pace的平均AUC为{avg('mixture_step_pace',3,'overall.auc'):.6f}、LogLoss为{avg('mixture_step_pace',3,'logloss'):.6f}；Fusion分别为{avg('fusion',3,'overall.auc'):.6f}、{avg('fusion',3,'logloss'):.6f}。两者数值接近，均值上Fusion略优；没有做等效性检验，不写成‘完全打平’或‘相同上限’。",
        '- 在两个训练预算、有无pace的四组配对中，混合相对单威布尔均为3/3 seed改善Pooled AUC与LogLoss。该结论针对这套实现与固定切分，并不覆盖所有指标。',
        f"- User-AUC不能省略：3epoch下，单威布尔＋pace为{avg('single_step_pace',3,'user_gauc'):.6f}，混合＋pace为{avg('mixture_step_pace',3,'user_gauc'):.6f}，Fusion为{avg('fusion',3,'user_gauc'):.6f}。单威布尔＋pace的用户内排序均值最好，混合＋pace在Pooled AUC、LogLoss及Intra-Step上更好。",
        '- pace同时改善单威布尔和混合威布尔；Fusion加pace的Pooled AUC在两个预算的三seed均值均略降。并非每个seed都下降：3epoch的seed2026有微小AUC改善，但LogLoss变差。只能说本次训练配置下观察到结构相关的特征收益，不能断言黑盒原则上无法使用pace。',
        '- 1epoch时混合＋pace在Pooled AUC、LogLoss、Intra-Step上均3/3 seed优于Fusion；延长训练后Fusion追上并在均值上略超，说明胜负需要注明训练预算。',
        '- 保留单威布尔作为完整对照，混合＋pace作为双塔主候选；不再为定稿结果追加调参。预测表现支持继续研究，不直接证明潜在预算/内容原因已被识别。', '',
        '## 2. 3epoch预算主结果', '',
        '所有表格均为完整test，各seed独立用完整valid LogLoss选checkpoint。±为三个seed间的样本标准差，不是置信区间；各列均值是逐seed指标的平均，没有集成三模型的预测。', '']+table(3)+['',
        'User-AUC在这里专指已有实现的**按有效用户请求数加权的User-GAUC**。它不是不加权用户宏平均，也不是session内AUC。字段名为`user_gauc.auc`，下文给出公式和覆盖。', '',
        '## 3. 任务、数据与处理边界', '',
        '预测对象是在当前重建屏结束时，用户是否还有下一屏：`y_continue=1`表示同session仍有后续完整屏，末屏为0。模型输出继续概率，退出概率为1−prediction。本报告全部AUC/LogLoss以继续标签及继续概率计算，不混用退出标签。', '',
        '| split | 时间范围 | Request数 | 用户数 | 继续率 |', '| --- | --- | ---: | ---: | ---: |']
    for split, dates in [('train','20220408–20220506'),('valid','20220507整场'),('test','20220508整场')]:
        item=data['splits'][split]
        lines.append(f"| {split} | {dates} | {item['rows']:,} | {item['users']} | {item['positive_rate']:.6f} |")
    lines += ['',
        '原始数据为KuaiRand-1K标准/随机曝光日志，按全部19个原始字段去重：11,756,073条降为11,661,848条。按用户时间排序；当前事件开始与上个事件开始＋截断播放时长的间隔超过30分钟则新开session，播放时长截断至0–300秒。排除每个用户观察窗首尾session、少于6条曝光的session，并按训练范围session数选取前5个tab。', '',
        '每6条曝光重建一屏，余数放会话头部为chunk0，只参与历史、不产生监督样本。屏边界依赖离线session分块，不应直接等同线上原生request边界。split不拆完整session；训练场次结束不晚于5月6日，valid/test分别完整落在5月7/8日，跨边界场次排除。', '',
        '用户历史先验由更早日期已完成session构建，物品先验由更早日期曝光构建；valid/test先验冻结至5月6日。Schema仅train-fit。当天已刷场次/耗时为开场前统计；前序反馈只取当前屏之前。当前屏视频属性可用，当前屏真实点击/长看等没有直接进入其自身历史反馈字段。', '',
        '已编码train固定物理shuffle，每epoch重复同一顺序；valid/test不shuffle。读取时仅将原始`elapsed_min`别名为`elapsed_min_raw`，不重新fit或写parquet。manifest里保留旧版`content_elapsed`与`delta1min`文字；当前路由和生效时钟以模型/配置为准，最终双塔只启用每屏一步的屏数预算。', '',
        '## 4. 特征集与Omni双塔', '',
        '| 类别 | 字段及数量 | 预算侧 | 内容侧 |', '| --- | --- | --- | --- |',
        '| 用户画像26 | user_id；user_active_degree、is_live_streamer、is_video_author、friend_user_num_range、register_days_range、follow_user_num_range、fans_user_num_range；onehot_feat0–17 | 参数网 | 输入 |',
        '| 开场上下文4 | tab、day_of_week、is_weekend、start_hour | 参数网 | 输入 |',
        '| 预算先验4 | user_hist_session_min、user_hist_req_cnt、day_prev_sess_cnt、day_prev_elapsed_min | 参数网 | 排除 |',
        '| 时钟2 | elapsed_min、req_idx及其raw列 | 最终仅req_idx_raw−1进入屏数生存公式 | 原字段排除；pace配置允许派生量 |',
        '| 内容历史9 | is_first_req；prev1_click_cnt/longview_cnt/pos_cnt；sess_cum_click_rate/longview_rate/pos_rate/hate_cnt；hist_tags | 排除 | 输入 |',
        '| 当屏列表9 | slate_tags/video_types/music_types；slate_prior_ctr_mean/longview_mean；slate_dur_mean_min；slate_unique_tag_cnt/author_cnt；slate_has_rand | 排除 | 输入 |', '',
        '54原字段＝30共享背景＋6预算/时钟专属＋18内容专属。预算参数网34字段、编码后252维；内容网48字段、316维；Fusion54字段、322维。**内容塔有完整用户画像与开场上下文**。两座塔各自用独立OmniEmbLayer及EmbView，背景字段相同、Embedding参数独立。', '',
        'user_id为16维；用户属性和开场上下文为8维；首屏/随机标志4维。hist/slate tags为16维，video/music types为8维；最大长度分别15/12/6/6，按原顺序截断、保留重复、sum聚合、padding为0，历史与当屏词表/Embedding独立。', '',
        '4个预算先验在上游做log1p，其余数值沿用v3透传；MinMaxDenseSetting固定[0,1]并不实际缩放原始数值。此次没有统一标准化、改mean聚合或重填首屏缺失。首屏指示、零历史、累计hate、sum模长仍可能携带进度代理，所以不能把字段路由称为“完全消除时钟信息”。', '',
        '带pace时在内容/Fusion向量末尾追加1维：`log1p(max(elapsed_min_raw,0)/max(req_idx_raw,1))`。内容输入成为317维，Fusion成为323维；不是新增原始信息源。它用当前屏结束时的累计分钟除当前屏数，原始量纲为分钟/屏，不是上一屏独立观看时长，也不是自动与进度统计独立的无量纲量。', '',
        '同一54原字段数据源不等于每个配置都有效使用54字段：无pace的仅屏数双塔完全不使用分钟信号；带pace时分钟通过派生量进入内容塔，Fusion则直接读取原始时钟字段。这里控制原始数据来源并显式比较不同路由，不能把所有配置表述成不受限制的完全相同输入通路。', '',
        '## 5. 模型与训练目标', '',
        '| 配置 | 预算 | 内容/Fusion输入 | 最终输出 |', '| --- | --- | --- | --- |',
        '| single_step | 单威布尔，仅屏数 | 48字段 | gamma_step × stay |',
        '| single_step_pace | 单威布尔，仅屏数 | 48字段＋pace | gamma_step × stay |',
        '| mixture_step | K=2混合威布尔，仅屏数 | 48字段 | gamma_step × stay |',
        '| mixture_step_pace | K=2混合威布尔，仅屏数 | 48字段＋pace | gamma_step × stay |',
        '| fusion / fusion_pace | 无预算乘积约束 | 54字段 / 54字段＋pace | sigmoid(MLP) |', '',
        '预算网络为252→64→输出head，SiLU；内容为316(或317)→64→32→1，Fusion为322(或323)→64→32→1，隐藏层SiLU、末端sigmoid。原内容末层bias=3，其他参数沿原构造初始化。', '',
        '单威布尔：令r=req_idx_raw−1≥0，开场34字段决定λ、z，`S(r)=exp(−(r/λ)^z)`，`gamma(r)=S(r+1)/S(r)`。尺度`λ=exp(clamp(raw_scale,−2,4.5)+1.6)`，形状`z=min(0.3+softplus(raw_shape),5)`。λ单位为屏；分钟分支关闭、gamma_time固定1。', '',
        '混合威布尔：`S_mix(r)=Σ_k π_k S_k(r)`，`gamma_mix=S_mix(r+1)/S_mix(r)`；存活后验`w_k(r)=π_k S_k(r)/S_mix(r)`，故`gamma_mix=Σ_k w_k(r) gamma_k(r)`。先验π由开场参数网决定，后验随存活更新，不能以固定π直接平均条件gamma。实现以float64及logsumexp计算风险增量/混合，最后按模型dtype输出概率。', '',
        '两个分量初始共享单分量head对应行的参数，但raw尺度bias分别偏移±log2、初始先验等权。共有模块先构造，新增head/pace后处理，避免消费RNG改变共有参数。pace新增第一层列零初始化，共有列复制。混合head从4变12维：保留双时钟兼容布局，单时钟有效行为2参数，混合有效行为6参数；未启用的分钟行及对照head不应误认为有效分支。混合改变了参数量与初始化，结果不是对参数数量已完全排除的纯分布族因果证明。', '',
        '两塔唯一损失：`BCE(clamp(gamma_step × stay, 1e−6, 1−1e−6), y_continue)`。从第一步同时反传预算和内容；不先分别训练两模型再相乘，没有额外CTR式辅助loss、detach、冻结、预热、scheduler或EMA。', '',
        '累计S应不增；允许z<1，因此gamma可以上升，乘上动态stay后的最终prediction也不保证单调。单次诊断中S上升为0/45,617，而gamma上升41,094/45,617，这是当前函数族允许的形状，不是单调gamma已成立。', '',
        '## 6. 训练、选择与指标口径', '',
        'seed=2026/2027/2028。每条轨迹完整train连续3epoch，每epoch421次更新，总1263步；batch4096，无梯度累积，CPU4线程。一个AdamW：预算LR=.005，内容/Fusion LR=.001；普通矩阵weight_decay=1e−6，Embedding/bias/一维参数零衰减；梯度裁剪范数10。', '',
        '完整valid节点为0/106/211/421/527/632/842/948/1053/1263。分别在≤421步和≤1263步内按完整valid LogLoss选best。每epoch不回滚best、不重置优化器；训练整条轨迹结束后才读出两个预算的完整test评估。1epoch结果是同一轨迹前缀，不能与3epoch结果当独立重训；延长预算同时增加更新和valid选择机会。', '',
        '- **Pooled AUC**：所有test请求间比较继续/退出样本的排序。',
        '- **LogLoss**：所有test请求的二元概率损失，越低越好；这是checkpoint选择指标。',
        '- **User-AUC（User-GAUC）**：先在每个用户内算AUC，排除只有一种标签的用户，再按其请求数加权：`Σ_u n_u AUC_u / Σ_u n_u`。同一用户的多场session可相互比较，不是每场会话内单独比较；ties用平均rank。有效823/855用户，48,855/48,898请求；排除32用户、43请求。',
        '- **Intra-Step GAUC**：按精确req_idx_raw分组，同样剔除单标签组、按请求数加权。有效131/681屏号组，46,797/48,898请求；排除550组、2,101请求。它在相同屏号下跨用户/场景比较，不自动等于纯内容质量。',
        '- Brier、固定20等宽桶ECE、均值偏差与第2–10屏指标作为校准诊断。均值偏差接近0不等于每个请求都已校准。', '',
        'User-AUC虽约0.49–0.52也需如实呈现，不能因为Pooled指标较好而省略。两者比较样本对不同，不是互相替代的指标；当前训练/选择目标没有直接优化User-AUC，因此不据数值差断定用户内排序问题已经解决。', '',
        '## 7. 1epoch训练周期对照', '']+table(1)+['',
        '1epoch时混合＋pace相对Fusion，平均AUC +0.001756、LogLoss −0.000485、User-AUC +0.015197、Intra-Step +0.002494。User-AUC提升是对Fusion的配对差，不表示高于单威布尔＋pace。', '',
        '## 8. 同预算配对消融：包含User-AUC', '',
        '表内均为左减右的三seed平均差；LogLoss负数更好，其余正数更好。末列只统计AUC与LogLoss同时改善的seed数，不冒充四项指标同时胜出。', '',
        '| 预算 | 左−右 | ΔAUC | ΔLogLoss | ΔUser-AUC | ΔIntra-Step | AUC/LL同时改善 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    pairs=[('mixture_step','single_step'),('mixture_step_pace','single_step_pace'),
           ('single_step_pace','single_step'),('mixture_step_pace','mixture_step'),
           ('fusion_pace','fusion'),('mixture_step_pace','fusion')]
    comparison_rows=[]
    for budget in [1,3]:
        for left,right in pairs:
            deltas={field:[a-b for a,b in zip(values(left,budget,field),values(right,budget,field))] for field in columns}
            wins=sum(a>0 and ll<0 for a,ll in zip(deltas['overall.auc'],deltas['logloss']))
            lines.append(f"| {budget}ep | {LABELS[left]} − {LABELS[right]} | "+' | '.join(f'{statistics.mean(deltas[field]):+.6f}' for field in columns)+f' | {wins}/3 |')
            comparison_rows.append(dict(budget=budget,left=left,right=right,deltas=deltas))
    lines += ['',
        '混合相对单分量的Pooled AUC与LogLoss更好，但四组比较的User-AUC均值都下降；尤其不能写成“混合分布全面优于单分布”。pace在双塔下提高四项均值，Fusion加pace的变化很小，且Pooled AUC方向为负；不能把小幅负变化解释成黑盒缺少理论表达能力。', '',
        '### 延长训练预算的变化：3epoch−1epoch', '',
        '| 配置 | ΔAUC | ΔLogLoss | ΔUser-AUC | ΔIntra-Step |', '| --- | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        lines.append('| '+LABELS[name]+' | '+' | '.join(f'{avg(name,3,col)-avg(name,1,col):+.6f}' for col in columns)+' |')
    lines += ['',
        'Fusion从1到3epoch的平均AUC增幅约+0.002334，混合＋pace约+0.000355。1epoch优势可以作为该协议下的较快获益证据，但没有匹配训练wall-clock成本，也没有证明3epoch已充分收敛或所有模型具有同一极限。', '',
        '### 已有LogLoss配对区间：混合＋pace−Fusion', '',
        '直接复用第13份按用户聚类、2000次bootstrap的95%区间；区间条件于各训练seed。没有为本报告重新采样，也没有将User-AUC附会成已经做过统计检验。', '',
        '| 预算 | seed | ΔAUC | ΔLogLoss | ΔUser-AUC | LogLoss差95%区间 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for budget in [1,3]:
        for row in audit['comparisons'][f'{budget}ep/mixture_step_pace - fusion']:
            seed=row['seed']; low,high=row['user_cluster_bootstrap_95']
            du=metric(summary[f'{seed}/mixture_step_pace/{budget}ep'],'user_gauc')-metric(summary[f'{seed}/fusion/{budget}ep'],'user_gauc')
            lines.append(f"| {budget}ep | {seed} | {row['auc_delta']:+.6f} | {row['logloss_delta']:+.6f} | {du:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines += ['',
        '3epoch三个seed的该LogLoss差区间均覆盖0；这不证明等效，只表示该检验下没有排除0。1epoch三seed方向一致也不表示全部区间都排除0。多次查看同一个test后，本报告属于固定切分上的探索性复验。', '',
        '## 9. 3epoch校准与第2–10屏', '',
        '第2–10屏共17,498条，占test约35.78%。偏差定义为平均预测继续率−真实继续率，正值为高估。', '',
        '| 配置 | Brier ↓ | ECE20 ↓ | 全集偏差 | 第2–10屏偏差 | 第2–10屏LogLoss ↓ |',
        '| --- | --- | --- | --- | --- | --- |']
    for name in NAMES:
        cals=[calibrations[f'{seed}/{name}/3ep'] for seed in SEEDS]
        rows=[summary[f'{seed}/{name}/3ep'] for seed in SEEDS]
        vals=[[c[field] for c in cals] for field in ['brier','ece20_fixed_width','mean_prediction_minus_observed']]
        vals += [[row['steps2to10'][field] for row in rows] for field in ['bias','logloss']]
        lines.append('| '+LABELS[name]+' | '+' | '.join(display(v) for v in vals)+' |')
    lines += ['',
        '这些指标回答概率评分与窗口校准，不与User-AUC或Intra-Step排序混为一谈。校准结果有跨seed差异，不能因某个窗口偏差更小就判定整个模型全面更优。', '',
        '## 10. 3epoch分场景结果', '',
        '按原始tab_raw汇总三seed，User-AUC仍按该场景内有效用户的请求数加权。场景用户可重叠，不把分域AUC简单平均当全局AUC；小场景的波动和单标签用户排除比例应一并看。', '',
        '| tab | 请求数 | 有效User-AUC请求/用户数 | 配置 | AUC均值 | LogLoss均值 | User-AUC均值 |',
        '| ---: | ---: | --- | --- | ---: | ---: | ---: |']
    for domain in range(5):
        key=f'({domain},)'
        for name in NAMES:
            rows=[summary[f'{seed}/{name}/3ep']['domains'][key] for seed in SEEDS]
            row=rows[0]; coverage=row['gauc']
            lines.append(f"| {domain} | {row['rows']:,} | {coverage['eligible_rows']:,}/{coverage['eligible_groups']} | {LABELS[name]} | {statistics.mean(r['auc'] for r in rows):.6f} | {statistics.mean(r['logloss'] for r in rows):.6f} | {statistics.mean(r['gauc']['auc'] for r in rows):.6f} |")
    lines += ['',
        '分域表均为请求加权定义下的独立指标，不能把某一域的改善概括为所有域一致改善。详细每seed、各域覆盖及指标保存在[原始summary](../../runs/plan13_ablation/summary.json)。', '',
        '## 11. 验证集与checkpoint选择', '',
        '以下“选中valid”与“epoch3末valid”分开报告，避免用训练末状态的均值冒充实际选中checkpoint。', '',
        '| 配置 | 3ep预算内选中valid LogLoss | epoch3末valid LogLoss | 各seed选中步2026/2027/2028 |',
        '| --- | --- | --- | --- |']
    progress={}
    for name in NAMES:
        selected=[summary[f'{seed}/{name}/3ep']['selected'] for seed in SEEDS]
        terminal=[]
        for seed in SEEDS:
            path=RUN/f'{seed}/{name}/joint/progress.jsonl'
            SOURCES[str(path.relative_to(WORKSPACE))]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            assert rows[-1]['step']==1263
            assert min(rows,key=lambda row:row['valid']['overall']['logloss']) == summary[f'{seed}/{name}/3ep']['selected']
            progress[f'{seed}/{name}']=rows
            terminal.append(rows[-1]['valid']['overall']['logloss'])
        lines.append('| '+LABELS[name]+' | '+display([s['valid']['overall']['logloss'] for s in selected])+' | '+display(terminal)+' | '+' / '.join(str(s['step']) for s in selected)+' |')
    lines += ['',
        'valid与test是相邻但不同日期。混合＋pace在valid更好而test均值略落后的事实，可以提示泛化差异，但不能据此确定是预算LR过大、某seed过拟合或某一种机制导致；这些解释没有独立干预验证。', '',
        '## 12. 单/混合威布尔的参数诊断：3epoch', '',
        '下表均为完整test请求上的参数均值，同一session在多屏出现会被重复计入，不是每session等权的总体预算均值；λ也不等于威布尔期望寿命。混合序号来自初始化，不预设永久的短/长人群身份。', '',
        '| 配置 | seed | 分量 | λ均值(屏) | shape均值 | 先验均值 | 存活后验均值 | scale上界比例 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES[:4]:
        for seed in SEEDS:
            clock=diagnostics[f'{seed}/{name}/3ep']['clocks']['step']
            if 'components' not in clock:
                lines.append(f"| {LABELS[name]} | {seed} | 单分量 | {clock['scale']['mean']:.3f} | {clock['shape']['mean']:.4f} | 1 | 1 | 未单列 |")
            else:
                for index,component in enumerate(clock['components']):
                    lines.append(f"| {LABELS[name]} | {seed} | {index} | {component['scale']['mean']:.3f} | {component['shape']['mean']:.4f} | {component['prior']['mean']:.4f} | {component['posterior']['mean']:.4f} | {component['scale_upper_clamped_fraction']:.2%} |")
    lines += ['',
        'shape的均值不代表每个请求的shape都处在同一侧，两个分量非零也不证明真实预算有两个群体。只监督gamma×stay仍可能存在不同分解给出相近乘积的不可识别性；当前代码检查可以验证路由与公式，不能验证“预算原因/内容原因”已被因果识别。', '',
        '## 13. 逐seed结果：所有最终配置与两个预算', '',
        '| 预算 | seed | 配置 | AUC | LogLoss | User-AUC | Intra-Step | 选中步 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for budget in [3,1]:
        for seed in SEEDS:
            for name in NAMES:
                row=summary[f'{seed}/{name}/{budget}ep']
                lines.append(f'| {budget}ep | {seed} | '+LABELS[name]+' | '+' | '.join(f'{metric(row,col):.6f}' for col in columns)+f" | {row['selected']['step']} |")
    lines += ['',
        '## 14. 代码与已有正确性证据', '',
        '当前阅读顺序：[字段/schema](../../pipeline/kr_request_schema.py) → [模型](../../pipeline/exit_model.py) → [训练链路](../../pipeline/exit_training.py) → [单次运行入口](../../pipeline/run_exit_experiment.py)。历史矩阵仍用[原plan13入口](../../pipeline/run_plan13_ablation.py)，没有将整理后的代码冒充生成历史结果的原源码。', '',
        '- 既有第13份审计覆盖18轨迹/36评估：完整数据覆盖、标签/用户对齐、共有初始化、首步两塔真实更新、乘积公式、预算选择范围、421/1263优化器步数、checkpoint独立重载一致；1epoch的12组既有配置预测逐值复现第12份，不重复当额外独立证据。',
        '- 第14份整理后已有76项测试通过；六配置的初始化/前向/梯度/一次AdamW更新与旧实现逐值一致，另有独立NumPy生存比及有限差分gradcheck。',
        '- 一个配置的真实8192条train连续3轮短程重放：6次更新的loss、参数哈希、AdamW矩状态及checkpoint选择逐值一致。它是诊断，不是新增效果实验。',
        '- seed2026/mixture_step_pace的3epoch预算checkpoint重载后，完整57,872条valid指标及48,898条test全部预测列复现；未重跑全部18条完整训练。新单次入口尚未全量重训，不能把测试范围写大。', '',
        '详见[单次诊断报告](单次运行诊断.md)。本次报告仅复用这些已完成的证据，没有重新执行测试/训练/推理。', '',
        '## 15. 定稿使用边界与产物', '',
        '可以报告：固定KR v3切分下的多seed预测效果、单/混合与pace消融、训练预算敏感性、User-AUC与Pooled指标的取舍，以及代码所实现的生存乘积结构。不能报告：所有指标全面获胜、3epoch已达模型上限、统计等效、真正的潜在预算已识别、pace天然正交、线上收益或下游反事实价值已得到验证。', '',
        '最终保留所有六配置与两个预算，不按test重新挑seed或改checkpoint。混合＋pace作为双塔主候选并保留单威布尔＋pace的User-AUC优势说明；后续若接下游，gamma/S的使用仍需单独验证，本报告不预先认定其业务语义正确。', '',
        '- [全部定稿原始指标](../../runs/plan13_ablation/summary.json) · [训练配置](../../runs/plan13_ablation/config.json) · [完整训练日志](../../runs/plan13_ablation/logs/train.log) · [已有审计与bootstrap](../../runs/plan13_ablation/audit.json)。',
        '- [本报告数据与来源哈希](综合报告数据.json) · [只读报告生成脚本](../../pipeline/report_final_experiment.py) · [原第13份报告](../13里程碑/实验报告.md)。', '',
        '仅重新生成报告（不会触发训练、推理或bootstrap）：', '',
        '```bash', 'uv run python -m workspace.kr_time_budget.pipeline.report_final_experiment', '```', '']
    REPORT.mkdir(parents=True,exist_ok=True)
    output=dict(scope='final plan13 only; 3epoch primary, 1epoch training-budget comparison',
        generation='read-only JSON aggregation; no training, inference, preprocessing, tests or bootstrap rerun',
        seeds=SEEDS,aggregates=aggregate,paired_deltas=comparison_rows,original_metrics=summary,
        original_audit=str((RUN/'audit.json').relative_to(WORKSPACE)),sources_sha256=SOURCES,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REPORT/'综合报告数据.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    destination=REPORT/'定稿实验报告.md'
    destination.write_text('\n'.join(lines))
    print(destination)
    print(f'36 existing evaluations; {len(SOURCES)} read-only JSON/log sources; User-AUC included')


if __name__=='__main__':
    main()

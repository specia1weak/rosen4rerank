"""Write a concise human handoff from the audited results."""
import json
from pathlib import Path


def main():
    runs = Path('workspace/kr_time_budget/runs/exit_v1')
    a = json.loads((runs / 'audit.json').read_text())
    source = json.loads(Path('workspace/kr_time_budget/data/v1/manifest.json').read_text())
    summary = a['summary']
    assert all(m['complete_folds'] == 5 and m['full_train_full_test'] and m['all_requests_predicted_once'] for m in summary.values())
    names = {'content':'Content-only', 'budget':'Budget-only Survival', 'fusion':'Black-box Fusion', 'ours':'Survival × Content'}
    table = ['| 模型 | User-GAUC ↑ | Session-GAUC ↑ | Pooled-AUC ↑ | LogLoss ↓ |', '|---|---:|---:|---:|---:|']
    for k, m in summary.items():
        table.append(f"| {names[k]} | {m['user_gauc']['auc']:.6f} | {m['session_gauc']['auc']:.6f} | {m['overall.auc']:.6f} | {m['logloss']:.6f} |")
    t = a['timing_audit']
    s = a['survival_audit']
    report = f'''# 2026-11-16 KR 时间预算：Argon 6.1 执行与审计

已完成原始数据读取、request parquet、5 折 train-fit schema、4 模型 full train/full test，以及结果审计。结论等级：**不确定；当前参数化模型未显示整体预测收益。机制与线上有效性结论为 `INVALID_FOR_CAUSAL_COMPARISON`。**

## 指标

以下均为五折合并 OOF，每个模型覆盖 {source['request_rows']:,} 条 request，每条只预测一次。GAUC 按组内样本数加权，单类组排除。模型固定最后第 8 epoch 测试，无测试选 checkpoint。

{chr(10).join(table)}

相乘模型的 Session-GAUC 改善，但没有转化为 User-GAUC、整体 AUC 或 LogLoss 的同步改善。User-GAUC 是方案指定的首要指标，不能只挑 Session-GAUC 宣称有效。

更关键的是，**Session-GAUC 在该标签构造下存在结构性退化**：每场只有末屏 y=0，前面均为1，因此无需训练的 `-req_idx` 在所有 eligible 会话上 Session-GAUC={a['clock_rank_diagnostics']['-1*req_idx']['session_gauc']['auc']:.6f}。它只凭“越往后得分越低”就完美排序，不能单独证明时间预算机制。相反，`+req_idx` 的 User-GAUC={a['clock_rank_diagnostics']['+1*req_idx']['user_gauc']['auc']:.6f}；这也说明不同分组指标回答不同问题。两个方向都报告，属于事后解析诊断，没有据此调参或训练新模型。

训练折正例率常数预测的 OOF LogLoss 为 {a['train_prevalence_constant']['logloss']:.6f}。各折内常数 AUC=0.5；五折不同常数合并后 pooled AUC 可以偏离 0.5。内容/Fusion 的训练损失很低、测试 LogLoss 很差，是明显过拟合现象。尚无删除 item-ID、加强正则或调参对照，不能断言具体是哪类特征导致。

User-GAUC eligible 用户 {summary['ours']['user_gauc']['eligible_groups']:,}/{summary['ours']['user_gauc']['total_groups']:,}；Session-GAUC eligible 会话 {summary['ours']['session_gauc']['eligible_groups']:,}/{summary['ours']['session_gauc']['total_groups']:,}，eligible request {summary['ours']['session_gauc']['eligible_rows']:,}。仅一个完整 request 的会话无法形成会话内 AUC，会从该指标排除。

## 数据和训练口径

- 原始日志 {source['raw_rows']:,} 行，按被使用原始列去重后 {source['deduplicated_rows']:,} 行；1000 用户。未用全月视频统计表。
- 原始重建场 {source['all_sessions']:,}，排除每用户首尾观测场和 <6 条交互场后，保留 {source['retained_sessions']:,} 场、{source['request_rows']:,} request。继续比例 {source['y_continue_mean']:.6%}。
- 切场：下一条 timestamp −（上一条 timestamp + 截断播放时长）>30分钟。M=6 从尾倒切，头部余数进入历史。
- 特征使用当前屏第一条 timestamp 相对场开始的时差；屏末 timestamp+末条播放量仅作诊断。w/R/G 是非负原始播放秒数的和。
- 五折按 user_id 分组；同用户的全部会话同折。每折约148.5万训练、37.1万测试，train/test 用户零交集。无单独 valid。
- 物品质量先验只取训练用户、严格更早日数据，平滑强度20；测试用户预算历史取训练总体先验。v1 的会话级预算统计额外保守滞后一个日，详见实验 README。
- 每折 rosen SchemaManager 只 fit train；train 固定物理 shuffle，test 固定顺序。rosen ParquetStreamDataset 自产 batch，DataLoader batch_size=None。
- 四模型每折共有模块逐 tensor 初始化完全一致，初始 state SHA256 一致。相同物理训练文件、数据顺序、batch4096、8epoch、每模型每折2904 optimizer steps；AdamW lr=1e-3，矩阵 decay=1e-6，Embedding/bias/一维参数不衰减；无 EMA、无 scheduler。
- Ours 前3epoch只训 gamma，后5epoch训练 gamma×content，budget_net lr=1e-4；共享编码器仍以1e-3更新。这是架构/目标的 screening，不是严格单变量因果实验。

## 必须保留的审计发现

1. **时间戳语义没有得到证明。** [KuaiRand 官方字段说明](https://github.com/chongminggao/KuaiRand/blob/main/README.md)仅称 time_ms 为交互时间戳。按 Argon 的“播放开始时间”解释，{t['overlapping_requests']:,} 个 request 的上一屏推算结束晚于本屏开始，占全部 request 的 {t['fraction_of_all_requests']:.2%}；中位重叠 {t['overlap_median_s']:.3f} 秒。该现象不能单独证明泄漏，但无法确认上一屏完整反馈在当前时刻已可用，不能称“已证明零泄漏”。
2. **倒切列表依赖未来会话长度。** 这六条连续日志不是已知真实同时曝光请求；即使当屏反馈列不进 X，request 位置本身也是离线代理构造。不能从此推导线上重排收益。
3. **用户/视频静态表没有历史快照可用时间。** 用户画像是否事前已知未经证明；视频 tag 用原始字符串签名，不拆多标签集合。
4. **相乘实现正确不代表预算机制被识别。** OOF 乘积最大绝对误差 {s['product_max_abs_error']:.1f}；但参数随 X_t 变化，约 {s['survival_increases_within_session_fraction']:.2%} 的相邻步 log S 上升，不能视为一条固定会话累计生存曲线。这里 delta 是列表总片长，不是真实未来观看量。
5. **逆生存分数不是行为策略概率。** 仅诊断性导出 cap=10 的 heuristic 分数，触顶比例 {s['heuristic_kappa_cap_fraction']:.2%}、ESS {s['heuristic_kappa_effective_sample_size']:,.0f}/{s['rows']:,}。不能当作已识别 IPCW，不能直接替代 NCIS 的日志 propensity。

## 当前交付与后续

按 Argon 文档末尾“现阶段6.1”的边界，本轮完成数据和退出预实验。6.3 的 w/R/G 已准备；尚未训练 LTV 头或6.2 Generator，没有 NCIS、CAVE uAUC/bAUC 或 LTV MSE 结果。

下一步优先澄清 time_ms 和反馈可用性、真实 request 边界；在同一协议补 item-ID 屏蔽/正则对照，用训练内 validation 用户选 epoch。若要解释固定预算机制，需要把预算分布参数限制到场开始时可见的背景，并与当前全特征参数化模型独立比较。若继续 LTV/Generator，还需明确 CAVE 指标定义、目标策略和日志动作概率。

## 文件与命令

- 实验说明：`workspace/kr_time_budget/history/README-v1.md`
- 数据清单：`workspace/kr_time_budget/data/v1/manifest.json`
- 宽表：`workspace/kr_time_budget/data/v1/requests.parquet`
- 各折数据/schema：`workspace/kr_time_budget/data/v1/fold_0` 至 `fold_4`
- 原始构建日志：`workspace/kr_time_budget/data/v1/logs/build.log`
- 训练日志：`workspace/kr_time_budget/runs/exit_v1/run.log`
- 汇总指标：`workspace/kr_time_budget/runs/exit_v1/summary.json`
- 审计（文件/顺序/schema hash、初始参数hash、完整覆盖、时间重叠和生存分数）：`workspace/kr_time_budget/runs/exit_v1/audit.json`
- 逐样本 OOF 和 checkpoint：`workspace/kr_time_budget/runs/exit_v1/fold_*/{{content,budget,fusion,ours}}/`
- 诊断性分数：`workspace/kr_time_budget/runs/exit_v1/survival_diagnostics.parquet`
- 验证日志：`workspace/kr_time_budget/data/v1/logs/tests.log`

验证结果：10项通过，覆盖倒切/余数历史、当前反馈不进入当前特征、30分钟边界、先验时间隔离、生存比数值稳定、GAUC权重/并列/单类排除、共有初始化、特征掩码、乘积和预热梯度，以及 Session-GAUC 的退化对照。只有框架既有 Polars API 弃用提示。

```bash
uv run python -m workspace.kr_time_budget.pipeline.build_requests --help
uv run python -u -m workspace.kr_time_budget.pipeline.run_exit
uv run python -u -m workspace.kr_time_budget.pipeline.audit
uv run python -m workspace.kr_time_budget.pipeline.write_report
uv run python -m pytest -q workspace/kr_time_budget/tests tests/rosen/utils/test_optimize.py tests/rosen/emb/test_sequence_group_selection.py
```

数据构建不能覆盖已有版本；须使用新的 --output。中断训练默认跳过已完成模型，未完成模型从头重跑，不恢复 optimizer。仓库未发现 .git，本轮没有创建 commit。
'''
    out = Path('workspace/kr_time_budget/history/20261116_kr-time-budget-argon-report.md')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(out)

if __name__ == '__main__':
    main()

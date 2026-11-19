# 显式掩码与统一预编码候选裁判

## 问题

局部裁判通过构造候选 Interaction 并注入私有控制键覆盖掩码，子类读原始 mask 时可能绕开该协议。
通用裁判逐候选重复 embedding 和历史编码，PIER/MultiGE 又各有独立分块实现，行为及开销不一致。

## 实现

- 删除 RerankModel 的候选 Interaction 适配器和 SlateRerankModel 对私有控制键的特判。
- 所有 16 个注册模型共用 SlateRerankModel.logits / score_permutations 两个固定入口。
  子类只能实现 _encode_items、_encode_context、_score_encoded，覆写固定入口在定义类时报错。
- encode_slate 一次读取原 Interaction，返回 EncodedSlate(items,dense,mask,context)。
  items 是输入槽位的独立物品表示；context 是模型自有的固定历史编码，不包含候选控制状态。
- components/scoring.py 按 candidate_chunk_size 分块，gather 物品/先验/掩码，K<N 时右补零至 N。
  _score_encoded 接收 [B,chunk,N,...] 张量及显式 [B,chunk,N] mask，返回对应 logits。
  输出截取前 K 位，所有模型的裁判概率统一乘以选中槽位 mask，无效位置为零。
- 公开裁判入口在 embedding 前校验 long 类型、batch/候选/长度、重复下标、越界、模型长度限制及设备。

新评分函数接口是 score_candidate_logits(encoded,orders,scorer=...,chunk_size=...,candidate_dense=...)，
score_candidate_probs 复用同一 logits 路径。旧 OCPM 专属多参数入口直接移除，没有兼容层。

## 编码复用边界

独立物品编码与候选排列无关，可在 gather 前完成。PRM 的跨物品注意力、EXTR 的分区注意力、
KuaiShou 的当前列表上下文、OCPM 的当前页表示及目标历史注意力均依赖所选列表，仍在候选内计算。

新增 OCPMRerankModel(SlateRerankModel)，PIER 和 MultiGE 共同继承它，不再重写 score_permutations。
OCPMContext 保存历史 OAM 页表示 [B,P,D] 与页 mask；历史页在候选分块前编码一次，
不再沿候选轴复制原始 [B,P,N,F,D] 历史并反复执行 OAM。
ListwiseGenerator.forward 的 history_pages 参数现在直接接收这些页表示；全部调用已迁移。

KuaiShou 的 TargetContextEncoding 保存历史物品表示、K/V 及有效性；在分块前编码一次，
评分时以广播视图对接当前候选。固定历史不跨请求或调用缓存，保留当前参数下的计算图和梯度。
分块只限制候选中间态，最终返回 [B,M,K] 仍需对应输出内存。

## 长度与行为边界

统一执行路径不意味着任意固定槽位模型天然支持动态列表。
DNN/DCN/PRM/KuaiShou 支持 1≤K≤N 的右补空槽裁判；
EXTR 的自然/广告分区、OCPM 的固定槽位池化仍要求 K=N，通过 requires_full_slate 在统一入口拒绝局部列表。
这次没有改变生成器输出长度、模型参数、buffer、模块注册顺序、初始化随机数或训练配置。

通用裁判此前对无效槽仍输出 sigmoid 值，现在统一归零；依赖该旧行为的外部调用方必须迁移。
自定义 _mask 仅在原请求编码阶段读取一次，候选评分使用已经选择/补齐的显式 mask，
不再要求每个子类了解 Interaction 私有开关。
自定义模型若此前覆写 logits/score_permutations，需迁移到明确的编码与评分扩展点。

## 验证

```bash
uv run pytest tests workspace/kr_time_budget/tests workspace/ml1m-rerank/tests
```

906 passed。[全量日志](../../../../logs/20261119_unified_judge_tests.log)。新增 58 项用例覆盖：

- 所有 16 个模型确实继承同一个裁判入口，分块大小 1/2/99 与独立构造的 schema 候选 logits 一致。
- 支持的 K=1/2/3、分块尾批、分数/掩码坐标、无标签/无 schema mask 的部分列表。
- 自定义源 mask 读取一次并显式传入，原 Interaction 不增加字段、不被改写。
- embedding 每次只执行一个当前视图及必要的一个历史视图；历史 OAM 或 KuaiShou 历史 K 只执行一次。
- PRM/KuaiShou/PIER/MultiGE metric 的分块梯度与独立候选求和一致。
- 非法排列在编码前报错，错误 scorer shape 被拒绝，子类不能建立第二条裁判执行路径。

改动前记录 16 个模型同 seed 的 state_dict、原序预测、选中结果及裁判输出，改动后在相同合成输入
（B=2,N=3,M=3，支持时 K=2）、CPU FP32 eval/no_grad 下核对：初始化 tensor 逐值相等，
原序分数/最终排列/选中分数逐值相等；全列表裁判最大绝对差 5.96e-8，部分列表输出一致。
该对照将原 mask 置为全有效，避免将本次无效槽归零修正混入数值保持判断。
[数值与初始化核对](../../../../logs/20261119_unified_judge_parity.log)。

合成局部裁判计时：CPU FP32 eval/no_grad、2线程、B=2/N=3/K=2/M=60、chunk=16，
预热后5次中位数，对比“逐候选构造 schema Interaction + 相同模型 logits”参考循环：

| 模型 | 分块路径 ms | 逐候选参考 ms |
| --- | ---: | ---: |
| DNN | 0.232 | 3.138 |
| PRM | 0.429 | 5.290 |
| KuaiShou | 0.725 | 10.093 |

这只验证移除重复编码和 Python 循环的局部收益，不代表数据集端到端延迟或所有设备性能。
[计时口径与结果](../../../../logs/20261119_unified_judge_timing.log)。未重训、未重跑完整数据集评估，历史 runs 未改写。

## 文档与交接

当前 API、架构、RERANK/EVALUATE、模型 README 已同步；四份 Repomix 包和单文件交接包
从当前源文件重新导出并逐文件核对。后续模型扩展以 [重排契约](../../details/RERANK.md) 为准。

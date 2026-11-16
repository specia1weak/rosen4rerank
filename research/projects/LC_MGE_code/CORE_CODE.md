# LC_MGE 核心代码导航

## 打包范围

主实现位于 `rerank_model_dir/MultiG_E_Submodularity_Model_lc/`。
核心包保留其训练、预测评估、数据解析和模型依赖，以及两个直接相关版本的模型与配置，方便理解实现差异。
本次整理通过文件筛选和导航完成，原有源码内容与目录结构保持不变。

| 目录 | 核心包中的内容 | 用途 |
| --- | --- | --- |
| `MultiG_E_Submodularity_Model_lc/` | `main.py`、`model.py`、`config.py`、`data_input.py`、`layers.py`、`tools.py`、`util.py`、`run.sh` | LC 版本主流程及其本地 Python 模块依赖 |
| `MultiG_E_Submodularity_Model/` | `model.py`、`config.py` | 含 Submodularity 生成器的相关版本，用于代码对照 |
| `MultiG_E_model/` | `model.py`、`config.py` | 多生成器与评估器的相关版本，用于代码对照 |

表中的目录均位于 `rerank_model_dir/` 下。两个对照版本仅打包模型与配置；需要运行它们时，应使用原目录中的完整代码。

## 建议阅读顺序

以下文件名均指 LC 主实现目录中的文件。

1. `config.py`：数据路径、训练模式、特征维度、候选排列数量和网络参数。
2. `data_input.py`：TFRecord 解析、特征与标签构造、排列生成及输入流水线。
3. `main.py`：Estimator 创建、训练、预测评估和导出入口；`calculate_result` 汇总预测结果。
4. `model.py`：模型主体。先看 `model_fn_estimator`，再阅读各个生成器与评估器。
5. `layers.py`、`util.py`、`tools.py`：网络层、张量工具、排列枚举和计时工具。

`model.py` 中的主要入口：

| 方法 | 作用 |
| --- | --- |
| `model_fn_estimator` | 串联特征处理、生成器、候选序列评估和训练/预测输出 |
| `_process_features`、`_create_weights` | 特征处理与权重初始化 |
| `_oam`、各类 attention 方法 | 页内属性与用户行为表示 |
| `_Pointwise_generator_model` | Pointwise 生成器 |
| `_Listwise_generator_model` | Listwise 生成器 |
| `_Generative_generator_model` | 自回归序列生成器 |
| `_Submodularity_generator_model` | Submodularity 生成器及 LC 相关计算 |
| `_OCPM`、`_OCPM_TOP_K` | 原始页面与候选序列的评估 |
| `_create_loss`、`_create_optimizer` | 损失组合与优化 |

## 未打包的内容

- DNN、DCN、PRM、EXTR、KuaiShou、PIER、PRS、NAR、seq2slate 等其他基线目录。
- 名称含 `metric` 的独立实验变体。
- 独立的 `evaluate.py`、`model_load.py`、空的 `restore.py` 和依赖外部模板的 `build_model.sh`；主流程的预测评估与导出代码仍保留在 `main.py` 中。
- `avg_std/` 辅助数据、缓存、字节码、日志、模型产物和 ZIP 归档。
- 旧的全量 Repomix 产物 `rerank_model_dir/rm.md`，避免重复嵌套打包。

## 重新生成单文件

在项目根目录执行：

```bash
repomix --config repomix.config.json
```

输出为根目录的 `repomix-output.md`，包含目录树、文件路径和以代码块展示的完整源文件内容。
包不压缩代码，保留注释和内部空行，文件首尾空白由 Repomix 规范化。
调整 `repomix.config.json` 中的 `include` 列表即可修改范围。

## 原代码运行入口

原 README 标注环境为 Python 3 + TensorFlow 1.5；源码使用 TensorFlow 1.x 风格接口，并依赖 NumPy、scikit-learn。本次打包不验证训练环境兼容性。
运行前需在 LC 目录的 `config.py` 中设置本机数据路径和 `TRAIN_MODE`：1 为训练，2 为预测评估，3 为训练后预测评估，4 为导出。

```bash
cd rerank_model_dir/MultiG_E_Submodularity_Model_lc
python main.py
```

也可使用原有 `bash run.sh` 后台运行，输出写入该目录的 `log`。
合并后的 Markdown 是代码阅读与分析材料；执行训练仍使用原始 Python 文件。

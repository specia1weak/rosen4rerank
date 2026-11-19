# Time Budget / Rosen

KR1K 继续预测的完整复现入口见 [experiments/kuairand/README.md](experiments/kuairand/README.md)。
从原始 CSV 重新生成数据并训练，不需要下载本人的 workspace：

```bash
uv sync --locked --group dev
POLARS_MAX_THREADS=4 uv run python -m experiments.kuairand.run_nxt_continue \
  --source_dir /path/to/KuaiRand-1K/data --experiment_name raw_weak_epoch1 --device cpu
```

从原始 CSV 实测 Test AUC **0.6978418304**（约 0.7）；完整 1 epoch / 421 steps。
实验代码和测试在 `experiments/kuairand/`；数据、报告与训练结果生成到本地 `workspace/kr_time_budget/`。
以下为迁入的 Rosen 框架说明。

面向多场景推荐实验的 PyTorch 框架，由 betterbole 核心独立迁出。

```bash
uv sync --group dev
uv run python -c "from rosen.emb import SchemaManager; from rosen.models.msr import build_model"
uv run pytest
```

其他项目中安装：`uv add --editable ../rosen`。

所有导入使用 `rosen`，例如 `from rosen.core.trainers import BaseTrainer`。
仓库包含核心框架、测试、文档和 `experiments/` 实验入口；原始数据与生成产物不纳入版本管理。

文档入口：[API](docs/rosen/API_REFERENCE.md)、[架构](docs/rosen/ARCHITECTURE.md)。
历史文档中的实验路径用于解释来源，对应实验文件未随包迁入。
历史 pickle/完整模型 checkpoint 可能记录旧包路径，使用前应转换；推荐保存 state_dict。

`rosen.experiment.ROOT_DIR` 默认是当前工作目录，可用 `ROSEN_ROOT_DIR` 指定实验根目录；
`WORKSPACE` 为其下的 workspace。安装到 site-packages 后不会将工作目录切换进安装目录。
可选的历史视觉 GAN 脚本需要 `uv sync --extra vision`，不属于推荐实验主链路。

## Git 管理约定

版本管理包含框架和实验源码、测试、`uv.lock`、`Argon-plans/` 计划，以及维护文档。
`.gitignore` 排除虚拟环境、缓存、IDE 配置、原始数据 `datasource/`、
整个 `workspace/`（含本地报告）、checkpoint 和 parquet。
这些文件保留在本地；报告中的数据、训练日志和 checkpoint 链接需要对应本地产物才能访问。
`references/rerank` 是指向本机外部目录的符号链接，也不提交；
仓库内的参考代码见 `research/projects/LC_MGE_code/`。

新机器使用 `uv sync --group dev` 安装依赖，并按
[实验运行说明](experiments/kuairand/README.md) 准备数据和运行实验。
提交前检查 `git status --short`、`git diff --cached --stat` 和
`git diff --cached --check`；使用 `git status --short --ignored` 查看被忽略的本地产物。

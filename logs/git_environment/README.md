# 本地 Git 整理记录（2026-11-17）

整理前仓库位于 `master`，尚无 commit、无远程地址、无 `.gitignore`；
原始数据和 KR 生成产物合计约 12 GB。

## 提交范围

- 提交框架与 KR 实验源码、测试、依赖锁文件、计划、报告及配图、维护文档和已有参考代码。
- 新增根目录 `.gitignore`，排除 Python 环境/缓存、构建产物、IDE/macOS 文件、
  环境变量文件、原始数据、KR 数据/训练结果/日志、checkpoint 和 parquet。
- 本机外部链接 `references/rerank` 不提交；本地数据和历史训练产物均保留。
- 根 README 说明版本管理边界，以及新机器恢复依赖和数据的入口。
- 仅在本仓库设置 `core.quotePath=false`，让中文路径直接显示；沿用现有分支和提交身份。

## 验证

- 默认 pytest 首次收集失败：根目录与 KR 实验目录均含名为 `tests` 的包。
  在 `pyproject.toml` 设置 `--import-mode=importlib` 后，
  `uv run --no-sync pytest -q`：**204 passed，362 warnings，4.70 秒**。
  完整输出保留在本地 [pytest.log](pytest.log)。
- 首次暂存 656 个文件，总计约 12.87 MiB，单文件均小于 5 MiB；
  本记录随后加入提交。没有暂存被忽略的文件。
- 检查候选文件名及常见私钥、AWS/GitHub/OpenAI 密钥特征，未发现匹配；
  此检查不代表完整安全审计。
- 已验证原始数据、训练产物、IDE 文件、虚拟环境、egg-info 和外部符号链接命中忽略规则。
- 首次导入的 `git diff --cached --check` 检出历史行尾空白 1813 处、
  文件末尾空行 47 处（其中根 README 的末尾空行已在本次编辑中去除）。
  历史源码、参考材料与 Markdown 格式保留，未做批量格式化。
  初始检查输出见本地 [diff-check.log](diff-check.log)。

本次为本地提交，不配置远程地址，也不执行 push。新的 clone 只包含受版本管理的内容；
报告引用的数据、原始训练日志、预测和 checkpoint 需要另行恢复本地产物。

# Rosen 独立包迁移

## 背景与决策
将 betterbole 核心迁入同级 ../rosen，包名及发行名改为 rosen，不包含 custom_models。
原仓库及历史实验保留。新项目采用 src/rosen，附框架文档、核心测试及 fixture。
依赖原 @examples 的数据准备测试未迁入。

## 适配与影响
内部导入统一改为 rosen，setuptools 仅发布 rosen 命名空间。
根据实际导入声明依赖，移除原项目 CUDA 专用锁定。
ROOT_DIR 默认调用方 cwd，可通过 ROSEN_ROOT_DIR 覆盖；任务队列锁写入 WORKSPACE。
调用者需更新导入；旧 pickle/完整模型 checkpoint 可能需要转换模块路径。
框架算法及数据结构保持原有实现。历史 GAN 脚本保留，不保证独立库导入。

## 验证
237 个源码 Python 文件 AST 检查通过，源码无 betterbole 导入。
源码编译、初次 wheel/sdist 构建通过；wheel 仅含 rosen 命名空间。
现有活动环境中的 rosen 与实验目录 smoke 通过。
现有环境缺 pandas、pyarrow、pytest，核心完整测试未运行。
用户要求不下载新环境后已停止 uv sync，并移除本次未完成的 .venv。

## 后续
使用已有且依赖齐全的环境运行 tests。说明见 ../rosen/README.md。

# 原始 KR1K 示例

附件原文收录当前仓库已有的 `docs/rosen/details/kuairand1k.py`，并附 `src/rosen/datasets/kuairand.py` 数据路径适配器。原始脚本未因本次整理而修改。

它展示单个 `is_click` 标签、多场景 `tab`、SchemaManager、ParquetStreamDataset、按 domain 注册 evaluator，以及模型工厂和两阶段 Trainer 用法。它不是 CTR + 长播的多任务示例，也不是六视频重排入口。

注意原文中的历史行为：脚本虽然生成 train/valid/test 三个文件，`ps_valid` 实际读取 `test_path` 作为 Trainer 的验证输入。因此它不作为本轮新示例的验证/测试协议参考；新的两个示例明确使用独立 valid 选 checkpoint，再加载测 test。

原脚本默认设备为 CUDA，并在模块顶层解析参数、设置工作目录；阅读或语法检查无需 import 执行。数据根由 `ROSEN_DATASOURCE_ROOT` 指定。更多框架依赖查配套 `rosen-core.md` 或完整仓库。

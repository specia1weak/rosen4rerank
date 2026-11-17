"""MovieLens-1M评分活动继续预测参考实现。

阅读顺序（模块均位于pipeline）：
1. prepare_sessions.py：时间戳为秒，空闲严格>30分钟开新Session，每评分一步。
2. prepare_training.py：严格过去历史、时间80/90分位切分、完整Session与停止观察窗。
3. build_current_information.py：保留切分/顺序，添加当前rating/5并使用当前累计pace。
4. current_schema.py -> current_model.py：独立Omni两塔、屏数威布尔、内容评分投影。
5. single_auxiliary.py：仅左NLL或仅右BCE；双辅助及stop-left见current_model.py。
6. train_current_epoch1.py -> train_single_auxiliary.py：三seed完整首轮训练与梯度审计。

schema.py/model.py/train.py是当前反馈版本的基础实现，保留以展示继承与历史初始化对齐。
数学和训练/评估工具复用experiments.kuairand.pipeline，不导入被Git忽略的workspace代码。
原始数据由ROSEN_DATASOURCE_ROOT下MovieLens/ml-1m提供，生成文件仍写workspace/ml1m。
从仓库根用uv run python -m experiments.movielens.pipeline.<模块>运行入口。
训练入口保留历史run的严格对齐检查，需依次准备数据和前置run；不是无前置条件的教程。
当前模型预测时已知当前评分，UID恒0；这是评分活动任务，不把分钟解释为观影时间预算。
tests包含微型表边界、真实Omni编码、联合梯度/stop-left和仅左/右辅助检查。
"""

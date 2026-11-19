"""KuaiRand请求级继续预测参考实现。

新机器直接运行：见 README.md，run_nxt_continue --source_dir 接收原始 CSV，完成预处理、训练和 Test。
nxt_continue_v1.py 是独立 raw -> dataset / 模型实现，不需要历史 workspace。

历史参考实现阅读顺序（下列模块位于pipeline）：
1. build_requests.py的sessionize：原始事件切场、按六视频构造请求。
2. build_requests_v3.py：前序反馈、历史先验、候选内容、时间切分和schema编码。
3. build_timestamp_data.py：将历史播放时长口径更正为纯事件时间戳。
4. build_previous_pace.py -> build_current_feedback.py：前序pace审计列和当刷反馈旁路。
5. kr_request_schema.py -> exit_model.py：独立Omni预算/内容编码，屏数单/双威布尔。
6. current_feedback*.py：当刷反馈、联合乘积BCE、辅助NLL/BCE及stop-left。
7. run_current_feedback*.py：数据加载、初始化对齐、训练与完整valid/test评估。

从仓库根用uv run python -m experiments.kuairand.pipeline.<模块>运行入口。
这些是历史实验的参考副本，保留数据版本和前置run对齐要求，不能跳过前置产物直接运行。
生成文件仍写workspace/kr_time_budget，不写入experiments；原始数据由rosen数据根目录提供。
当前反馈预测发生在当刷反馈已知后；双威布尔指同一屏数轴两个分量，不是双时钟。
模型与Loss在独立模块；tests检查时间戳、反馈边界、概率和梯度路由，不需要真实数据集。
"""

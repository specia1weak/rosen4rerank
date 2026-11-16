This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where line numbers have been added, content has been formatted for parsing in markdown style.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
- Pay special attention to the Repository Description. These contain important context and guidelines specific to this project.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Content has been formatted for parsing in markdown style

# User Provided Header
# KR 特征分类与数据处理流水线：独立审阅导航

这是通过 Repomix 生成的源码审阅快照。目的：请独立判断特征分类、特征时序、切分、编码和模型实际输入是否正确；不要默认既有设计或文档结论成立。请按“严重程度、文件及原始行号、具体触发条件、证据、修复建议”汇报问题，并区分已确认错误、建模取舍与待验证风险。

## 建议阅读顺序

1. `pipeline/kr_request_schema.py`：54字段、30共享+6预算时钟+18内容、34字段预算参数网与48字段内容塔；FeatureSource仅是编码元信息。
2. `pipeline/build_requests_v3.py`：当前数据构建主入口；原始日志去重、会话筛选、请求聚合、日级过去先验、切分、log1p、train-fit Schema及物理shuffle。
3. `pipeline/build_requests.py::sessionize`、`pipeline/features.py::past_daily`：v3实际复用的历史辅助函数；这两个文件其余旧版main/Schema/切分不是当前主线，不能据此推断当前协议。
4. `data/v3/manifest.json`、`data_audit.json`、`schema/feature_meta.json`：已落盘数据的规模、协议和fit后字段设置。feature_meta引用的完整parquet词表未放入本包。
5. Rosen的SchemaManager、Sparse/MultiSparse/MinMaxDense、OmniEmbLayer/EmbView、ParquetStreamDataset及TensorFormatter源码：检查变换发生在何处、padding/OOV/NULL、截断与聚合、tensor dtype和额外raw列。
6. `pipeline/run_exit_v7_joint.py::load_data`：当前v9训练复用的数据加载，尤其是elapsed_min_raw别名；`run_exit_v3.py::ExitTrainer/scores`与`run_exit.py`提供被复用的训练/评估工具。这些文件的旧main不是当前训练入口。
7. `pipeline/time_budget_v9.py`、`run_exit_v9_pooled.py`：当前Argon 10模型和训练入口；v8代码保留是为了查看被v9复用的优化器参数分组、统计函数及上一轮对照实现。
8. `debug_dataflow.py`和tests：辅助还原单个会话与理解已有验证边界。注意debug_dataflow的默认前向仍是旧v5，只把其前半段当作数据流审阅辅助，当前双塔路由以v9源码为准。

上述pipeline、data、tests均位于`workspace/kr_time_budget/`。本文件是代码/元数据集合，不是可以直接执行的单文件程序，也没有收齐所有历史模型及框架的传递依赖。

## 请重点核查

- raw去重、时间排序、30分钟切场、边界会话过滤、6条倒切及头部余数处理是否合理；倒切依赖会话最终长度，是否影响目标部署时的请求/屏号定义和信息可得性。
- 预测时点究竟是屏初还是屏末；elapsed_min使用屏末时间与最后条play_clip是否和推理时点一致，标签与当屏视频/互动是否存在时间错位。
- prev1、sess_cum、hist_tags是否严格排除当前及未来反馈；req_idx=0余数是否恰当地参与历史；首屏填零、累计hate、is_first_req及标签长度是否形成代理时钟。
- 用户先验和物品先验是否只使用可获得的历史：当前日排除、valid/test冻结到May6、会话跨日/重叠、当天已看场次和时长是否存在未来泄漏。
- 用户/视频静态元数据的获取时间假设、top-tab选择和train-fit词表的使用范围是否符合评估协议。
- 只有4个预算先验做上游log1p；MinMaxDense固定[0,1]实际为透传；其余数值范围与embedding量级是否合理。分钟和屏数是否始终分开，raw列是否在公式中正确使用。
- MultiSparse的重复标签、清洗、head截断、sum聚合、空序列、PAD/NULL/OOV行为，是否与内容含义及历史长度一致。
- 30个共享背景、4个预算先验、2个raw时钟、18个内容字段，是否在真实EmbView与公式里按约定消费；两塔表是否独立；是否有漏字段或额外信息混入。
- Parquet是否已经编码，Dataset是否再次transform，batch_size=None是否正确，train/valid/test顺序、raw用户/屏号与标签对齐是否正确。
- 现有assert和测试能够证明什么、不能证明什么；代码可运行、非零梯度和既有审计不等同于特征语义正确。

## 快照边界与已知文档时效

当前数据是v3；当前Argon 10入口是v9。v9可能仍在主线程迭代，本快照仅表示打包时读取到的源码，不能把其存在理解成全部正式实验已完成。
`特征分类.md`包含历史阶段描述（例如早期req_idx公式尚待定义），而v9已经实现双时钟；原data/v3 manifest中的content_elapsed也是建库时的旧模型说明。出现矛盾请明确指出，以实际构建代码、落盘Schema和当前模型路由分别核实，而不是把不同阶段混为一体。
完整原始CSV、训练parquet、完整词表和checkpoint不包含在包中；这里只包含完整选定源码与少量已存在的JSON元数据。未对源码压缩、去注释或摘要替代，输出保留原始行号。

`01_snapshot_manifest.json`记录本次选定每个源文件的SHA256；可用来确认所审阅版本。Repomix的文件树可定位全部文件。理论方案和效果承诺没有作为审阅约束，源码正确性应独立判断。

快照时间（UTC）：2026-11-16T08:44:47.223292+00:00


# Directory Structure
````
src/
  rosen/
    core/
      trainers/
        base.py
      enum_type.py
      interaction.py
    data/
      dataset.py
      padding.py
    datasets/
      base.py
      kuairand.py
    emb/
      schema/
        __init__.py
        base.py
        categorical.py
        numerical.py
        utils.py
      emblayer.py
      manager.py
    models/
      msr/
        base.py
      base.py
    utils/
      optimize.py
      shuffle.py
      sort.py
workspace/
  kr_time_budget/
    data/
      v3/
        schema/
          feature_meta.json
        data_audit.json
        manifest.json
    pipeline/
      audit_feature_categories.py
      build_requests_v3.py
      build_requests.py
      features.py
      kr_request_schema.py
      run_exit_v3.py
      run_exit_v7_joint.py
      run_exit_v8_branches.py
      run_exit_v9_pooled.py
      run_exit.py
      time_budget_v8.py
      time_budget_v9.py
    tests/
      test_feature_categories.py
      test_v3.py
      test_v8.py
      test_v9.py
    debug_dataflow.py
    feature_categories.json
    特征分类.md
01_snapshot_manifest.json
````

# Files

## File: src/rosen/core/trainers/base.py
````python
  1: from __future__ import annotations
  2: 
  3: from contextlib import nullcontext
  4: from pathlib import Path
  5: from typing import Callable, Optional, Union
  6: 
  7: import torch
  8: 
  9: from rosen.core.interaction import Interaction
 10: from rosen.core.train.context import (
 11:     EvaluationFeedback,
 12:     TrainContext,
 13:     TrainerComponents,
 14:     TrainerDataLoaders,
 15: )
 16: from rosen.core.train.hooks import (
 17:     CustomTrainStepProtocol,
 18:     TrainerStateProtocol,
 19: )
 20: from rosen.emb import SchemaManager
 21: from rosen.experiment.param import ConfigBase
 22: from rosen.models.base import BaseModel
 23: 
 24: 
 25: class BaseTrainer:
 26:     """Minimal trainer for ordinary train/eval/checkpoint flow.
 27: 
 28:     This class deliberately does not interpret model training-cycle methods or
 29:     health monitoring. Those fixed boundaries live in ``BaseTrainerWithLife``;
 30:     experiment audits live under :mod:`rosen.core.trainers.audit`.
 31:     """
 32: 
 33:     def __init__(
 34:             self,
 35:             model: BaseModel,
 36:             optimizer: torch.optim.Optimizer,
 37:             manager: SchemaManager,
 38:             loaders: TrainerDataLoaders,
 39:             components: TrainerComponents,
 40:             cfg: ConfigBase,
 41:     ):
 42:         if components.health_monitor is not None:
 43:             raise ValueError(
 44:                 "BaseTrainer does not run training-cycle health monitoring; "
 45:                 "use BaseTrainerWithLife."
 46:             )
 47:         self.model = model.to(cfg.device)
 48:         self.optimizer = optimizer
 49:         self.manager = manager
 50:         self.train_loader = loaders.train
 51:         self.valid_loader = loaders.valid
 52:         self.test_loader = loaders.test
 53: 
 54:         self.evaluator = components.evaluator_manager
 55:         self.timer = components.timer
 56:         self.recorder = components.recorder
 57:         self.early_stopper = components.early_stopper
 58:         self.cfg = cfg
 59: 
 60:         self.epoch = 0
 61:         self.global_step = 0
 62:         self.last_evaluation_feedback: EvaluationFeedback | None = None
 63: 
 64:     def _timed(self, name: str):
 65:         return self.timer(name) if self.timer is not None else nullcontext()
 66: 
 67:     def _context(
 68:             self,
 69:             *,
 70:             batch_idx: int,
 71:             stage: str = "train",
 72:             evaluation_feedback: EvaluationFeedback | None = None,
 73:     ) -> TrainContext:
 74:         return TrainContext(
 75:             epoch=self.epoch,
 76:             global_step=self.global_step,
 77:             batch_idx=int(batch_idx),
 78:             optimizer=self.optimizer,
 79:             manager=self.manager,
 80:             cfg=self.cfg,
 81:             timer=self.timer,
 82:             recorder=self.recorder,
 83:             stage=str(stage),
 84:             phase="default",
 85:             evaluation_feedback=evaluation_feedback,
 86:         )
 87: 
 88:     def predict_step(self, batch_interaction: Interaction):
 89:         return self.model.predict(batch_interaction)
 90: 
 91:     def _checkpoint_payload(self, metrics: Optional[dict] = None) -> dict:
 92:         return {
 93:             "epoch": self.epoch,
 94:             "global_step": self.global_step,
 95:             "model_state_dict": self.model.state_dict(),
 96:             "optimizer_state_dict": self.optimizer.state_dict(),
 97:             "model_trainer_state_dict": (
 98:                 self.model.trainer_state_dict()
 99:                 if isinstance(self.model, TrainerStateProtocol) else None
100:             ),
101:             "metrics": metrics,
102:             "experiment_name": self.cfg.experiment_name,
103:             "dataset_name": self.cfg.dataset_name,
104:         }
105: 
106:     def save_checkpoint(self, tag: str = "", metrics: Optional[dict] = None) -> Union[Path, None]:
107:         ckpt_dir = getattr(self.cfg, "ckpt_dir", "")
108:         if not ckpt_dir:
109:             return None
110: 
111:         save_dir = Path(ckpt_dir)
112:         save_dir.mkdir(parents=True, exist_ok=True)
113: 
114:         safe_tag = f"_{tag}" if tag else ""
115:         filename = f"epoch_{self.epoch:04d}_step_{self.global_step:08d}{safe_tag}.pt"
116:         save_path = save_dir / filename
117: 
118:         torch.save(self._checkpoint_payload(metrics), save_path)
119:         return save_path
120: 
121:     def load_checkpoint(self, path: Union[str, Path], *, map_location=None) -> dict:
122:         checkpoint = torch.load(path, map_location=map_location or self.cfg.device)
123:         self.model.load_state_dict(checkpoint["model_state_dict"])
124:         self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
125:         self.epoch = int(checkpoint.get("epoch", 0))
126:         self.global_step = int(checkpoint.get("global_step", 0))
127:         model_trainer_state = checkpoint.get("model_trainer_state_dict")
128:         if isinstance(self.model, TrainerStateProtocol) and model_trainer_state is not None:
129:             self.model.load_trainer_state_dict(model_trainer_state)
130:         return checkpoint
131: 
132:     def calculate_train_loss(self, batch, ctx: TrainContext):
133:         del ctx
134:         return self.model.calculate_loss(batch)
135: 
136:     def _default_train_step(self, batch, ctx: TrainContext):
137:         raw_loss = self.calculate_train_loss(batch, ctx)
138:         loss = self.model.lsp.settle(raw_loss)
139:         loss.backward()
140:         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
141:         self.optimizer.step()
142:         return loss
143: 
144:     def train_step(self, batch, ctx: TrainContext) -> torch.Tensor:
145:         """Run one train step inside the model's implicit ``main`` Loss scope."""
146:         with self.model.lsp.scope():
147:             if not isinstance(self.model, CustomTrainStepProtocol):
148:                 return self._default_train_step(batch, ctx)
149: 
150:             loss = self.model.custom_train_step(batch, ctx)
151:             if not isinstance(loss, torch.Tensor):
152:                 raise TypeError(
153:                     "custom_train_step() must settle and return a scalar Tensor, "
154:                     f"got {type(loss).__name__}."
155:                 )
156:             if loss.numel() != 1:
157:                 raise ValueError(
158:                     "custom_train_step() must return a scalar Tensor, "
159:                     f"got shape {tuple(loss.shape)}."
160:                 )
161:             if self.model.lsp.has_terms:
162:                 raise RuntimeError(
163:                     "custom_train_step() returned with unsettled LossPort terms."
164:                 )
165:             if not self.model.lsp.has_settlements:
166:                 raise RuntimeError(
167:                     "custom_train_step() must call model.lsp.settle() for each "
168:                     "optimization objective before returning."
169:                 )
170:             return loss
171: 
172:     def should_stop_train_epoch(self, ctx: TrainContext, loss) -> bool:
173:         """Extension point for trainers that intentionally cap an epoch."""
174:         del loss
175:         limit = int(getattr(self.cfg, "max_train_steps", 0) or 0)
176:         return limit > 0 and int(ctx.global_step) + 1 >= limit
177: 
178:     def _has_reached_train_step_limit(self) -> bool:
179:         limit = int(getattr(self.cfg, "max_train_steps", 0) or 0)
180:         return limit > 0 and self.global_step >= limit
181: 
182:     def _run_train_batches(self) -> int:
183:         log_interval = max(1, int(getattr(self.cfg, "train_log_interval", 100)))
184:         last_batch_idx = -1
185:         for batch_idx, batch_interaction in enumerate(self.train_loader):
186:             batch_interaction = batch_interaction.to(self.cfg.device)
187:             self.optimizer.zero_grad(set_to_none=True)
188: 
189:             ctx = self._context(batch_idx=batch_idx, stage="train")
190:             last_batch_idx = batch_idx
191:             current_step = self.global_step + 1
192:             self.model.lsp.start_step(
193:                 step=current_step,
194:                 epoch=ctx.epoch,
195:                 batch_idx=ctx.batch_idx,
196:                 stage=ctx.stage,
197:             )
198:             loss = self.train_step(batch_interaction, ctx)
199: 
200:             loss_value = float(loss.detach().float().cpu().item())
201:             if current_step % log_interval == 0:
202:                 print(f"step:{current_step} loss:{loss_value}")
203: 
204:             self.global_step += 1
205:             if self.should_stop_train_epoch(ctx, loss_value):
206:                 break
207:         return last_batch_idx
208: 
209:     def train_epoch(self):
210:         with self._timed("train_epoch"):
211:             self.model.train()
212:             self._run_train_batches()
213:             self.epoch += 1
214: 
215:     @torch.no_grad()
216:     def _evaluate_loader(self, model, loader):
217:         if loader is None or self.evaluator is None:
218:             return None
219: 
220:         model.eval()
221:         for batch_idx, batch_interaction in enumerate(loader):
222:             batch_interaction = batch_interaction.to(self.cfg.device)
223:             uids = batch_interaction[self.manager.uid_field]
224:             labels = batch_interaction[self.manager.label_field]
225:             if model is self.model:
226:                 scores = self.predict_step(batch_interaction)
227:             else:
228:                 scores = model.predict(batch_interaction)
229:             self.evaluator.collect(uids, labels, batch_interaction, batch_preds_1d=scores)
230: 
231:         metrics_result = self.evaluator.summary(self.epoch)
232:         self.evaluator.clear()
233:         return metrics_result
234: 
235:     def _evaluate_model(self, model):
236:         return self._evaluate_loader(model, self.valid_loader)
237: 
238:     @torch.no_grad()
239:     def evaluate_epoch(self):
240:         with self._timed("valid_epoch"):
241:             return self._evaluate_model(self.model)
242: 
243:     @torch.no_grad()
244:     def evaluate_test(self):
245:         with self._timed("test_epoch"):
246:             return self._evaluate_loader(self.model, self.test_loader)
247: 
248:     def run(
249:             self,
250:             *,
251:             stop_condition: Callable[["BaseTrainer", EvaluationFeedback], bool] | None = None,
252:     ):
253:         last_metrics = None
254:         for _ in range(self.cfg.max_epochs):
255:             if self._has_reached_train_step_limit():
256:                 break
257:             self.train_epoch()
258:             self.last_evaluation_feedback = None
259:             metrics = self.evaluate_epoch()
260:             last_metrics = metrics
261:             print(metrics)
262:             feedback = self.last_evaluation_feedback or EvaluationFeedback(
263:                 metrics_by_target={"student": metrics},
264:                 primary_target="student",
265:                 primary_metrics=metrics,
266:             )
267:             should_stop = bool(
268:                 stop_condition is not None and stop_condition(self, feedback)
269:             )
270:             if self.early_stopper:
271:                 is_best, early_stop = self.early_stopper.step(metrics, epoch=self.epoch)
272:                 status_text = self.early_stopper.status_text()
273:                 if status_text:
274:                     print(status_text)
275:                 if is_best:
276:                     save_path = self.save_checkpoint(tag="best", metrics=metrics)
277:                     if save_path is not None:
278:                         print(f"[Checkpoint] saved: {save_path}")
279:                 if early_stop:
280:                     print(f"[EarlyStop] stop at epoch {self.epoch}")
281:                     should_stop = True
282:             if should_stop:
283:                 break
284:         return last_metrics
````

## File: src/rosen/core/enum_type.py
````python
  1: # -*- coding: utf-8 -*-
  2: # @Time   : 2020/8/9
  3: # @Author : Yupeng Hou
  4: # @Email  : houyupeng@ruc.edu.cn
  5: 
  6: """
  7: recbole.utils.enum_type
  8: #######################
  9: """
 10: 
 11: from enum import Enum
 12: 
 13: 
 14: class ModelType(Enum):
 15:     """Type of models.
 16: 
 17:     - ``GENERAL``: General Recommendation
 18:     - ``SEQUENTIAL``: Sequential Recommendation
 19:     - ``CONTEXT``: Context-aware Recommendation
 20:     - ``KNOWLEDGE``: Knowledge-based Recommendation
 21:     """
 22: 
 23:     GENERAL = 1
 24:     SEQUENTIAL = 2
 25:     CONTEXT = 3
 26:     KNOWLEDGE = 4
 27:     TRADITIONAL = 5
 28:     DECISIONTREE = 6
 29: 
 30: 
 31: class KGDataLoaderState(Enum):
 32:     """States for Knowledge-based DataLoader.
 33: 
 34:     - ``RSKG``: Return both knowledge graph information and user-item interaction information.
 35:     - ``RS``: Only return the user-item interaction.
 36:     - ``KG``: Only return the triplets with negative examples in a knowledge graph.
 37:     """
 38: 
 39:     RSKG = 1
 40:     RS = 2
 41:     KG = 3
 42: 
 43: 
 44: class EvaluatorType(Enum):
 45:     """Type for evaluation metrics.
 46: 
 47:     - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
 48:     - ``VALUE``: Value-based metrics like AUC, etc.
 49:     """
 50: 
 51:     RANKING = 1
 52:     VALUE = 2
 53: 
 54: 
 55: class InputType(Enum):
 56:     """Type of Models' input.
 57: 
 58:     - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
 59:     - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
 60:     """
 61: 
 62:     POINTWISE = 1
 63:     PAIRWISE = 2
 64:     LISTWISE = 3
 65: 
 66: 
 67: class FeatureType(Enum):
 68:     """Type of features.
 69: 
 70:     - ``TOKEN``: Token features like user_id and item_id.
 71:     - ``FLOAT``: Float features like rating and timestamp.
 72:     - ``TOKEN_SEQ``: Token sequence features like review.
 73:     - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
 74:     """
 75: 
 76:     TOKEN = "token"
 77:     FLOAT = "float"
 78:     TOKEN_SEQ = "token_seq"
 79:     FLOAT_SEQ = "float_seq"
 80: 
 81: 
 82: class FeatureSource(Enum):
 83:     """Source of features.
 84: 
 85:     - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
 86:     - ``USER``: Features from ``.user`` (other than ``user_id``).
 87:     - ``ITEM``: Features from ``.item`` (other than ``item_id``).
 88:     - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
 89:     - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
 90:     - ``KG``: Features from ``.kg``.
 91:     - ``NET``: Features from ``.net``.
 92:     """
 93:     UNKNOWN = "unknown"
 94:     INTERACTION = "inter"
 95:     USER = "user"
 96:     ITEM = "item"
 97:     USER_ID = "user_id"
 98:     ITEM_ID = "item_id"
 99:     SEQ = "seq"
100:     SEQ_GROUP = "seq_group"
101:     KG = "kg"
102:     NET = "net"
````

## File: src/rosen/core/interaction.py
````python
  1: # @Time   : 2020/7/10
  2: # @Author : Yupeng Hou
  3: # @Email  : houyupeng@ruc.edu.cn
  4: 
  5: # UPDATE
  6: # @Time    : 2022/7/8, 2020/9/15, 2020/9/16, 2020/8/12
  7: # @Author  : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan
  8: # @email   : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn
  9: 
 10: """
 11: recbole.data.interaction
 12: ############################
 13: """
 14: 
 15: import numpy as np
 16: import pandas as pd
 17: import torch
 18: import torch.nn.utils.rnn as rnn_utils
 19: 
 20: # 他只是对齐......
 21: def _convert_to_tensor(data):
 22:     """
 23:     增强版转换函数：完美兼容 uint32 并自动对齐 PyTorch 常用类型。
 24:     """
 25:     # 1. 预处理：如果是含有 uint32 的 NumPy 数组，强制转为 int64 (Long)
 26:     # 因为 PyTorch 不支持 uint32，且 ID 类数据转 int64 最保险（防止溢出）
 27:     if isinstance(data, np.ndarray) and data.dtype == np.uint32:
 28:         data = data.astype(np.int64)
 29:     elif isinstance(data, pd.Series) and data.dtype == "uint32":
 30:         data = data.astype(np.int64)
 31: 
 32:     elem = data[0]
 33: 
 34:     # 情况 A：单值/数值数组处理
 35:     if isinstance(elem, (float, int, np.floating, np.integer)):
 36:         new_data = torch.as_tensor(data)
 37: 
 38:     # 情况 B：序列处理 (例如 tags, history_videos)
 39:     elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
 40:         # 对每一个子序列进行递归或类型防御
 41:         seq_data = []
 42:         for d in data:
 43:             # 防御子序列中的 uint32
 44:             if isinstance(d, np.ndarray) and d.dtype == np.uint32:
 45:                 d = d.astype(np.int64)
 46:             seq_data.append(torch.as_tensor(d))
 47: 
 48:         new_data = rnn_utils.pad_sequence(seq_data, batch_first=True) # 这里对序列做了pad保证同长度！
 49: 
 50:     else:
 51:         raise ValueError(f"[{type(elem)}] is not supported!")
 52: 
 53:     # 3. 类型后置对齐
 54:     # float64 转 float32 是推荐系统的基操，节省一半显存
 55:     if new_data.dtype == torch.float64:
 56:         new_data = new_data.float()
 57:     # uint8 虽然 PyTorch 支持，但在做 Embedding 索引时通常需要 Long
 58:     elif new_data.dtype == torch.uint8:
 59:         new_data = new_data.long()
 60: 
 61:     return new_data
 62: 
 63: 
 64: class Interaction(object):
 65:     """The basic class representing a batch of interaction records.
 66: 
 67:     Note:
 68:         While training, there is no strict rules for data in one Interaction object.
 69: 
 70:         While testing, it should be guaranteed that all interaction records of one single
 71:         user will not appear in different Interaction object, and records of the same user
 72:         should be continuous. Meanwhile, the positive cases of one user always need to occur
 73:         **earlier** than this user's negative cases.
 74: 
 75:         A correct example:
 76:             =======     =======     =======
 77:             user_id     item_id     label
 78:             =======     =======     =======
 79:             1           2           1
 80:             1           6           1
 81:             1           3           1
 82:             1           1           0
 83:             2           3           1
 84:             ...         ...         ...
 85:             =======     =======     =======
 86: 
 87:         Some wrong examples for Interaction objects used in testing:
 88: 
 89:         1.
 90:             =======     =======     =======     ============
 91:             user_id     item_id     label
 92:             =======     =======     =======     ============
 93:             1           2           1
 94:             1           6           0           # positive cases of one user always need to
 95: 
 96:                                                 occur earlier than this user's negative cases
 97:             1           3           1
 98:             1           1           0
 99:             2           3           1
100:             ...         ...         ...
101:             =======     =======     =======     ============
102: 
103:         2.
104:             =======     =======     =======     ========
105:             user_id     item_id     label
106:             =======     =======     =======     ========
107:             1           2           1
108:             1           6           1
109:             1           3           1
110:             2           3           1           # records of the same user should be continuous.
111:             1           1           0
112:             ...         ...         ...
113:             =======     =======     =======     ========
114: 
115:     Attributes:
116:         interaction (dict or pandas.DataFrame): keys are meaningful str (also can be called field name),
117:             and values are Torch Tensor of numpy Array with shape (batch_size, \\*).
118:     """
119: 
120:     def __init__(self, interaction):
121:         self.interaction = dict()
122:         if isinstance(interaction, dict):
123:             for key, value in interaction.items():
124:                 if isinstance(value, (list, np.ndarray)):
125:                     self.interaction[key] = _convert_to_tensor(value)
126:                 elif isinstance(value, torch.Tensor):
127:                     self.interaction[key] = value
128:                 else:
129:                     raise ValueError(
130:                         f"The type of {key}[{type(value)}] is not supported!"
131:                     )
132:         elif isinstance(interaction, pd.DataFrame):
133:             for key in interaction:
134:                 value = interaction[key].values
135:                 self.interaction[key] = _convert_to_tensor(value)
136:         else:
137:             raise ValueError(
138:                 f"[{type(interaction)}] is not supported for initialize `Interaction`!"
139:             )
140:         self.length = -1
141:         for k in self.interaction:
142:             self.length = max(self.length, self.interaction[k].unsqueeze(-1).shape[0])
143: 
144:     def __iter__(self):
145:         return self.interaction.__iter__()
146: 
147:     def __getattr__(self, item):
148:         if "interaction" not in self.__dict__:
149:             raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
150:         if item in self.interaction:
151:             return self.interaction[item]
152:         raise AttributeError(f"'Interaction' object has no attribute '{item}'")
153: 
154:     def __getitem__(self, index):
155:         if isinstance(index, str):
156:             return self.interaction[index]
157:         if isinstance(index, (np.ndarray, torch.Tensor)):
158:             index = index.tolist()
159: 
160:         ret = {}
161:         for k in self.interaction:
162:             ret[k] = self.interaction[k][index]
163:         return Interaction(ret)
164: 
165:     def __setitem__(self, key, value):
166:         if not isinstance(key, str):
167:             raise KeyError(f"{type(key)} object does not support item assigment")
168:         self.interaction[key] = value
169: 
170:     def __delitem__(self, key):
171:         if key not in self.interaction:
172:             raise KeyError(f"{type(key)} object does not in this interaction")
173:         del self.interaction[key]
174: 
175:     def __contains__(self, item):
176:         return item in self.interaction
177: 
178:     def __len__(self):
179:         return self.length
180: 
181:     def __str__(self):
182:         info = [f"The batch_size of interaction: {self.length}"]
183:         for k in self.interaction:
184:             inter = self.interaction[k]
185:             temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
186:             info.append(temp_str)
187:         info.append("\n")
188:         return "\n".join(info)
189: 
190:     def __repr__(self):
191:         return self.__str__()
192: 
193:     @property
194:     def columns(self):
195:         """
196:         Returns:
197:             list of str: The columns of interaction.
198:         """
199:         return list(self.interaction.keys())
200: 
201:     def to(self, device, selected_field=None):
202:         """Transfer Tensors in this Interaction object to the specified device.
203: 
204:         Args:
205:             device (torch.device), str: target device.
206:             selected_field (str or iterable object, optional): if specified, only Tensors
207:             with keys in selected_field will be sent to device.
208: 
209:         Returns:
210:             Interaction: a coped Interaction object with Tensors which are sent to
211:             the specified device.
212:         """
213:         ret = {}
214:         if isinstance(selected_field, str):
215:             selected_field = [selected_field]
216: 
217:         if selected_field is not None:
218:             selected_field = set(selected_field)
219:             for k in self.interaction:
220:                 if k in selected_field:
221:                     ret[k] = self.interaction[k].to(device)
222:                 else:
223:                     ret[k] = self.interaction[k]
224:         else:
225:             for k in self.interaction:
226:                 ret[k] = self.interaction[k].to(device)
227:         return Interaction(ret)
228: 
229:     def cpu(self):
230:         """Transfer Tensors in this Interaction object to cpu.
231: 
232:         Returns:
233:             Interaction: a coped Interaction object with Tensors which are sent to cpu.
234:         """
235:         ret = {}
236:         for k in self.interaction:
237:             ret[k] = self.interaction[k].cpu()
238:         return Interaction(ret)
239: 
240:     def numpy(self):
241:         """Transfer Tensors to numpy arrays.
242: 
243:         Returns:
244:             dict: keys the same as Interaction object, are values are corresponding numpy
245:             arrays transformed from Tensor.
246:         """
247:         ret = {}
248:         for k in self.interaction:
249:             ret[k] = self.interaction[k].numpy()
250:         return ret
251: 
252:     def repeat(self, sizes):
253:         """Repeats each tensor along the batch dim.
254: 
255:         Args:
256:             sizes (int): repeat times.
257: 
258:         Example:
259:             >>> a = Interaction({'k': torch.zeros(4)})
260:             >>> a.repeat(3)
261:             The batch_size of interaction: 12
262:                 k, torch.Size([12]), cpu
263: 
264:             >>> a = Interaction({'k': torch.zeros(4, 7)})
265:             >>> a.repeat(3)
266:             The batch_size of interaction: 12
267:                 k, torch.Size([12, 7]), cpu
268: 
269:         Returns:
270:             a copyed Interaction object with repeated Tensors.
271:         """
272:         ret = {}
273:         for k in self.interaction:
274:             ret[k] = self.interaction[k].repeat(
275:                 [sizes] + [1] * (len(self.interaction[k].shape) - 1)
276:             )
277:         return Interaction(ret)
278: 
279:     def repeat_interleave(self, repeats, dim=0):
280:         """Similar to repeat_interleave of PyTorch.
281: 
282:         Details can be found in:
283: 
284:             https://pytorch.org/docs/stable/tensors.html?highlight=repeat#torch.Tensor.repeat_interleave
285: 
286:         Note:
287:             ``torch.repeat_interleave()`` is supported in PyTorch >= 1.2.0.
288:         """
289:         ret = {}
290:         for k in self.interaction:
291:             ret[k] = self.interaction[k].repeat_interleave(repeats, dim=dim)
292:         return Interaction(ret)
293: 
294:     def update(self, new_inter):
295:         """Similar to ``dict.update()``
296: 
297:         Args:
298:             new_inter (Interaction): current interaction will be updated by new_inter.
299:         """
300:         for k in new_inter.interaction:
301:             self.interaction[k] = new_inter.interaction[k]
302: 
303:     def drop(self, column):
304:         """Drop column in interaction.
305: 
306:         Args:
307:             column (str): the column to be dropped.
308:         """
309:         if column not in self.interaction:
310:             raise ValueError(f"Column [{column}] is not in [{self}].")
311:         del self.interaction[column]
312: 
313:     def _reindex(self, index):
314:         """Reset the index of interaction inplace.
315: 
316:         Args:
317:             index: the new index of current interaction.
318:         """
319:         for k in self.interaction:
320:             self.interaction[k] = self.interaction[k][index]
321: 
322:     def shuffle(self):
323:         """Shuffle current interaction inplace."""
324:         index = torch.randperm(self.length)
325:         self._reindex(index)
326: 
327:     def sort(self, by, ascending=True):
328:         """Sort the current interaction inplace.
329: 
330:         Args:
331:             by (str or list of str): Field that as the key in the sorting process.
332:             ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
333:                 Defaults to ``True``
334:         """
335:         if isinstance(by, str):
336:             if by not in self.interaction:
337:                 raise ValueError(f"[{by}] is not exist in interaction [{self}].")
338:             by = [by]
339:         elif isinstance(by, (list, tuple)):
340:             for b in by:
341:                 if b not in self.interaction:
342:                     raise ValueError(f"[{b}] is not exist in interaction [{self}].")
343:         else:
344:             raise TypeError(f"Wrong type of by [{by}].")
345: 
346:         if isinstance(ascending, bool):
347:             ascending = [ascending]
348:         elif isinstance(ascending, (list, tuple)):
349:             for a in ascending:
350:                 if not isinstance(a, bool):
351:                     raise TypeError(f"Wrong type of ascending [{ascending}].")
352:         else:
353:             raise TypeError(f"Wrong type of ascending [{ascending}].")
354: 
355:         if len(by) != len(ascending):
356:             if len(ascending) == 1:
357:                 ascending = ascending * len(by)
358:             else:
359:                 raise ValueError(
360:                     f"by [{by}] and ascending [{ascending}] should have same length."
361:                 )
362: 
363:         for b, a in zip(by[::-1], ascending[::-1]):
364:             if len(self.interaction[b].shape) == 1:
365:                 key = self.interaction[b]
366:             else:
367:                 key = self.interaction[b][..., 0]
368:             index = np.argsort(key, kind="stable")
369:             if not a:
370:                 index = torch.tensor(np.array(index)[::-1])
371:             self._reindex(index)
372: 
373:     def add_prefix(self, prefix):
374:         """Add prefix to current interaction's columns.
375: 
376:         Args:
377:             prefix (str): The prefix to be added.
378:         """
379:         self.interaction = {
380:             prefix + key: value for key, value in self.interaction.items()
381:         }
382: 
383: 
384: def cat_interactions(interactions):
385:     """Concatenate list of interactions to single interaction.
386: 
387:     Args:
388:         interactions (list of :class:`Interaction`): List of interactions to be concatenated.
389: 
390:     Returns:
391:         :class:`Interaction`: Concatenated interaction.
392:     """
393:     if not isinstance(interactions, (list, tuple)):
394:         raise TypeError(f"Interactions [{interactions}] should be list or tuple.")
395:     if len(interactions) == 0:
396:         raise ValueError(
397:             f"Interactions [{interactions}] should have some interactions."
398:         )
399: 
400:     columns_set = set(interactions[0].columns)
401:     for inter in interactions:
402:         if columns_set != set(inter.columns):
403:             raise ValueError(
404:                 f"Interactions [{interactions}] should have some interactions."
405:             )
406: 
407:     new_inter = {
408:         col: torch.cat([inter[col] for inter in interactions]) for col in columns_set
409:     }
410:     return Interaction(new_inter)
````

## File: src/rosen/data/dataset.py
````python
  1: from __future__ import annotations
  2: 
  3: from pathlib import Path
  4: from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union
  5: 
  6: import numpy as np
  7: import polars as pl
  8: import pyarrow as pa
  9: import pyarrow.compute as pc
 10: import pyarrow.parquet as pq
 11: import torch
 12: from torch.utils.data import IterableDataset
 13: 
 14: from rosen.emb import SchemaManager
 15: from .padding import ColumnFormatter, FeatureContext, TensorFormatter
 16: 
 17: 
 18: ScannerSource = Union[str, Path, Sequence[Union[str, Path]], pl.LazyFrame]
 19: 
 20: 
 21: def _resolve_parquet_paths(source: Union[str, Path, Sequence[Union[str, Path]]]) -> list[str]:
 22:     if isinstance(source, (list, tuple)):
 23:         paths = [str(Path(path)) for path in source]
 24:     else:
 25:         path_obj = Path(source)
 26:         if path_obj.is_dir():
 27:             paths = [str(path) for path in sorted(path_obj.glob("*.parquet"))]
 28:         else:
 29:             paths = [str(path_obj)]
 30: 
 31:     if not paths:
 32:         raise FileNotFoundError(f"No parquet files found from input: {source}")
 33:     return paths
 34: 
 35: 
 36: def _parquet_top_level_columns(path: Union[str, Path]) -> set[str]:
 37:     return set(pq.ParquetFile(path).schema_arrow.names)
 38: 
 39: 
 40: def _fill_null_scalar(col_type: pa.DataType):
 41:     if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
 42:         return pa.scalar("", type=col_type)
 43:     if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
 44:         return pa.scalar([], type=col_type)
 45:     if pa.types.is_boolean(col_type):
 46:         return pa.scalar(False, type=col_type)
 47:     return pa.scalar(0, type=col_type)
 48: 
 49: 
 50: def arrow_table_to_batch_dict(
 51:     table: pa.Table,
 52:     selected_col_names: Optional[Sequence[str]] = None,
 53: ) -> Dict[str, np.ndarray]:
 54:     selected_set = set(selected_col_names) if selected_col_names is not None else None
 55:     batch_dict: Dict[str, np.ndarray] = {}
 56:     for col_name, col_data in zip(table.column_names, table.columns):
 57:         if selected_set is not None and col_name not in selected_set:
 58:             continue
 59: 
 60:         clean_col = pc.fill_null(col_data, _fill_null_scalar(col_data.type))
 61:         batch_dict[col_name] = clean_col.to_numpy(zero_copy_only=False)
 62:     return batch_dict
 63: 
 64: 
 65: def _concat_tables(tables: Sequence[pa.Table]) -> Optional[pa.Table]:
 66:     non_empty_tables = [table for table in tables if table is not None and table.num_rows > 0]
 67:     if not non_empty_tables:
 68:         return None
 69:     if len(non_empty_tables) == 1:
 70:         return non_empty_tables[0]
 71:     return pa.concat_tables(non_empty_tables)
 72: 
 73: 
 74: class DataScanner:
 75:     def __init__(
 76:         self,
 77:         source: ScannerSource,
 78:         read_cols: Optional[Iterable[str]] = None,
 79:         filter_expr: Optional[pl.Expr] = None,
 80:     ):
 81:         self.source = source
 82:         self.read_cols = list(dict.fromkeys(read_cols or [])) or None
 83:         self.filter_expr = filter_expr
 84:         self.parquet_paths = None if isinstance(source, pl.LazyFrame) else _resolve_parquet_paths(source)
 85:         self.lazy_frame = self._build_lazy_frame(source)
 86: 
 87:     def _build_lazy_frame(self, source: ScannerSource) -> pl.LazyFrame:
 88:         if isinstance(source, pl.LazyFrame):
 89:             lf = source
 90:         else:
 91:             lf = pl.scan_parquet(self.parquet_paths)
 92: 
 93:         if self.filter_expr is not None:
 94:             lf = lf.filter(self.filter_expr)
 95:         if self.read_cols is not None:
 96:             lf = lf.select(self.read_cols)
 97:         return lf
 98: 
 99:     def _apply_worker_shard(self, lf: pl.LazyFrame, worker_id: int, num_workers: int) -> pl.LazyFrame:
100:         if num_workers <= 1:
101:             return lf
102:         worker_col = "__scanner_worker_row_id"
103:         return (
104:             lf.with_row_index(worker_col)
105:             .filter((pl.col(worker_col) % num_workers) == worker_id)
106:             .drop(worker_col)
107:         )
108: 
109:     def _iter_lazy_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
110:         lf = self._apply_worker_shard(self.lazy_frame, worker_id=worker_id, num_workers=num_workers)
111:         for batch_df in lf.collect_batches(
112:             chunk_size=batch_size,
113:             maintain_order=True,
114:             engine="streaming",
115:         ):
116:             if batch_df.height == 0:
117:                 continue
118:             yield batch_df.to_arrow()
119: 
120:     def _iter_parquet_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
121:         row_group_offset = 0
122:         for parquet_file in self.parquet_paths or []:
123:             parquet_file_reader = pq.ParquetFile(parquet_file)
124:             row_group_count = parquet_file_reader.metadata.num_row_groups
125:             if num_workers > 1:
126:                 row_groups = [
127:                     row_group_idx
128:                     for row_group_idx in range(row_group_count)
129:                     if (row_group_offset + row_group_idx) % num_workers == worker_id
130:                 ]
131:                 if not row_groups:
132:                     row_group_offset += row_group_count
133:                     continue
134:             else:
135:                 row_groups = None
136: 
137:             for record_batch in parquet_file_reader.iter_batches(
138:                 batch_size=batch_size,
139:                 row_groups=row_groups,
140:                 columns=self.read_cols,
141:             ):
142:                 yield pa.Table.from_batches([record_batch])
143:             row_group_offset += row_group_count
144: 
145:     def iter_batches(
146:         self,
147:         batch_size: int,
148:         worker_id: int = 0,
149:         num_workers: int = 1,
150:     ) -> Iterator[pa.Table]:
151:         if self.parquet_paths is not None and self.filter_expr is None:
152:             yield from self._iter_parquet_batches(batch_size, worker_id, num_workers)
153:             return
154: 
155:         yield from self._iter_lazy_batches(batch_size, worker_id, num_workers)
156: 
157: 
158: class DataTransformer:
159:     def __init__(
160:         self,
161:         manager: SchemaManager,
162:         preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
163:         filter_expr: Optional[pl.Expr] = None,
164:         output_col_names: Optional[Iterable[str]] = None,
165:     ):
166:         self.manager = manager
167:         self.preprocess_exprs = list(preprocess_exprs or [])
168:         self.filter_expr = filter_expr
169:         self.output_col_names = list(dict.fromkeys(output_col_names or manager.fields()))
170: 
171:     def __call__(self, raw_table: pa.Table) -> Optional[pa.Table]:
172:         if raw_table.num_rows == 0:
173:             return None
174: 
175:         raw_df = pl.from_arrow(raw_table)
176:         if self.preprocess_exprs:
177:             raw_df = raw_df.with_columns(self.preprocess_exprs)
178:         if self.filter_expr is not None:
179:             raw_df = raw_df.filter(self.filter_expr)
180:             if raw_df.height == 0:
181:                 return None
182: 
183:         transformed_df = (
184:             self.manager.transform(raw_df.lazy())
185:             .select(self.output_col_names)
186:             .collect(engine="streaming")
187:         )
188:         if transformed_df.height == 0:
189:             return None
190:         return transformed_df.to_arrow()
191: 
192: 
193: class ShuffleBuffer:
194:     def __init__(
195:         self,
196:         capacity: int,
197:         batch_size: int,
198:         shuffle: bool,
199:         drop_last: bool = True,
200:         selected_col_names: Optional[Sequence[str]] = None,
201:     ):
202:         self.capacity = max(batch_size, int(capacity))
203:         self.batch_size = int(batch_size)
204:         self.shuffle = shuffle
205:         self.drop_last = drop_last
206:         self.selected_col_names = list(selected_col_names) if selected_col_names is not None else None
207:         self.reset()
208: 
209:     def reset(self):
210:         self.buffer: list[pa.Table] = []
211:         self.current_rows = 0
212: 
213:     def push(self, table: Optional[pa.Table]):
214:         if table is None or table.num_rows == 0:
215:             return
216:         self.buffer.append(table)
217:         self.current_rows += table.num_rows
218: 
219:     def _drain(self, force: bool) -> Iterator[Dict[str, np.ndarray]]:
220:         table = _concat_tables(self.buffer)
221:         if table is None or table.num_rows == 0:
222:             self.reset()
223:             return
224: 
225:         total_rows = table.num_rows
226:         batch_dict = arrow_table_to_batch_dict(table, selected_col_names=self.selected_col_names)
227:         indices = np.random.permutation(total_rows) if self.shuffle else np.arange(total_rows)
228:         num_complete_batches = total_rows // self.batch_size
229: 
230:         for batch_idx in range(num_complete_batches):
231:             start_idx = batch_idx * self.batch_size
232:             end_idx = start_idx + self.batch_size
233:             batch_indices = indices[start_idx:end_idx]
234:             yield {col: arr[batch_indices] for col, arr in batch_dict.items()}
235: 
236:         residual_start = num_complete_batches * self.batch_size
237:         residual_indices = indices[residual_start:]
238:         residual_count = len(residual_indices)
239: 
240:         self.reset()
241:         if residual_count == 0:
242:             return
243: 
244:         if force:
245:             if not self.drop_last:
246:                 yield {col: arr[residual_indices] for col, arr in batch_dict.items()}
247:             return
248: 
249:         residual_table = table.take(pa.array(residual_indices, type=pa.int64()))
250:         self.buffer = [residual_table]
251:         self.current_rows = residual_table.num_rows
252: 
253:     def yield_batches(self) -> Iterator[Dict[str, np.ndarray]]:
254:         if self.current_rows < self.capacity:
255:             return
256:         yield from self._drain(force=False)
257: 
258:     def flush(self) -> Iterator[Dict[str, np.ndarray]]:
259:         yield from self._drain(force=True)
260: 
261: 
262: class PipelineStreamDataset(IterableDataset):
263:     def __init__(
264:         self,
265:         scanner: DataScanner,
266:         formatter: TensorFormatter,
267:         transformer: Optional[DataTransformer] = None,
268:         buffer: Optional[ShuffleBuffer] = None,
269:         scan_batch_size: Optional[int] = None,
270:     ):
271:         super().__init__()
272:         self.scanner = scanner
273:         self.transformer = transformer
274:         self.buffer = buffer
275:         self.formatter = formatter
276:         self.scan_batch_size = scan_batch_size or (buffer.batch_size if buffer is not None else 4096)
277: 
278:     def __iter__(self):
279:         worker_info = torch.utils.data.get_worker_info()
280:         worker_id = worker_info.id if worker_info is not None else 0
281:         num_workers = worker_info.num_workers if worker_info is not None else 1
282: 
283:         if self.buffer is not None:
284:             self.buffer.reset()
285: 
286:         for raw_table in self.scanner.iter_batches(
287:             batch_size=self.scan_batch_size,
288:             worker_id=worker_id,
289:             num_workers=num_workers,
290:         ):
291:             table = self.transformer(raw_table) if self.transformer else raw_table
292:             if table is None or table.num_rows == 0:
293:                 continue
294: 
295:             if self.buffer is None:
296:                 yield self.formatter.format(
297:                     arrow_table_to_batch_dict(table, selected_col_names=self.formatter.context.output_col_names)
298:                 )
299:                 continue
300: 
301:             self.buffer.push(table)
302:             for batch_dict in self.buffer.yield_batches():
303:                 yield self.formatter.format(batch_dict)
304: 
305:         if self.buffer is not None:
306:             for batch_dict in self.buffer.flush():
307:                 yield self.formatter.format(batch_dict)
308: 
309: 
310: class ParquetStreamDataset(PipelineStreamDataset):
311:     def __init__(
312:         self,
313:         parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
314:         manager: SchemaManager,
315:         batch_size: int = 4096,
316:         shuffle: bool = True,
317:         drop_last: bool = True,
318:         shuffle_buffer_size: int = 2000000,
319:         extra_col_names: Optional[Iterable[str]] = None,
320:         extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
321:     ):
322:         context = FeatureContext.from_manager(
323:             manager,
324:             extra_col_names=extra_col_names,
325:             extra_col_formatters=extra_col_formatters,
326:         )
327:         scanner = DataScanner(parquet_path, read_cols=context.read_col_names)
328:         formatter = TensorFormatter(context)
329:         buffer = ShuffleBuffer(
330:             capacity=shuffle_buffer_size if shuffle else batch_size,
331:             batch_size=batch_size,
332:             shuffle=shuffle,
333:             drop_last=drop_last,
334:             selected_col_names=context.output_col_names,
335:         )
336:         super().__init__(
337:             scanner=scanner,
338:             formatter=formatter,
339:             transformer=None,
340:             buffer=buffer,
341:             scan_batch_size=batch_size * 4 if shuffle else batch_size,
342:         )
343: 
344:         self.parquet_path = parquet_path
345:         self.parquet_paths = list(scanner.parquet_paths or [])
346:         self.manager = manager
347:         self.batch_size = batch_size
348:         self.shuffle = shuffle
349:         self.drop_last = drop_last
350:         self.trigger_size = buffer.capacity
351:         self.context = context
352:         self.output_col_names = list(context.output_col_names)
353:         self.valid_col_names = list(context.output_col_names)
354:         self.read_col_names = list(context.read_col_names)
355:         self.extra_col_names = list(extra_col_names or [])
356:         self.extra_col_formatters = dict(extra_col_formatters or {})
357:         self.col_formatters = formatter.col_formatters
358: 
359: 
360: def _empty_object_sequences(batch_size: int) -> np.ndarray:
361:     values = np.empty(batch_size, dtype=object)
362:     for idx in range(batch_size):
363:         values[idx] = []
364:     return values
365: 
366: 
367: def _default_missing_column(name: str, formatter: ColumnFormatter, batch_size: int) -> np.ndarray:
368:     formatter_name = formatter.__class__.__name__
369:     if formatter_name.startswith("Padded"):
370:         return _empty_object_sequences(batch_size)
371:     if formatter_name in {"DenseFormatter"}:
372:         return np.zeros(batch_size, dtype=np.float32)
373:     if formatter_name in {"VectorDenseFormatter"}:
374:         dim = int(getattr(formatter, "dim", 0))
375:         return np.zeros((batch_size, dim), dtype=np.float32)
376:     return np.zeros(batch_size, dtype=np.int64)
377: 
378: 
379: class ProfileSequenceStore:
380:     HISTORY_TIME_FILTERS = {
381:         "cursor_order",
382:         "strict_time",
383:         "drop_same_ts",
384:         "same_ts_only",
385:     }
386: 
387:     def __init__(
388:         self,
389:         manager: SchemaManager,
390:         profile_root: Union[str, Path],
391:         group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
392:     ):
393:         self.manager = manager
394:         self.profile_root = Path(profile_root)
395:         self.group_dirs = {str(k): Path(v) for k, v in dict(group_dirs or {}).items()}
396:         self.groups = self._compile_groups()
397:         self.tables = {
398:             group_name: self._load_group(group_name, spec)
399:             for group_name, spec in self.groups.items()
400:         }
401: 
402:     def _compile_groups(self) -> Dict[str, dict]:
403:         groups: Dict[str, dict] = {}
404:         for setting in self.manager.settings:
405:             if not setting.is_sequence_setting:
406:                 continue
407:             group_name = setting.group_name
408:             group = setting.group
409:             spec = groups.setdefault(
410:                 group_name,
411:                 {
412:                     "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
413:                     "seq_len_col": setting.seq_len_field_name,
414:                     "max_len": int(setting.max_len),
415:                     "fields": [],
416:                     "time_field": setting.time_field_name,
417:                     "anchor_col": getattr(group, "anchor_field_name", None),
418:                     "history_time_filter": getattr(
419:                         group,
420:                         "history_time_filter",
421:                         "cursor_order",
422:                     ),
423:                     "truncate_mode": setting.truncate_mode,
424:                 },
425:             )
426:             shared_values = {
427:                 "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
428:                 "seq_len_col": setting.seq_len_field_name,
429:                 "max_len": int(setting.max_len),
430:                 "time_field": setting.time_field_name,
431:                 "anchor_col": getattr(group, "anchor_field_name", None),
432:                 "history_time_filter": getattr(group, "history_time_filter", "cursor_order"),
433:                 "truncate_mode": setting.truncate_mode,
434:             }
435:             for key, value in shared_values.items():
436:                 if spec[key] != value:
437:                     raise ValueError(
438:                         f"Profile sequence group '{group_name}' has inconsistent "
439:                         f"{key}: {spec[key]!r} != {value!r}."
440:                     )
441:             if setting.field_name not in spec["fields"]:
442:                 spec["fields"].append(setting.field_name)
443:             if setting.time_field_name and setting.time_field_name not in spec["fields"]:
444:                 spec["fields"].append(setting.time_field_name)
445:             if spec["history_time_filter"] not in self.HISTORY_TIME_FILTERS:
446:                 raise ValueError(
447:                     f"Unknown history_time_filter {spec['history_time_filter']!r} "
448:                     f"for profile sequence group '{group_name}'."
449:                 )
450:         return groups
451: 
452:     def _resolve_group_dir(self, group_name: str) -> Path:
453:         if group_name in self.group_dirs:
454:             return self.group_dirs[group_name]
455:         return self.profile_root / group_name
456: 
457:     @staticmethod
458:     def _profile_value_to_array(value) -> np.ndarray:
459:         if value is None:
460:             return np.asarray([], dtype=np.int64)
461:         try:
462:             return np.asarray(value, dtype=np.int64)
463:         except (TypeError, ValueError):
464:             return np.asarray(value, dtype=object)
465: 
466:     def _load_group(self, group_name: str, spec: dict) -> Dict[int, Dict[str, np.ndarray]]:
467:         group_dir = self._resolve_group_dir(group_name)
468:         if group_dir.is_dir():
469:             files = sorted(group_dir.glob("*.parquet"))
470:         else:
471:             files = [group_dir] if group_dir.exists() else []
472:         if not files:
473:             raise FileNotFoundError(f"No profile parquet files found for group '{group_name}' under {group_dir}")
474: 
475:         read_cols = [spec["owner_col"], *spec["fields"]]
476:         df = pl.read_parquet([str(path) for path in files], columns=read_cols)
477:         table: Dict[int, Dict[str, np.ndarray]] = {}
478:         for row in df.iter_rows(named=True):
479:             uid = int(row[spec["owner_col"]])
480:             table[uid] = {}
481:             for field in spec["fields"]:
482:                 value = row[field]
483:                 table[uid][field] = self._profile_value_to_array(value)
484:         return table
485: 
486:     def inject(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
487:         if not self.groups:
488:             return batch_dict
489:         batch_size = len(next(iter(batch_dict.values())))
490: 
491:         for group_name, spec in self.groups.items():
492:             owner_col = spec["owner_col"]
493:             seq_len_col = spec["seq_len_col"]
494:             if owner_col not in batch_dict or seq_len_col not in batch_dict:
495:                 raise KeyError(
496:                     f"Profile group '{group_name}' requires columns "
497:                     f"'{owner_col}' and '{seq_len_col}' in main batch."
498:                 )
499: 
500:             users = np.asarray(batch_dict[owner_col]).reshape(-1)
501:             cursors = np.asarray(batch_dict[seq_len_col]).reshape(-1)
502:             table = self.tables[group_name]
503:             max_len = int(spec["max_len"])
504:             history_time_filter = spec["history_time_filter"]
505:             time_field = spec["time_field"]
506:             anchor_col = spec["anchor_col"]
507:             anchors = None
508:             if history_time_filter != "cursor_order":
509:                 if not time_field or not anchor_col:
510:                     raise ValueError(
511:                         f"Profile group '{group_name}' requires time_field_name and "
512:                         f"anchor_field_name for history_time_filter={history_time_filter!r}."
513:                     )
514:                 if anchor_col not in batch_dict:
515:                     raise KeyError(
516:                         f"Profile group '{group_name}' requires anchor column "
517:                         f"'{anchor_col}' in main batch."
518:                     )
519:                 anchors = np.asarray(batch_dict[anchor_col]).reshape(-1)
520: 
521:             sliced_fields = {
522:                 field: np.empty(batch_size, dtype=object)
523:                 for field in spec["fields"]
524:             }
525:             actual_lens = np.zeros(batch_size, dtype=np.int64)
526: 
527:             for row_idx, (uid, cursor) in enumerate(zip(users, cursors)):
528:                 profile = table.get(int(uid))
529:                 cursor_int = max(0, int(cursor))
530:                 base_field = spec["fields"][0]
531:                 base_seq = profile.get(base_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
532:                 end = min(cursor_int, len(base_seq))
533:                 selected_indices = np.arange(end, dtype=np.int64)
534: 
535:                 if history_time_filter != "cursor_order":
536:                     history_time = profile.get(time_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
537:                     if len(history_time) < end:
538:                         raise ValueError(
539:                             f"Profile group '{group_name}' has {len(history_time)} "
540:                             f"timestamps for cursor {end} (user={int(uid)})."
541:                         )
542:                     candidate_time = history_time[:end]
543:                     anchor = anchors[row_idx]
544:                     if history_time_filter in {"strict_time", "drop_same_ts"}:
545:                         selected_indices = selected_indices[candidate_time < anchor]
546:                     else:
547:                         selected_indices = selected_indices[candidate_time == anchor]
548: 
549:                 if spec["truncate_mode"] == "head":
550:                     selected_indices = selected_indices[:max_len]
551:                 elif spec["truncate_mode"] == "tail":
552:                     selected_indices = selected_indices[-max_len:]
553:                 else:
554:                     raise ValueError(
555:                         f"Unknown truncate_mode {spec['truncate_mode']!r} "
556:                         f"for profile sequence group '{group_name}'."
557:                     )
558:                 actual_lens[row_idx] = len(selected_indices)
559: 
560:                 for field in spec["fields"]:
561:                     full_seq = profile.get(field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
562:                     if len(full_seq) < end:
563:                         raise ValueError(
564:                             f"Profile group '{group_name}' field '{field}' has "
565:                             f"length {len(full_seq)} for cursor {end} (user={int(uid)})."
566:                         )
567:                     sliced_fields[field][row_idx] = full_seq[selected_indices]
568: 
569:             batch_dict.update(sliced_fields)
570:             batch_dict[seq_len_col] = actual_lens
571: 
572:         return batch_dict
573: 
574: 
575: class ProfileParquetStreamDataset(IterableDataset):
576:     def __init__(
577:         self,
578:         parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
579:         manager: SchemaManager,
580:         profile_root: Union[str, Path],
581:         batch_size: int = 4096,
582:         shuffle: bool = True,
583:         drop_last: bool = True,
584:         shuffle_buffer_size: int = 2000000,
585:         extra_col_names: Optional[Iterable[str]] = None,
586:         extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
587:         group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
588:         profile_store: Optional[ProfileSequenceStore] = None,
589:     ):
590:         super().__init__()
591:         self.manager = manager
592:         self.batch_size = batch_size
593:         self.shuffle = shuffle
594:         self.drop_last = drop_last
595:         self.context = FeatureContext.from_manager(
596:             manager,
597:             extra_col_names=extra_col_names,
598:             extra_col_formatters=extra_col_formatters,
599:         )
600:         self.formatter = TensorFormatter(self.context)
601:         self.profile_store = (
602:             profile_store
603:             if profile_store is not None
604:             else ProfileSequenceStore(
605:                 manager=manager,
606:                 profile_root=profile_root,
607:                 group_dirs=group_dirs,
608:             )
609:         )
610: 
611:         parquet_paths = _resolve_parquet_paths(parquet_path)
612:         available_cols = _parquet_top_level_columns(parquet_paths[0])
613:         read_cols = [col for col in self.context.output_col_names if col in available_cols]
614:         for spec in self.profile_store.groups.values():
615:             required_cols = [spec["owner_col"], spec["seq_len_col"]]
616:             if spec["history_time_filter"] != "cursor_order":
617:                 required_cols.append(spec["anchor_col"])
618:             for required_col in required_cols:
619:                 if required_col in available_cols and required_col not in read_cols:
620:                     read_cols.append(required_col)
621: 
622:         self.scanner = DataScanner(parquet_path, read_cols=read_cols)
623:         self.buffer = ShuffleBuffer(
624:             capacity=shuffle_buffer_size if shuffle else batch_size,
625:             batch_size=batch_size,
626:             shuffle=shuffle,
627:             drop_last=drop_last,
628:             selected_col_names=None,
629:         )
630:         self.scan_batch_size = batch_size * 4 if shuffle else batch_size
631:         self.parquet_path = parquet_path
632:         self.parquet_paths = parquet_paths
633:         self.profile_root = Path(profile_root)
634:         self.output_col_names = list(self.context.output_col_names)
635:         self.valid_col_names = list(self.context.output_col_names)
636:         self.read_col_names = list(read_cols)
637:         self.extra_col_names = list(extra_col_names or [])
638:         self.extra_col_formatters = dict(extra_col_formatters or {})
639:         self.trigger_size = self.buffer.capacity
640:         self.col_formatters = self.formatter.col_formatters
641: 
642:     def _complete_batch(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
643:         if not batch_dict:
644:             return batch_dict
645:         batch_size = len(next(iter(batch_dict.values())))
646:         self.profile_store.inject(batch_dict)
647:         for col in self.context.output_col_names:
648:             if col in batch_dict:
649:                 continue
650:             formatter = self.formatter.col_formatters.get(col, self.formatter.fallback_formatter)
651:             batch_dict[col] = _default_missing_column(col, formatter, batch_size)
652:         return {col: batch_dict[col] for col in self.context.output_col_names if col in batch_dict}
653: 
654:     def __iter__(self):
655:         worker_info = torch.utils.data.get_worker_info()
656:         worker_id = worker_info.id if worker_info is not None else 0
657:         num_workers = worker_info.num_workers if worker_info is not None else 1
658:         self.buffer.reset()
659: 
660:         for raw_table in self.scanner.iter_batches(
661:             batch_size=self.scan_batch_size,
662:             worker_id=worker_id,
663:             num_workers=num_workers,
664:         ):
665:             self.buffer.push(raw_table)
666:             for batch_dict in self.buffer.yield_batches():
667:                 yield self.formatter.format(self._complete_batch(batch_dict))
668: 
669:         for batch_dict in self.buffer.flush():
670:             yield self.formatter.format(self._complete_batch(batch_dict))
671: 
672: 
673: class RawParquetStreamDataset(PipelineStreamDataset):
674:     def __init__(
675:         self,
676:         parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
677:         manager: SchemaManager,
678:         batch_size: int = 4096,
679:         shuffle: bool = True,
680:         drop_last: bool = True,
681:         shuffle_buffer_size: int = 2000000,
682:         extra_col_names: Optional[Iterable[str]] = None,
683:         extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
684:         transform_buffer_size: Optional[int] = None,
685:         raw_filter_expr: Optional[pl.Expr] = None,
686:         raw_preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
687:     ):
688:         preprocess_exprs = list(raw_preprocess_exprs or [])
689:         scanner_filter_expr = raw_filter_expr if not preprocess_exprs else None
690:         transformer_filter_expr = None if scanner_filter_expr is not None else raw_filter_expr
691:         context = FeatureContext.from_raw_manager(
692:             manager,
693:             extra_col_names=extra_col_names,
694:             extra_col_formatters=extra_col_formatters,
695:         )
696:         scanner = DataScanner(
697:             parquet_path,
698:             read_cols=context.read_col_names,
699:             filter_expr=scanner_filter_expr,
700:         )
701:         formatter = TensorFormatter(context)
702:         transformer = DataTransformer(
703:             manager=manager,
704:             preprocess_exprs=preprocess_exprs,
705:             filter_expr=transformer_filter_expr,
706:             output_col_names=context.output_col_names,
707:         )
708:         capacity = transform_buffer_size
709:         if capacity is None:
710:             capped_rows = batch_size * (8 if shuffle else 1)
711:             capacity = max(batch_size, min(shuffle_buffer_size, capped_rows))
712:         buffer = ShuffleBuffer(
713:             capacity=capacity,
714:             batch_size=batch_size,
715:             shuffle=shuffle,
716:             drop_last=drop_last,
717:             selected_col_names=context.output_col_names,
718:         )
719:         super().__init__(
720:             scanner=scanner,
721:             formatter=formatter,
722:             transformer=transformer,
723:             buffer=buffer,
724:             scan_batch_size=batch_size * 2 if shuffle else batch_size,
725:         )
726: 
727:         self.parquet_path = parquet_path
728:         self.parquet_paths = list(scanner.parquet_paths or [])
729:         self.manager = manager
730:         self.batch_size = batch_size
731:         self.shuffle = shuffle
732:         self.drop_last = drop_last
733:         self.trigger_size = buffer.capacity
734:         self.context = context
735:         self.output_col_names = list(context.output_col_names)
736:         self.valid_col_names = list(context.output_col_names)
737:         self.read_col_names = list(context.read_col_names)
738:         self.extra_col_names = list(extra_col_names or [])
739:         self.extra_col_formatters = dict(extra_col_formatters or {})
740:         self.raw_filter_expr = raw_filter_expr
741:         self.raw_preprocess_exprs = preprocess_exprs
742:         self.transform_buffer_size = buffer.capacity
743:         self.col_formatters = formatter.col_formatters
````

## File: src/rosen/data/padding.py
````python
  1: from __future__ import annotations
  2: 
  3: from abc import ABC, abstractmethod
  4: from dataclasses import dataclass, field
  5: from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional
  6: 
  7: import numpy as np
  8: import torch
  9: 
 10: from rosen.core.interaction import Interaction
 11: 
 12: if TYPE_CHECKING:
 13:     from rosen.emb import SchemaManager
 14: 
 15: 
 16: PaddingSide = Literal["left", "right"]
 17: 
 18: 
 19: def _is_missing_sequence(value) -> bool:
 20:     return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
 21: 
 22: 
 23: def _pad_list(
 24:     sequences,
 25:     max_len: int,
 26:     padding_side: PaddingSide = "right",
 27:     pad_val=0,
 28:     dtype=np.int64,
 29: ) -> np.ndarray:
 30:     padded = np.full((len(sequences), max_len), pad_val, dtype=dtype)
 31:     for i, seq in enumerate(sequences):
 32:         if _is_missing_sequence(seq) or len(seq) == 0:
 33:             continue
 34: 
 35:         seq_list = list(seq)
 36:         valid_len = min(len(seq_list), max_len)
 37:         trunc_seq = seq_list[-valid_len:]
 38:         if padding_side == "right":
 39:             padded[i, :valid_len] = trunc_seq
 40:         elif padding_side == "left":
 41:             padded[i, max_len - valid_len:] = trunc_seq
 42:         else:
 43:             raise ValueError(f"Unknown padding_side: {padding_side}")
 44: 
 45:     return padded
 46: 
 47: 
 48: def _pad_nested_list(
 49:     sequences,
 50:     max_seq_len: int,
 51:     max_tag_len: int,
 52:     padding_side: PaddingSide = "right",
 53:     pad_val: int = 0,
 54: ) -> np.ndarray:
 55:     padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
 56:     for i, seq in enumerate(sequences):
 57:         if _is_missing_sequence(seq) or len(seq) == 0:
 58:             continue
 59: 
 60:         seq_list = list(seq)
 61:         valid_seq_len = min(len(seq_list), max_seq_len)
 62:         trunc_seq = seq_list[-valid_seq_len:]
 63:         start_idx = 0 if padding_side == "right" else max_seq_len - valid_seq_len
 64: 
 65:         for j in range(valid_seq_len):
 66:             tags = trunc_seq[j]
 67:             if _is_missing_sequence(tags) or len(tags) == 0:
 68:                 continue
 69: 
 70:             tags_list = list(tags)
 71:             valid_tag_len = min(len(tags_list), max_tag_len)
 72:             trunc_tags = tags_list[:valid_tag_len]
 73:             padded[i, start_idx + j, :valid_tag_len] = trunc_tags
 74: 
 75:     return padded
 76: 
 77: 
 78: class ColumnFormatter(ABC):
 79:     @abstractmethod
 80:     def format(self, data) -> torch.Tensor:
 81:         pass
 82: 
 83: 
 84: @dataclass(frozen=True)
 85: class DenseFormatter(ColumnFormatter):
 86:     def format(self, data) -> torch.Tensor:
 87:         return torch.tensor(data, dtype=torch.float32)
 88: 
 89: 
 90: @dataclass(frozen=True)
 91: class VectorDenseFormatter(ColumnFormatter):
 92:     dim: int
 93:     zero_fill: bool = True
 94: 
 95:     def format(self, data) -> torch.Tensor:
 96:         arr = np.zeros((len(data), self.dim), dtype=np.float32)
 97:         for i, row in enumerate(data):
 98:             if _is_missing_sequence(row) or len(row) == 0:
 99:                 if not self.zero_fill:
100:                     raise ValueError(
101:                         f"Expected non-empty vector dim={self.dim}, but got an empty value at row {i}."
102:                     )
103:                 continue
104: 
105:             row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
106:             use_len = min(row_arr.shape[0], self.dim)
107:             arr[i, :use_len] = row_arr[:use_len]
108: 
109:         return torch.tensor(arr, dtype=torch.float32)
110: 
111: 
112: @dataclass(frozen=True)
113: class IntFormatter(ColumnFormatter):
114:     def format(self, data) -> torch.Tensor:
115:         if data.dtype == np.uint32:
116:             data = data.astype(np.int64)
117:         return torch.tensor(data, dtype=torch.long)
118: 
119: 
120: @dataclass(frozen=True)
121: class FallbackFormatter(ColumnFormatter):
122:     def format(self, data) -> torch.Tensor:
123:         if data.dtype in (np.float32, np.float64):
124:             return torch.tensor(data, dtype=torch.float32)
125:         if data.dtype == np.uint32:
126:             return torch.tensor(data.astype(np.int64), dtype=torch.long)
127:         return torch.tensor(data)
128: 
129: 
130: @dataclass(frozen=True)
131: class PaddedIntSequenceFormatter(ColumnFormatter):
132:     max_len: int
133:     padding_side: PaddingSide = "right"
134:     pad_val: int = 0
135: 
136:     def format(self, data) -> torch.Tensor:
137:         arr = _pad_list(
138:             data,
139:             max_len=self.max_len,
140:             padding_side=self.padding_side,
141:             pad_val=self.pad_val,
142:             dtype=np.int64,
143:         )
144:         return torch.tensor(arr, dtype=torch.long)
145: 
146: 
147: @dataclass(frozen=True)
148: class PaddedFloatSequenceFormatter(ColumnFormatter):
149:     max_len: int
150:     padding_side: PaddingSide = "right"
151:     pad_val: float = 0.0
152: 
153:     def format(self, data) -> torch.Tensor:
154:         arr = _pad_list(
155:             data,
156:             max_len=self.max_len,
157:             padding_side=self.padding_side,
158:             pad_val=self.pad_val,
159:             dtype=np.float32,
160:         )
161:         return torch.tensor(arr, dtype=torch.float32)
162: 
163: 
164: @dataclass(frozen=True)
165: class PaddedNestedSequenceFormatter(ColumnFormatter):
166:     max_seq_len: int
167:     max_tag_len: int
168:     padding_side: PaddingSide = "right"
169:     pad_val: int = 0
170: 
171:     def format(self, data) -> torch.Tensor:
172:         arr = _pad_nested_list(
173:             data,
174:             max_seq_len=self.max_seq_len,
175:             max_tag_len=self.max_tag_len,
176:             padding_side=self.padding_side,
177:             pad_val=self.pad_val,
178:         )
179:         return torch.tensor(arr, dtype=torch.long)
180: 
181: 
182: DEFAULT_FALLBACK_FORMATTER = FallbackFormatter()
183: 
184: 
185: def _dedupe_names(names: Iterable[str]) -> list[str]:
186:     return list(dict.fromkeys(name for name in names if name))
187: 
188: 
189: def _resolve_raw_read_col_names(
190:     manager: "SchemaManager",
191:     extra_col_names: Iterable[str],
192:     extra_col_formatters: Dict[str, ColumnFormatter],
193: ) -> list[str]:
194:     raw_col_names: list[str] = []
195:     for setting in manager.settings:
196:         raw_col_names.extend(setting.get_raw_field_names())
197: 
198:     raw_col_names.extend(manager.label_fields)
199:     raw_col_names.extend(manager.domain_fields)
200:     if manager.time_field:
201:         raw_col_names.append(manager.time_field)
202: 
203:     raw_col_names.extend(extra_col_names)
204:     raw_col_names.extend(extra_col_formatters.keys())
205:     return _dedupe_names(raw_col_names)
206: 
207: 
208: @dataclass(frozen=True)
209: class FeatureContext:
210:     manager: "SchemaManager"
211:     read_col_names: tuple[str, ...]
212:     output_col_names: tuple[str, ...]
213:     extra_col_formatters: Dict[str, ColumnFormatter] = field(default_factory=dict)
214: 
215:     @classmethod
216:     def from_manager(
217:         cls,
218:         manager: "SchemaManager",
219:         extra_col_names: Optional[Iterable[str]] = None,
220:         extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
221:     ) -> "FeatureContext":
222:         formatter_map = dict(extra_col_formatters or {})
223:         extra_names = list(extra_col_names or [])
224:         output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
225:         return cls(
226:             manager=manager,
227:             read_col_names=tuple(output_cols),
228:             output_col_names=tuple(output_cols),
229:             extra_col_formatters=formatter_map,
230:         )
231: 
232:     @classmethod
233:     def from_raw_manager(
234:         cls,
235:         manager: "SchemaManager",
236:         extra_col_names: Optional[Iterable[str]] = None,
237:         extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
238:     ) -> "FeatureContext":
239:         formatter_map = dict(extra_col_formatters or {})
240:         extra_names = list(extra_col_names or [])
241:         output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
242:         read_cols = _resolve_raw_read_col_names(manager, extra_names, formatter_map)
243:         return cls(
244:             manager=manager,
245:             read_col_names=tuple(read_cols),
246:             output_col_names=tuple(output_cols),
247:             extra_col_formatters=formatter_map,
248:         )
249: 
250: 
251: @dataclass
252: class TensorFormatter:
253:     context: FeatureContext
254:     fallback_formatter: ColumnFormatter = DEFAULT_FALLBACK_FORMATTER
255:     col_formatters: Dict[str, ColumnFormatter] = field(init=False)
256: 
257:     def __post_init__(self):
258:         self.col_formatters = self._compile_formatters()
259: 
260:     def _compile_formatters(self) -> Dict[str, ColumnFormatter]:
261:         formatters: Dict[str, ColumnFormatter] = {}
262:         for setting in self.context.manager.settings:
263:             formatters.update(setting.get_formatters())
264: 
265:         for ctx_col in (
266:             *self.context.manager.label_fields,
267:             *self.context.manager.domain_fields,
268:             self.context.manager.time_field,
269:         ):
270:             if ctx_col and ctx_col not in formatters:
271:                 formatters[ctx_col] = self.fallback_formatter
272: 
273:         formatters.update(self.context.extra_col_formatters)
274:         return formatters
275: 
276:     def format_tensors(self, batch_dict) -> Dict[str, torch.Tensor]:
277:         tensor_dict = {}
278:         for col, data in batch_dict.items():
279:             formatter = self.col_formatters.get(col, self.fallback_formatter)
280:             try:
281:                 tensor_dict[col] = formatter.format(data)
282:             except Exception as exc:
283:                 raise RuntimeError(
284:                     f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的 formatter 为: {formatter}。错误信息: {exc}"
285:                 ) from exc
286:         return tensor_dict
287: 
288:     def format(self, batch_dict) -> Interaction:
289:         return Interaction(self.format_tensors(batch_dict))
````

## File: src/rosen/datasets/base.py
````python
 1: import os
 2: from pathlib import Path
 3: 
 4: 
 5: class DatasetBase:
 6:     """Base class for adapters backed by an external source-data root.
 7: 
 8:     The root may contain original files or lossless, experiment-agnostic canonical
 9:     forms such as typed Parquet. It is read-only during experiments. Schemas,
10:     filtered splits, caches, and other experiment-derived assets belong under the
11:     repository ``workspace`` instead.
12:     """
13: 
14:     SYSTEM_DATA_DIR = Path(
15:         os.path.expandvars(
16:             os.environ.get(
17:                 "ROSEN_DATASOURCE_ROOT",
18:                 os.environ.get("ROSEN_DATA_DIR", "/Users/lxogri/Developer/pyprojects/time-budget/datasource"),
19:             )
20:         )
21:     ).expanduser()
22: 
23:     def __init__(self):
24:         pass
````

## File: src/rosen/datasets/kuairand.py
````python
 1: import polars as pl
 2: from rosen.datasets.base import DatasetBase
 3: import pprint
 4: 
 5: class KuaiRandDataset(DatasetBase):
 6:     BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "KuaiRand"
 7:     DATA_KUAI_1K = BASE_DIR / "KuaiRand-1K" / "data"
 8:     STD_LOG_FORMER_DATA = DATA_KUAI_1K / "log_standard_4_08_to_4_21_1k.csv"
 9:     STD_LOG_FORMER_DATA_P1 = STD_LOG_FORMER_DATA
10:     STD_LOG_FORMER_DATA_P2 = DATA_KUAI_1K / "log_standard_4_22_to_5_08_1k.csv"
11:     RAND_LOG_FORMER_DATA = DATA_KUAI_1K / "log_random_4_22_to_5_08_1k.csv"
12:     USER_FEATURES = DATA_KUAI_1K / "user_features_1k.csv"
13:     VIDEO_FEATURES = DATA_KUAI_1K / "video_features_basic_1k.csv"
14: 
15: # ==========================================
16: # 执行与测试示例
17: # ==========================================
18: if __name__ == "__main__":
19:     # 为了防止路径不存在报错，这里用 try-except 包裹文件读取逻辑
20:     # 请确保你的实际路径中存在这些 CSV 文件
21:     from rosen.datasets.overview import get_general_info, get_head_info, get_group_stats
22:     try:
23:         # 读取三个文件 (这里假设存在且格式标准)
24:         df_log = pl.read_csv(KuaiRandDataset.STD_LOG_FORMER_DATA_P2)
25:         df_user = pl.read_csv(KuaiRandDataset.USER_FEATURES)
26:         df_video = pl.read_csv(KuaiRandDataset.VIDEO_FEATURES)
27: 
28:         datasets = {
29:             "Log Standard": df_log,
30:             "User Features": df_user,
31:             "Video Features": df_video
32:         }
33: 
34:         # 1. 测试通用信息打印
35:         print("=== 一、通用信息统计 ===")
36:         for name, df in datasets.items():
37:             info = get_general_info(df)
38:             print(f"\n[{name}] 数据集统计:")
39:             # 使用 pprint 打印字典，结构清晰且不会省略数据
40:             pprint.pprint(info, depth=4, compact=False)
41: 
42:         # 2. 测试分组统计函数 (以 df_log 中的某个假设列为例，例如 'is_click')
43:         print("\n=== 二、分组统计 ===")
44:         # 这里假设 df_log 有一个 'is_click' 列，你可以替换为你实际想看的列名
45:         test_col = "tab"
46:         group_stats = get_group_stats(df_log, test_col)
47:         print(f"对 '{test_col}' 的分组统计结果:")
48:         pprint.pprint(group_stats)
49: 
50:         # 3. 测试 Head(3) 视图
51:         print("\n=== 三、Head(3) 完整数据及类型视图 ===")
52:         head_view = get_head_info(df_user)  # 以 User Features 为例
53:         print("[User Features] 前三行完整数据:")
54:         pprint.pprint(head_view)
55: 
56:     except FileNotFoundError as e:
57:         print(f"文件未找到，请检查路径: {e}")
58:     except Exception as e:
59:         print(f"发生错误: {e}")
````

## File: src/rosen/emb/schema/__init__.py
````python
 1: from .base import EmbSetting, EmbType, ProfileSeqGroupConfig, SeqGroupConfig
 2: from .categorical import BaseCategoricalSetting, MultiSparseSetting, QuantileEmbSetting, SparseEmbSetting
 3: from .numerical import BaseNumericalSetting, MinMaxDenseSetting, VectorDenseSetting
 4: from .sequence import SequenceSetting
 5: from .utils import NULL_FALLBACK, clear_seq_expr, explode_expr, seq_length_expr
 6: 
 7: __all__ = [
 8:     "NULL_FALLBACK",
 9:     "EmbSetting",
10:     "EmbType",
11:     "SeqGroupConfig",
12:     "ProfileSeqGroupConfig",
13:     "BaseCategoricalSetting",
14:     "BaseNumericalSetting",
15:     "SparseEmbSetting",
16:     "MultiSparseSetting",
17:     "SequenceSetting",
18:     "QuantileEmbSetting",
19:     "MinMaxDenseSetting",
20:     "VectorDenseSetting",
21:     "clear_seq_expr",
22:     "explode_expr",
23:     "seq_length_expr",
24: ]
````

## File: src/rosen/emb/schema/base.py
````python
  1: from abc import ABC, abstractmethod
  2: from dataclasses import dataclass
  3: from enum import Enum
  4: from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional
  5: 
  6: import polars as pl
  7: import torch
  8: from torch import nn
  9: 
 10: from rosen.core.enum_type import FeatureSource
 11: 
 12: if TYPE_CHECKING:
 13:     from rosen.data.padding import ColumnFormatter
 14: 
 15: 
 16: class EmbType(Enum):
 17:     UNKNOWN = "none"
 18:     SPARSE = "sparse"
 19:     MULTI_SPARSE = "multi_sparse"
 20:     QUANTILE = "quantile"
 21:     SPARSE_SEQ = "sparse_seq"
 22:     SPARSE_SET = "sparse_set"
 23:     DENSE = "dense"
 24:     VECTOR_DENSE = "vector_dense"
 25:     DENSE_SEQ = "dense_seq"
 26:     SEQUENCE = "sequence"
 27: 
 28: 
 29: @dataclass
 30: class SeqGroupConfig:
 31:     group_name: str
 32:     seq_len_field_name: str
 33:     max_len: int
 34:     padding_side: Literal["left", "right"] = "right"
 35:     time_field_name: Optional[str] = None
 36: 
 37: 
 38: @dataclass
 39: class ProfileSeqGroupConfig(SeqGroupConfig):
 40:     owner_field_name: str = "user_id"
 41:     id_field_name: Optional[str] = None
 42:     anchor_field_name: Optional[str] = None
 43:     profile_name: Optional[str] = None
 44:     history_time_filter: Literal[
 45:         "cursor_order",
 46:         "strict_time",
 47:         "drop_same_ts",
 48:         "same_ts_only",
 49:     ] = "cursor_order"
 50:     backend: Literal["profile"] = "profile"
 51: 
 52: 
 53: class EmbSetting(ABC):
 54:     emb_type = EmbType.UNKNOWN
 55: 
 56:     def __init__(
 57:         self,
 58:         field_name: str,
 59:         embedding_dim: int,
 60:         source: FeatureSource = FeatureSource.UNKNOWN,
 61:         padding_zero: bool = True,
 62:         use_oov: bool = True,
 63:     ):
 64:         self.field_name = field_name
 65:         self.embedding_dim = embedding_dim
 66:         self.source = source
 67:         self.is_fitted = False
 68: 
 69:         self.padding_zero = padding_zero
 70:         self.use_oov = use_oov
 71:         self.vocab: Dict[str, int] = {}
 72:         self.oov_idx: int = -1
 73: 
 74:     @property
 75:     def vocab_size(self) -> int:
 76:         return len(self.vocab)
 77: 
 78:     @property
 79:     def num_embeddings(self) -> int:
 80:         if not self.is_fitted:
 81:             return -1
 82:         return self.vocab_size + int(self.padding_zero) + int(self.use_oov)
 83: 
 84:     @property
 85:     def serialized_type_name(self) -> str:
 86:         return self.emb_type.name
 87: 
 88:     @property
 89:     def compatible_type_names(self) -> set[str]:
 90:         return {self.serialized_type_name}
 91: 
 92:     @property
 93:     def requires_embedding_module(self) -> bool:
 94:         return self.num_embeddings > 0
 95: 
 96:     @property
 97:     def embedding_field_name(self) -> str:
 98:         return self.field_name
 99: 
100:     @property
101:     def is_sequence_setting(self) -> bool:
102:         return False
103: 
104:     def get_output_field_names(self) -> List[str]:
105:         return [self.field_name]
106: 
107:     def get_raw_field_names(self) -> List[str]:
108:         return [self.field_name]
109: 
110:     @abstractmethod
111:     def get_fit_exprs(self) -> List[pl.Expr]:
112:         pass
113: 
114:     @abstractmethod
115:     def parse_fit_result(self, result_df: pl.DataFrame):
116:         pass
117: 
118:     def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
119:         return self.parse_fit_result(result_df)
120: 
121:     @abstractmethod
122:     def get_transform_expr(self):
123:         pass
124: 
125:     def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
126:         raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element fitting.")
127: 
128:     def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
129:         raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element transform.")
130: 
131:     @abstractmethod
132:     def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
133:         pass
134: 
135:     def get_sequence_formatters(
136:         self,
137:         max_len: int,
138:         padding_side: Literal["left", "right"],
139:     ) -> Dict[str, "ColumnFormatter"]:
140:         from rosen.data.padding import (
141:             DenseFormatter,
142:             IntFormatter,
143:             PaddedFloatSequenceFormatter,
144:             PaddedIntSequenceFormatter,
145:         )
146: 
147:         base_formatters = self.get_formatters()
148:         fmt = base_formatters[self.field_name]
149:         if isinstance(fmt, DenseFormatter):
150:             return {self.field_name: PaddedFloatSequenceFormatter(max_len=max_len, padding_side=padding_side)}
151:         if isinstance(fmt, IntFormatter):
152:             return {self.field_name: PaddedIntSequenceFormatter(max_len=max_len, padding_side=padding_side)}
153:         raise NotImplementedError(
154:             f"Formatter {fmt.__class__.__name__} cannot be promoted to a sequence automatically."
155:         )
156: 
157:     def to_dict(self) -> Dict[str, Any]:
158:         return {
159:             "type": self.serialized_type_name,
160:             "field_name": self.field_name,
161:             "embedding_size": self.embedding_dim,
162:             "num_embeddings": self.num_embeddings,
163:             "vocab_size": self.vocab_size,
164:             "feature_source": self.source.name,
165:             "is_fitted": self.is_fitted,
166:             "padding_zero": self.padding_zero,
167:             "use_oov": self.use_oov,
168:             "vocab": self.vocab,
169:             "oov_idx": self.oov_idx,
170:         }
171: 
172:     def load_state(self, state_dict: Dict[str, Any]):
173:         self.is_fitted = state_dict.get("is_fitted", True)
174:         self.vocab = state_dict.get("vocab", {})
175:         self.use_oov = state_dict.get("use_oov", self.use_oov)
176:         self.padding_zero = state_dict.get("padding_zero", self.padding_zero)
177:         self.oov_idx = state_dict.get("oov_idx", self.oov_idx)
178: 
179:     @classmethod
180:     @abstractmethod
181:     def from_dict(cls, data: Dict[str, Any]) -> "EmbSetting":
182:         pass
183: 
184:     @abstractmethod
185:     def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
186:         pass
````

## File: src/rosen/emb/schema/categorical.py
````python
  1: from typing import Any, Dict, List, Optional
  2: 
  3: import numpy as np
  4: import polars as pl
  5: import torch
  6: from torch import nn
  7: 
  8: from rosen.core.enum_type import FeatureSource
  9: 
 10: from .base import EmbSetting, EmbType
 11: from .utils import NULL_FALLBACK, map_list_to_indices, mean_pooling
 12: 
 13: 
 14: class BaseCategoricalSetting(EmbSetting):
 15:     """离散特征基类：统一管理词表与 OOV 逻辑。"""
 16:     NULL_TOKEN = "__NULL__"
 17: 
 18:     def __init__(
 19:         self,
 20:         field_name: str,
 21:         embedding_dim: int,
 22:         source: FeatureSource,
 23:         padding_zero: bool = True,
 24:         use_null: bool = True,
 25:         use_oov: bool = True,
 26:         min_freq: int = 1,
 27:     ):
 28:         super().__init__(field_name, embedding_dim, source, padding_zero, use_oov)
 29:         if use_null and not padding_zero:
 30:             raise ValueError("use_null=True requires padding_zero=True for sparse categorical features.")
 31:         self.min_freq = min_freq
 32:         self.use_null = use_null
 33:         self.vocab = {}
 34:         self.null_idx = -1
 35:         self.oov_idx = -1
 36: 
 37:     def _build_vocab_indices(self, valid_vals: List[str]):
 38:         start_idx = 1 if self.padding_zero else 0
 39:         self.vocab = {str(val): idx + start_idx for idx, val in enumerate(valid_vals)}
 40:         next_idx = len(self.vocab) + start_idx
 41:         if self.use_null:
 42:             self.null_idx = next_idx
 43:             next_idx += 1
 44:         else:
 45:             self.null_idx = 0 if self.padding_zero else -1
 46:         if self.use_oov:
 47:             self.oov_idx = next_idx
 48:         else:
 49:             self.oov_idx = 0 if self.padding_zero else -1
 50:         self.is_fitted = True
 51: 
 52:     @property
 53:     def num_embeddings(self) -> int:
 54:         if not self.is_fitted:
 55:             return -1
 56:         return (
 57:             self.vocab_size
 58:             + int(self.padding_zero)
 59:             + int(self.use_null)
 60:             + int(self.use_oov)
 61:         )
 62: 
 63: 
 64: class SparseEmbSetting(BaseCategoricalSetting):
 65:     emb_type = EmbType.SPARSE
 66: 
 67:     def __init__(
 68:         self,
 69:         field_name: str,
 70:         source: FeatureSource,
 71:         embedding_dim: int = 16,
 72:         padding_zero: bool = True,
 73:         use_null: bool = True,
 74:         fill_null: str = "null",
 75:         min_freq: int = 1,
 76:         use_oov: bool = True,
 77:     ):
 78:         super().__init__(
 79:             field_name,
 80:             embedding_dim,
 81:             source,
 82:             padding_zero,
 83:             use_null=use_null,
 84:             use_oov=use_oov,
 85:             min_freq=min_freq,
 86:         )
 87:         if fill_null not in {"null", "oov", "zero", None}:
 88:             raise ValueError("fill_null must be one of {'null', 'oov', 'zero', None}")
 89:         self.fill_null = fill_null
 90: 
 91:     def get_fit_exprs(self) -> List[pl.Expr]:
 92:         return [
 93:             pl.col(self.field_name)
 94:             .cast(pl.Utf8)
 95:             .drop_nulls()
 96:             .value_counts()
 97:             .implode()
 98:             .alias(self.field_name)
 99:         ]
100: 
101:     def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
102:         exploded_raw = seq_expr.explode().cast(pl.Utf8)
103:         exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
104:         return [exploded.value_counts().implode().alias(alias)]
105: 
106:     def parse_fit_result(self, result_df: pl.DataFrame):
107:         self.parse_element_fit_result(result_df, alias=self.field_name)
108: 
109:     def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
110:         rows = result_df.get_column(alias).to_list()[0]
111:         valid_vals = sorted(
112:             [
113:                 row[alias]
114:                 for row in rows
115:                 if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
116:             ]
117:         )
118:         self._build_vocab_indices(valid_vals)
119: 
120:     def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
121:         expr = expr if expr is not None else pl.element()
122:         replace_kwargs = {"default": pl.lit(self.oov_idx, dtype=pl.UInt32)} if self.oov_idx >= 0 else {}
123:         mapped_expr = expr.cast(pl.Utf8).replace_strict(self.vocab, **replace_kwargs)
124:         if not self.use_null:
125:             return mapped_expr.cast(pl.UInt32)
126: 
127:         if self.fill_null == "zero":
128:             null_expr = pl.lit(0, dtype=pl.UInt32)
129:         elif self.fill_null == "oov":
130:             null_expr = pl.lit(self.oov_idx, dtype=pl.UInt32)
131:         else:
132:             null_expr = pl.lit(self.null_idx, dtype=pl.UInt32)
133:         return pl.when(expr.is_null()).then(null_expr).otherwise(mapped_expr).cast(pl.UInt32)
134: 
135:     def get_transform_expr(self) -> pl.Expr:
136:         return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)
137: 
138:     def get_formatters(self):
139:         from rosen.data.padding import IntFormatter
140: 
141:         return {self.field_name: IntFormatter()}
142: 
143:     def to_dict(self) -> Dict[str, Any]:
144:         data = super().to_dict()
145:         data["min_freq"] = self.min_freq
146:         data["use_null"] = self.use_null
147:         data["null_idx"] = self.null_idx
148:         data["fill_null"] = self.fill_null
149:         return data
150: 
151:     def load_state(self, state_dict: Dict[str, Any]):
152:         super().load_state(state_dict)
153:         self.min_freq = state_dict.get("min_freq", 1)
154:         self.use_null = state_dict.get("use_null", self.use_null)
155:         self.null_idx = state_dict.get("null_idx", self.null_idx)
156:         self.fill_null = state_dict.get("fill_null", self.fill_null)
157: 
158:     @classmethod
159:     def from_dict(cls, data: Dict[str, Any]) -> "SparseEmbSetting":
160:         obj = cls(
161:             field_name=data["field_name"],
162:             source=FeatureSource[data["feature_source"]],
163:             embedding_dim=data["embedding_size"],
164:             padding_zero=data.get("padding_zero", True),
165:             use_null=data.get("use_null", True),
166:             fill_null=data.get("fill_null", "null"),
167:             min_freq=data.get("min_freq", 1),
168:             use_oov=data.get("use_oov", True),
169:         )
170:         obj.load_state(data)
171:         return obj
172: 
173:     def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
174:         return emb_modules[self.embedding_field_name](interaction[self.field_name])
175: 
176: 
177: class QuantileEmbSetting(EmbSetting):
178:     emb_type = EmbType.QUANTILE
179: 
180:     def __init__(
181:         self,
182:         field_name: str,
183:         source: FeatureSource,
184:         bucket_count: int = 10,
185:         embedding_dim: int = 16,
186:         boundaries: Optional[List[float]] = None,
187:     ):
188:         super().__init__(field_name, embedding_dim, source, padding_zero=True, use_oov=False)
189:         self.bucket_count = bucket_count
190:         self.boundaries = boundaries if boundaries is not None else []
191:         if self.boundaries:
192:             self.is_fitted = True
193: 
194:     @property
195:     def vocab_size(self) -> int:
196:         return len(self.boundaries) + 1 if self.boundaries else 0
197: 
198:     @property
199:     def num_embeddings(self) -> int:
200:         if not self.is_fitted:
201:             return -1
202:         return self.vocab_size + 1
203: 
204:     def get_fit_exprs(self) -> List[pl.Expr]:
205:         q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
206:         return [
207:             pl.col(self.field_name).drop_nulls().cast(pl.Float64).quantile(q).alias(f"{self.field_name}_q_{idx}")
208:             for idx, q in enumerate(q_list)
209:         ]
210: 
211:     def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
212:         q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
213:         exploded = seq_expr.explode().drop_nulls().cast(pl.Float64)
214:         return [
215:             exploded.quantile(q).alias(f"{alias}_q_{idx}")
216:             for idx, q in enumerate(q_list)
217:         ]
218: 
219:     def parse_fit_result(self, result_df: pl.DataFrame):
220:         self.parse_element_fit_result(result_df, alias=self.field_name)
221: 
222:     def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
223:         boundaries = []
224:         for idx in range(self.bucket_count - 1):
225:             value = result_df.get_column(f"{alias}_q_{idx}")[0]
226:             if value is not None and not np.isnan(value):
227:                 boundaries.append(value)
228:         self.boundaries = sorted(set(boundaries))
229:         self.is_fitted = True
230: 
231:     def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
232:         expr = expr if expr is not None else pl.element()
233:         if not self.is_fitted or not self.boundaries:
234:             return pl.lit(0, dtype=pl.UInt32)
235: 
236:         bucket_count = len(self.boundaries) + 1
237:         labels = [str(idx) for idx in range(1, bucket_count + 1)]
238:         return (
239:             expr
240:             .cut(breaks=self.boundaries, labels=labels)
241:             .cast(pl.String)
242:             .cast(pl.UInt32)
243:             .fill_null(0)
244:         )
245: 
246:     def get_transform_expr(self) -> pl.Expr:
247:         return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)
248: 
249:     def get_formatters(self):
250:         from rosen.data.padding import IntFormatter
251: 
252:         return {self.field_name: IntFormatter()}
253: 
254:     def to_dict(self) -> Dict[str, Any]:
255:         data = super().to_dict()
256:         data["bucket_count"] = self.bucket_count
257:         data["boundaries"] = self.boundaries
258:         return data
259: 
260:     def load_state(self, state_dict: Dict[str, Any]):
261:         super().load_state(state_dict)
262:         self.bucket_count = state_dict.get("bucket_count", self.bucket_count)
263:         self.boundaries = state_dict.get("boundaries", [])
264: 
265:     @classmethod
266:     def from_dict(cls, data: Dict[str, Any]) -> "QuantileEmbSetting":
267:         obj = cls(
268:             field_name=data["field_name"],
269:             source=FeatureSource[data["feature_source"]],
270:             bucket_count=data.get("bucket_count", 10),
271:             embedding_dim=data["embedding_size"],
272:             boundaries=data.get("boundaries", []),
273:         )
274:         obj.is_fitted = data.get("is_fitted", False)
275:         return obj
276: 
277:     def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
278:         return emb_modules[self.embedding_field_name](interaction[self.field_name])
279: 
280: 
281: class MultiSparseSetting(BaseCategoricalSetting):
282:     emb_type = EmbType.MULTI_SPARSE
283: 
284:     def __init__(
285:         self,
286:         field_name: str,
287:         source: FeatureSource,
288:         embedding_dim: int = 16,
289:         max_tag_len: int = 5,
290:         is_string_format: bool = False,
291:         separator: str = ",",
292:         padding_zero: bool = True,
293:         min_freq: int = 1,
294:         use_oov: bool = True,
295:         agg: str = "sum",
296:     ):
297:         super().__init__(
298:             field_name,
299:             embedding_dim,
300:             source,
301:             padding_zero,
302:             use_null=False,
303:             use_oov=use_oov,
304:             min_freq=min_freq,
305:         )
306:         self.max_tag_len = max_tag_len
307:         self.is_string_format = is_string_format
308:         self.separator = separator
309:         self.agg = agg
310: 
311:     @property
312:     def compatible_type_names(self) -> set[str]:
313:         return {self.serialized_type_name, EmbType.SPARSE_SET.name}
314: 
315:     @property
316:     def fill_empty_sequence_with_fallback(self) -> bool:
317:         return False
318: 
319:     def _clean_list_expr(self, expr: pl.Expr, fill_empty_with_fallback: bool = False) -> pl.Expr:
320:         if self.is_string_format:
321:             cleaned = (
322:                 expr.cast(pl.Utf8)
323:                 .fill_null("")
324:                 .str.split(self.separator)
325:                 .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
326:             )
327:         else:
328:             cleaned = (
329:                 expr.cast(pl.List(pl.Utf8))
330:                 .fill_null([])
331:                 .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
332:             )
333: 
334:         if not fill_empty_with_fallback:
335:             return cleaned
336: 
337:         return (
338:             pl.when(cleaned.list.len() > 0)
339:             .then(cleaned)
340:             .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
341:         )
342: 
343:     def _truncate_tag_expr(self, expr: pl.Expr) -> pl.Expr:
344:         return expr.list.head(self.max_tag_len)
345: 
346:     def get_fit_exprs(self) -> List[pl.Expr]:
347:         exploded_raw = self._truncate_tag_expr(self._clean_list_expr(pl.col(self.field_name))).explode().cast(pl.Utf8)
348:         exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
349:         return [exploded.value_counts().implode().alias(self.field_name)]
350: 
351:     def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
352:         nested_expr = seq_expr.list.eval(
353:             self._truncate_tag_expr(self._clean_list_expr(pl.element(), fill_empty_with_fallback=False))
354:         )
355:         exploded_raw = nested_expr.explode().explode().cast(pl.Utf8)
356:         exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
357:         return [exploded.value_counts().implode().alias(alias)]
358: 
359:     def parse_fit_result(self, result_df: pl.DataFrame):
360:         self.parse_element_fit_result(result_df, alias=self.field_name)
361: 
362:     def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
363:         rows = result_df.get_column(alias).to_list()[0]
364:         valid_vals = sorted(
365:             [
366:                 row[alias]
367:                 for row in rows
368:                 if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
369:             ]
370:         )
371:         self._build_vocab_indices(valid_vals)
372: 
373:     def get_transform_expr(self) -> pl.Expr:
374:         expr = self._truncate_tag_expr(
375:             self._clean_list_expr(pl.col(self.field_name), fill_empty_with_fallback=False)
376:         )
377:         return map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)
378: 
379:     def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
380:         expr = expr if expr is not None else pl.element()
381:         cleaned = self._truncate_tag_expr(
382:             self._clean_list_expr(expr, fill_empty_with_fallback=False)
383:         )
384:         return map_list_to_indices(cleaned, self.vocab, self.oov_idx)
385: 
386:     def get_formatters(self):
387:         from rosen.data.padding import PaddedIntSequenceFormatter
388: 
389:         return {self.field_name: PaddedIntSequenceFormatter(max_len=self.max_tag_len)}
390: 
391:     def get_sequence_formatters(self, max_len: int, padding_side: str):
392:         from rosen.data.padding import PaddedNestedSequenceFormatter
393: 
394:         return {
395:             self.field_name: PaddedNestedSequenceFormatter(
396:                 max_seq_len=max_len,
397:                 max_tag_len=self.max_tag_len,
398:                 padding_side=padding_side,
399:             )
400:         }
401: 
402:     def to_dict(self) -> Dict[str, Any]:
403:         data = super().to_dict()
404:         data["min_freq"] = self.min_freq
405:         data["max_tag_len"] = self.max_tag_len
406:         data["is_string_format"] = self.is_string_format
407:         data["separator"] = self.separator
408:         data["agg"] = self.agg
409:         return data
410: 
411:     def load_state(self, state_dict: Dict[str, Any]):
412:         super().load_state(state_dict)
413:         self.min_freq = state_dict.get("min_freq", 1)
414:         self.max_tag_len = state_dict.get("max_tag_len", state_dict.get("max_len", self.max_tag_len))
415:         self.is_string_format = state_dict.get("is_string_format", False)
416:         self.separator = state_dict.get("separator", ",")
417:         self.agg = state_dict.get("agg", "sum")
418: 
419:     @classmethod
420:     def from_dict(cls, data: Dict[str, Any]) -> "MultiSparseSetting":
421:         obj = cls(
422:             field_name=data["field_name"],
423:             source=FeatureSource[data["feature_source"]],
424:             embedding_dim=data["embedding_size"],
425:             max_tag_len=data.get("max_tag_len", data.get("max_len", 5)),
426:             is_string_format=data.get("is_string_format", False),
427:             separator=data.get("separator", ","),
428:             padding_zero=data.get("padding_zero", True),
429:             min_freq=data.get("min_freq", 1),
430:             use_oov=data.get("use_oov", True),
431:             agg=data.get("agg", "sum"),
432:         )
433:         obj.load_state(data)
434:         return obj
435: 
436:     def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
437:         idx_tensor = interaction[self.field_name]
438:         emb = emb_modules[self.embedding_field_name](idx_tensor)
439:         emb = torch.sum(emb, dim=-2)
440:         if self.agg == "mean":
441:             emb = mean_pooling(emb, idx_tensor, self.padding_zero)
442:         return emb
443: 
444: 
445: if __name__ == "__main__":
446:     source = FeatureSource.USER
447: 
448:     sparse_null = SparseEmbSetting(
449:         "demo",
450:         source=source,
451:         padding_zero=True,
452:         use_null=True,
453:         fill_null="null",
454:         use_oov=True,
455:     )
456:     sparse_null._build_vocab_indices(["a", "b"])
457:     expr = sparse_null.get_transform_expr()
458:     df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
459:     assert df["demo"].to_list() == [sparse_null.null_idx, 1, sparse_null.oov_idx]
460:     assert sparse_null.num_embeddings == 5
461: 
462:     sparse_zero = SparseEmbSetting(
463:         "demo",
464:         source=source,
465:         padding_zero=True,
466:         use_null=True,
467:         fill_null="zero",
468:         use_oov=True,
469:     )
470:     sparse_zero._build_vocab_indices(["a", "b"])
471:     expr = sparse_zero.get_transform_expr()
472:     df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
473:     assert df["demo"].to_list() == [0, 1, sparse_zero.oov_idx]
474:     assert sparse_zero.num_embeddings == 5
475: 
476:     sparse_oov = SparseEmbSetting(
477:         "demo",
478:         source=source,
479:         padding_zero=True,
480:         use_null=True,
481:         fill_null="oov",
482:         use_oov=True,
483:     )
484:     sparse_oov._build_vocab_indices(["a", "b"])
485:     expr = sparse_oov.get_transform_expr()
486:     df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
487:     assert df["demo"].to_list() == [sparse_oov.oov_idx, 1, sparse_oov.oov_idx]
488:     assert sparse_oov.num_embeddings == 5
489: 
490:     sparse_strict = SparseEmbSetting(
491:         "demo",
492:         source=source,
493:         padding_zero=False,
494:         use_null=False,
495:         use_oov=False,
496:     )
497:     sparse_strict._build_vocab_indices(["1", "2"])
498:     expr = sparse_strict.get_transform_expr()
499:     df = pl.DataFrame({"demo": ["1", "2"]}).lazy().with_columns(expr).collect()
500:     assert df["demo"].to_list() == [0, 1]
501: 
502:     multi = MultiSparseSetting(
503:         "tags",
504:         source=source,
505:         padding_zero=True,
506:         use_oov=True,
507:         is_string_format=True,
508:         separator=",",
509:     )
510:     multi._build_vocab_indices(["a", "b"])
511:     expr = multi.get_transform_expr()
512:     df = pl.DataFrame({"tags": [None, "", "a,b", "zzz"]}).lazy().with_columns(expr).collect()
513:     assert df["tags"].to_list() == [[], [], [1, 2], [multi.oov_idx]]
514: 
515:     print("categorical.py self-check passed")
````

## File: src/rosen/emb/schema/numerical.py
````python
  1: from typing import Any, Dict, List, Optional
  2: 
  3: import polars as pl
  4: import torch
  5: from torch import nn
  6: 
  7: from rosen.core.enum_type import FeatureSource
  8: 
  9: from .base import EmbSetting, EmbType
 10: 
 11: 
 12: class BaseNumericalSetting(EmbSetting):
 13:     """连续数值特征基类：显式声明无词表。"""
 14: 
 15:     def __init__(self, field_name: str, source: FeatureSource, embedding_dim: int = 1):
 16:         super().__init__(
 17:             field_name=field_name,
 18:             embedding_dim=embedding_dim,
 19:             source=source,
 20:             padding_zero=False,
 21:             use_oov=False,
 22:         )
 23:         self.is_fitted = False
 24: 
 25:     @property
 26:     def vocab_size(self) -> int:
 27:         return 0
 28: 
 29:     @property
 30:     def num_embeddings(self) -> int:
 31:         return -1
 32: 
 33:     @property
 34:     def requires_embedding_module(self) -> bool:
 35:         return False
 36: 
 37:     def get_formatters(self):
 38:         from rosen.data.padding import DenseFormatter
 39: 
 40:         return {self.field_name: DenseFormatter()}
 41: 
 42: 
 43: class MinMaxDenseSetting(BaseNumericalSetting):
 44:     emb_type = EmbType.DENSE
 45: 
 46:     def __init__(self, field_name: str, source: FeatureSource, min_val: float = None, max_val: float = None):
 47:         super().__init__(field_name=field_name, source=source, embedding_dim=1)
 48:         self.min_val = min_val
 49:         self.max_val = max_val
 50:         self.is_fitted = min_val is not None and max_val is not None
 51: 
 52:     def get_fit_exprs(self) -> List[pl.Expr]:
 53:         return [
 54:             pl.col(self.field_name).cast(pl.Float64).drop_nulls().min().alias(f"{self.field_name}_min"),
 55:             pl.col(self.field_name).cast(pl.Float64).drop_nulls().max().alias(f"{self.field_name}_max"),
 56:         ]
 57: 
 58:     def parse_fit_result(self, result_df: pl.DataFrame):
 59:         self.parse_element_fit_result(result_df, alias=self.field_name)
 60: 
 61:     def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
 62:         self.min_val = result_df.get_column(f"{alias}_min").to_list()[0]
 63:         self.max_val = result_df.get_column(f"{alias}_max").to_list()[0]
 64:         if self.min_val is None or self.max_val is None:
 65:             self.min_val = 0.0
 66:             self.max_val = 1.0
 67:         elif self.min_val == self.max_val:
 68:             self.max_val = self.min_val + 1e-6
 69:         self.is_fitted = True
 70: 
 71:     def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
 72:         exploded = seq_expr.explode().cast(pl.Float64)
 73:         return [
 74:             exploded.min().alias(f"{alias}_min"),
 75:             exploded.max().alias(f"{alias}_max"),
 76:         ]
 77: 
 78:     def get_element_transform_expr(self, expr: pl.Expr = None) -> pl.Expr:
 79:         expr = expr if expr is not None else pl.element()
 80:         range_val = self.max_val - self.min_val
 81:         return ((expr.cast(pl.Float32, strict=False) - self.min_val) / range_val).fill_null(0.0).cast(pl.Float32)
 82: 
 83:     def get_transform_expr(self) -> pl.Expr:
 84:         return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)
 85: 
 86:     def to_dict(self) -> Dict[str, Any]:
 87:         data = super().to_dict()
 88:         data["min_val"] = self.min_val
 89:         data["max_val"] = self.max_val
 90:         return data
 91: 
 92:     def load_state(self, state_dict: Dict[str, Any]):
 93:         super().load_state(state_dict)
 94:         self.min_val = state_dict.get("min_val")
 95:         self.max_val = state_dict.get("max_val")
 96: 
 97:     @classmethod
 98:     def from_dict(cls, data: Dict[str, Any]) -> "MinMaxDenseSetting":
 99:         obj = cls(
100:             field_name=data["field_name"],
101:             source=FeatureSource[data["feature_source"]],
102:             min_val=data.get("min_val"),
103:             max_val=data.get("max_val"),
104:         )
105:         obj.is_fitted = data.get("is_fitted", False)
106:         return obj
107: 
108:     def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
109:         return interaction[self.field_name].unsqueeze(-1)
110: 
111: 
112: class VectorDenseSetting(BaseNumericalSetting):
113:     emb_type = EmbType.VECTOR_DENSE
114: 
115:     def __init__(self, field_name: str, source: 'FeatureSource', embedding_dim: Optional[int] = None,
116:                  zero_fill: bool = True):
117:         super().__init__(field_name=field_name, source=source, embedding_dim=embedding_dim or 0)
118:         self.zero_fill = zero_fill
119:         self.is_fitted = embedding_dim is not None
120: 
121:     def get_fit_exprs(self) -> List[pl.Expr]:
122:         if self.is_fitted:
123:             return []
124: 
125:         len_expr = pl.col(self.field_name).drop_nulls().list.len()
126:         return [
127:             len_expr.value_counts().implode().alias(f"{self.field_name}_len_counts"),
128:             pl.col(self.field_name).is_null().sum().alias(f"{self.field_name}_null_count"),
129:             len_expr.eq(0).sum().alias(f"{self.field_name}_empty_count"),
130:         ]
131: 
132:     def parse_fit_result(self, result_df: pl.DataFrame):
133:         if self.is_fitted:
134:             return
135: 
136:         len_counts_key = f"{self.field_name}_len_counts"
137:         len_rows = result_df.get_column(len_counts_key).to_list()[0]
138:         null_count = int(result_df.get_column(f"{self.field_name}_null_count")[0] or 0)
139:         empty_count = int(result_df.get_column(f"{self.field_name}_empty_count")[0] or 0)
140: 
141:         length_distribution: Dict[int, int] = {}
142:         for row in len_rows or []:
143:             count = row.get("count", row.get("counts", 0))
144:             length = None
145:             for key, value in row.items():
146:                 if key not in ("count", "counts"):
147:                     length = value
148:                     break
149:             if length is None:
150:                 continue
151:             length_distribution[int(length)] = int(count)
152: 
153:         non_empty_distribution = {
154:             int(length): int(count)
155:             for length, count in length_distribution.items()
156:             if int(length) > 0 and int(count) > 0
157:         }
158: 
159:         if not non_empty_distribution:
160:             raise ValueError(
161:                 f"Feature '{self.field_name}' has no non-empty vectors in training data. "
162:                 f"null_rows={null_count}, empty_rows={empty_count}. "
163:                 "Please set `embedding_dim` explicitly if this feature should be kept."
164:             )
165: 
166:         if len(non_empty_distribution) != 1:
167:             dist_lines = [
168:                 f"  {length} -> {count} rows"
169:                 for length, count in sorted(non_empty_distribution.items())
170:             ]
171:             distribution_text = "\n".join(dist_lines)
172:             raise ValueError(
173:                 f"Feature '{self.field_name}' has inconsistent vector lengths on training data.\n"
174:                 f"Observed non-empty length distribution:\n{distribution_text}\n"
175:                 f"null_rows={null_count}, empty_rows={empty_count}\n"
176:                 "Please set `embedding_dim` explicitly for this feature."
177:             )
178: 
179:         self.embedding_dim = next(iter(non_empty_distribution))
180:         self.is_fitted = True
181: 
182:     def get_transform_expr(self) -> pl.Expr:
183:         if not self.is_fitted or not self.embedding_dim:
184:             raise RuntimeError(f"Feature '{self.field_name}' is not fitted yet. embedding_dim is unknown.")
185: 
186:         # Keep the raw float list column. Fixed-width padding/truncation is
187:         # handled in the dataset formatter so checkpoints preserve the original
188:         # vector information instead of silently zeroing mismatched rows.
189:         return pl.col(self.field_name).cast(pl.List(pl.Float32)).alias(self.field_name)
190: 
191:     def get_formatters(self):
192:         from rosen.data.padding import VectorDenseFormatter
193: 
194:         return {
195:             self.field_name: VectorDenseFormatter(
196:                 dim=int(self.embedding_dim),
197:                 zero_fill=self.zero_fill,
198:             )
199:         }
200: 
201:     def to_dict(self) -> Dict[str, Any]:
202:         data = super().to_dict()
203:         data["embedding_dim"] = self.embedding_dim
204:         data["zero_fill"] = self.zero_fill
205:         return data
206: 
207:     @property
208:     def compatible_type_names(self) -> set[str]:
209:         # Backward compatibility: older checkpoints stored vector dense as DENSE.
210:         return {self.serialized_type_name, EmbType.DENSE.name}
211: 
212:     def load_state(self, state_dict: Dict[str, Any]):
213:         super().load_state(state_dict)
214:         self.embedding_dim = state_dict.get("embedding_dim", self.embedding_dim)
215:         self.zero_fill = state_dict.get("zero_fill", self.zero_fill)
216:         self.is_fitted = self.embedding_dim is not None and self.embedding_dim > 0
217: 
218:     @classmethod
219:     def from_dict(cls, data: Dict[str, Any]) -> "VectorDenseSetting":
220:         obj = cls(
221:             field_name=data["field_name"],
222:             source=FeatureSource[data["feature_source"]],
223:             embedding_dim=data.get("embedding_dim", data.get("embedding_size")),
224:             zero_fill=data.get("zero_fill", True),
225:         )
226:         obj.is_fitted = data.get("is_fitted", obj.embedding_dim is not None)
227:         return obj
228: 
229:     def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
230:         return interaction[self.field_name].to(torch.float32)
````

## File: src/rosen/emb/schema/utils.py
````python
 1: from typing import Dict
 2: 
 3: import polars as pl
 4: import torch
 5: 
 6: 
 7: # Historical compatibility placeholder.
 8: # We no longer use this in the main feature pipeline:
 9: # - sparse nulls are mapped directly to a reserved null index
10: # - sequence empties are kept empty and padded by the dataset formatter
11: NULL_FALLBACK = "NULL_FALLBACK"
12: 
13: 
14: def explode_expr(field_name: str, is_string_format: bool = True, separator: str = ",") -> pl.Expr:
15:     expr = pl.col(field_name).drop_nulls()
16:     if is_string_format:
17:         exploded = (
18:             expr.cast(pl.Utf8)
19:             .str.split(separator)
20:             .explode()
21:             .cast(pl.Utf8)
22:             .str.strip_chars()
23:         )
24:     else:
25:         exploded = expr.explode().cast(pl.Utf8)
26:     valid_mask = exploded.is_not_null() & (exploded != "")
27:     return exploded.filter(valid_mask)
28: 
29: 
30: def clear_seq_expr(
31:     field_name: str,
32:     is_string_format: bool,
33:     separator: str,
34:     fill_empty_with_fallback: bool = False,
35: ) -> pl.Expr:
36:     if is_string_format:
37:         expr = (
38:             pl.col(field_name)
39:             .cast(pl.Utf8)
40:             .fill_null("")
41:             .str.split(separator)
42:             .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
43:         )
44:     else:
45:         expr = (
46:             pl.col(field_name)
47:             .cast(pl.List(pl.Utf8))
48:             .fill_null([])
49:             .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
50:         )
51: 
52:     if not fill_empty_with_fallback:
53:         return expr
54: 
55:     return (
56:         pl.when(expr.list.len() > 0)
57:         .then(expr)
58:         .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
59:     )
60: 
61: 
62: def seq_length_expr(
63:     field_name: str,
64:     is_string_format: bool,
65:     separator: str,
66:     alias: str,
67: ) -> pl.Expr:
68:     clean_expr = clear_seq_expr(
69:         field_name=field_name,
70:         is_string_format=is_string_format,
71:         separator=separator,
72:         fill_empty_with_fallback=False,
73:     )
74:     return clean_expr.list.len().cast(pl.UInt32).alias(alias)
75: 
76: 
77: def map_list_to_indices(expr: pl.Expr, vocab: Dict[str, int], oov_idx: int) -> pl.Expr:
78:     keys = pl.Series(list(vocab.keys()), dtype=pl.Utf8)
79:     vals = pl.Series(list(vocab.values()), dtype=pl.UInt32)
80:     default_expr = pl.lit(oov_idx, dtype=pl.UInt32) if oov_idx >= 0 else pl.lit(None, dtype=pl.UInt32)
81:     return expr.list.eval(
82:         pl.element()
83:         .replace_strict(old=keys, new=vals, default=default_expr)
84:         .cast(pl.UInt32)
85:     )
86: 
87: 
88: def mean_pooling(emb: torch.Tensor, idx_tensor: torch.Tensor, padding_zero: bool) -> torch.Tensor:
89:     if padding_zero:
90:         mask_sum = (idx_tensor != 0).sum(dim=-1, keepdim=True).clamp(min=1)
91:     else:
92:         mask_shape = idx_tensor.shape[:-1] + (1,)
93:         mask_sum = torch.full(mask_shape, idx_tensor.shape[-1], dtype=emb.dtype, device=idx_tensor.device)
94:         return emb / mask_sum
95:     return emb / mask_sum.to(dtype=emb.dtype)
````

## File: src/rosen/emb/emblayer.py
````python
  1: from typing import Iterable, List, Dict, Union, Literal, TYPE_CHECKING, Optional, Any
  2: 
  3: import numpy as np
  4: import pandas as pd
  5: import torch
  6: from rosen.emb.schema import (
  7:     EmbSetting,
  8:     MultiSparseSetting,
  9:     SequenceSetting,
 10: )
 11: from rosen.core.enum_type import FeatureSource
 12: from torch import nn
 13: 
 14: from rosen.core.interaction import Interaction
 15: 
 16: if TYPE_CHECKING:
 17:     from rosen.emb.manager import SchemaManager
 18: 
 19: 
 20: # 将settings中的col分成用户侧和物品侧，BoleEmbLayer会把宽表Interaction 变成对应的Dict Embedding
 21: 
 22: 
 23: SPLIT_METHODS = Literal["source", "name", "none", None]
 24: 
 25: 
 26: class RecEmbedding(nn.Module):
 27:     def __init__(self, num_embeddings, embedding_dim, padding_idx=0, init_method='normal', init_std=1e-4):
 28:         super(RecEmbedding, self).__init__()
 29:         self.embedding = nn.Embedding(
 30:             num_embeddings=num_embeddings,
 31:             embedding_dim=embedding_dim,
 32:             padding_idx=padding_idx
 33:         )
 34:         self.init_method = init_method
 35:         self.init_std = init_std
 36:         self.reset_parameters()
 37: 
 38:     def reset_parameters(self):
 39:         if self.init_method == 'normal':
 40:             nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)
 41:         elif self.init_method == 'xavier':
 42:             nn.init.xavier_uniform_(self.embedding.weight.data)
 43:         else:
 44:             pass
 45:         if self.embedding.padding_idx is not None:
 46:             with torch.no_grad():
 47:                 self.embedding.weight[self.embedding.padding_idx].fill_(0)
 48: 
 49:     def reinitialize(self, init_method: Optional[str] = None, init_std: Optional[float] = None):
 50:         if init_method is not None:
 51:             self.init_method = init_method
 52:         if init_std is not None:
 53:             self.init_std = init_std
 54:         self.reset_parameters()
 55: 
 56:     def forward(self, x):
 57:         return self.embedding(x)
 58: 
 59: 
 60: class BoleEmbLayer(nn.Module):
 61:     def __init__(self, emb_settings: Iterable[EmbSetting]):
 62:         super(BoleEmbLayer, self).__init__()
 63:         self.settings: List[EmbSetting] = list(emb_settings)
 64:         self.emb_modules = nn.ModuleDict()
 65:         self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}
 66: 
 67:         for setting in self.settings:
 68:             if self.source2settings.get(setting.source) is None:
 69:                 self.source2settings[setting.source] = []
 70:             self.source2settings[setting.source].append(setting)
 71: 
 72:             if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
 73:                 self.emb_modules[setting.embedding_field_name] = RecEmbedding(
 74:                     num_embeddings=setting.num_embeddings,
 75:                     embedding_dim=setting.embedding_dim,
 76:                     padding_idx={True: 0, False: None}[setting.padding_zero]
 77:                 )
 78: 
 79:     def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor]:
 80:         split_by = split_by.lower() if isinstance(split_by, str) else None
 81:         if split_by not in ("source", "name", "none", None):
 82:             raise ValueError
 83: 
 84:         output_embs = {}
 85:         for source, settings in self.source2settings.items():
 86:             if not settings:
 87:                 continue
 88:             emb_list = []
 89:             for setting in settings:
 90:                 emb = setting.compute_tensor(interaction, self.emb_modules)
 91:                 emb_list.append([setting.field_name, emb])
 92: 
 93:             if emb_list:
 94:                 if split_by == "source":
 95:                     output_embs[source] = torch.cat([emb for name, emb in emb_list], dim=-1)
 96:                 elif split_by == "name":
 97:                     for name, emb in emb_list:
 98:                         output_embs[name] = emb
 99:                 else:
100:                     output_embs["all"] = output_embs.get("all", [])
101:                     output_embs["all"].extend([emb for name, emb in emb_list])
102: 
103:         if split_by is None or split_by == "none":
104:             if output_embs.get("all", None) is None:
105:                 return None
106:             else:
107:                 return torch.cat(output_embs.get("all"), dim=-1)
108:         else:
109:             return output_embs
110: 
111: 
112: class SideEmb(nn.Module):
113:     def __init__(self, all_settings: Iterable[EmbSetting],
114:                  target_sources: Union[FeatureSource, Iterable[FeatureSource]]=None):
115:         super(SideEmb, self).__init__()
116: 
117:         # 处理单 Source 和多 Source 的兼容
118: 
119:         if target_sources is not None:
120:             if not isinstance(target_sources, Iterable):
121:                 target_sources = [target_sources]
122:             # 核心逻辑：过滤出属于自己这一侧的 settings
123:             side_settings = [s for s in all_settings if s.source in target_sources]
124:         else:
125:             side_settings = all_settings
126: 
127:         self.target_sources = target_sources  # 转为 tuple 方便后续固定顺序遍历
128:         self.embedding = BoleEmbLayer(side_settings)
129:         self.embedding_size = sum([s.embedding_dim for s in side_settings])
130: 
131:     def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[torch.Tensor, Dict[FeatureSource, torch.Tensor]]:
132:         # 直接把大宽表吐出的 interaction 传给底层
133:         emb_dict = self.embedding(interaction, split_by)
134:         return emb_dict
135: 
136: 
137: # 具体化实现，极其干净优雅
138: class UserSideEmb(SideEmb):
139:     VALID_SOURCE = (FeatureSource.USER_ID, FeatureSource.USER)
140: 
141:     def __init__(self, all_settings: Iterable[EmbSetting]):
142:         super().__init__(all_settings, self.VALID_SOURCE)
143: 
144: 
145: class ItemSideEmb(SideEmb):
146:     VALID_SOURCE = (FeatureSource.ITEM_ID, FeatureSource.ITEM)
147: 
148:     def __init__(self, all_settings: Iterable[EmbSetting]):
149:         super().__init__(all_settings, self.VALID_SOURCE)
150: 
151: 
152: class InterSideEmb(SideEmb):
153:     VALID_SOURCE = (FeatureSource.INTERACTION,)
154: 
155:     def __init__(self, all_settings: Iterable[EmbSetting]):
156:         super().__init__(all_settings, self.VALID_SOURCE)
157: 
158: # 无任何约束
159: class CustomSideEmb(SideEmb):
160:     def __init__(self, all_settings: Iterable[EmbSetting]):
161:         super().__init__(all_settings)
162: 
163: """已废弃"""
164: class ProfileEncoder(nn.Module):
165:     """
166:     全自动静态画像编码器：告别硬编码，完全数据驱动！
167:     """
168: 
169:     def __init__(self, settings: List['EmbSetting'], profile_path: str,
170:                  id_source: FeatureSource, feature_source: FeatureSource):
171:         super().__init__()
172: 
173:         # 1. 解析图纸：找出谁是主键 (ID)，谁是属性 (Features)
174:         self.id_setting = next((s for s in settings if s.source == id_source), None)
175:         if not self.id_setting:
176:             raise ValueError(f"在 settings 中未找到主键源: {id_source}")
177: 
178:         self.feature_settings = [s for s in settings if s.source == feature_source]
179:         self.feature_names = [s.field_name for s in self.feature_settings]
180: 
181:         # 2. 初始化底层 Embedding 层 (直接复用你写好的 SideEmb)
182:         self.side_emb = SideEmb(settings, target_sources=[id_source, feature_source])
183: 
184:         # 3. 读取静态宽表
185:         df = pd.read_parquet(profile_path)
186: 
187:         # 4. 构建地基 (最大 ID 决定了查找表的行数)
188:         id_col = self.id_setting.field_name
189:         num_entities = df[id_col].max() + 1
190:         entity_ids = torch.tensor(df[id_col].astype(np.int64).values, dtype=torch.long)
191: 
192:         # 5. 全自动装配流水线
193:         for setting in self.feature_settings:
194:             field = setting.field_name
195: 
196:             # --- 分支 A：处理变长序列特征 (必须 Padding) ---
197:             if isinstance(setting, MultiSparseSetting):
198:                 max_len = getattr(setting, 'max_tag_len', 50)
199:                 padded_matrix = self._pad_sequences(df[field], max_len)
200: 
201:                 # 初始化空矩阵 [num_entities, max_len]
202:                 buffer_tensor = torch.zeros((num_entities, max_len), dtype=torch.long)
203:                 vals = torch.tensor(padded_matrix, dtype=torch.long)
204: 
205:             # --- 分支 B：处理单值特征 ---
206:             else:
207:                 # 初始化一维向量 [num_entities]
208:                 buffer_tensor = torch.zeros(num_entities, dtype=torch.long)
209:                 clean_series = df[field].fillna(0).astype(np.int64)
210:                 vals = torch.tensor(clean_series.values, dtype=torch.long)
211: 
212:             # 散列赋值：自动对齐坑位 (完美解决残缺错位问题)
213:             buffer_tensor[entity_ids] = vals
214: 
215:             # 动态注册为模型的 Buffer (名字比如: buf_tag, buf_author_id)
216:             self.register_buffer(f"buf_{field}", buffer_tensor)
217: 
218:     def _pad_sequences(self, series: pd.Series, max_len: int, pad_val: int = 0) -> np.ndarray:
219:         """高性能 Numpy 序列补齐工具，自带数据清洗"""
220:         padded = np.full((len(series), max_len), pad_val, dtype=np.int64)
221:         for i, seq in enumerate(series):
222:             # 防御由于 left join 产生的 NaN/None
223:             if seq is None or (isinstance(seq, float) and np.isnan(seq)) or len(seq) == 0:
224:                 continue
225: 
226:             valid_len = min(len(seq), max_len)
227:             # 【核心修复】不仅截断，还要强制清洗内部元素的类型为 int64
228:             padded[i, :valid_len] = np.array(seq[:valid_len], dtype=np.int64)
229: 
230:         return padded
231: 
232:     def forward(self, query_ids: torch.Tensor, split_by: SPLIT_METHODS= "none"):
233:         """
234:         前向传播：根据传入的 query_ids (比如一串 item_id)，
235:         瞬间从缓存中拉出所有特征，组装成 interaction 字典，喂给 SideEmb。
236:         """
237:         # 1. 构建虚拟 Interaction 字典
238:         interaction = {self.id_setting.field_name: query_ids}
239:         # 2. 动态查表
240:         for field in self.feature_names:
241:             # 取出注册好的 Buffer 矩阵
242:             buffer_tensor = getattr(self, f"buf_{field}")
243:             interaction[field] = buffer_tensor[query_ids]
244: 
245:         # 3. 完美接入你的 SideEmb
246:         return self.side_emb(interaction, split_by=split_by)
247: 
248: class UserProfileEncoder(ProfileEncoder):
249:     ID_SOURCE = FeatureSource.USER_ID
250:     FEATURE_SOURCE = FeatureSource.USER
251:     def __init__(self, settings: List['EmbSetting'], profile_path: str,):
252:         super().__init__(settings, profile_path,
253:                          id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)
254: 
255: class ItemProfileEncoder(ProfileEncoder):
256:     ID_SOURCE = FeatureSource.ITEM_ID
257:     FEATURE_SOURCE = FeatureSource.ITEM
258:     def __init__(self, settings: List['EmbSetting'], profile_path: str,):
259:         super().__init__(settings, profile_path,
260:                          id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)
261: 
262: 
263: class SeqEmbedder(nn.Module):
264:     def __init__(self, seq_field_name, settings: List[EmbSetting], profile_encoder: ProfileEncoder):
265:         super().__init__()
266:         self.settings = settings
267:         filtered_list = [setting for setting in settings if setting.field_name == seq_field_name]
268:         if len(filtered_list) != 1:
269:             raise ValueError
270:         self.id_setting = filtered_list[0]
271:         if not isinstance(self.id_setting, SequenceSetting):
272:             raise ValueError
273:         self.id_setting: SequenceSetting = self.id_setting
274:         self.seq_field_name = seq_field_name
275:         self.seq_len_field_name = self.id_setting.seq_len_field_name
276: 
277:         self.profile_encoder = profile_encoder
278: 
279:     def forward(self, interaction: Interaction, split_by: SPLIT_METHODS="none"):
280:         inter_dict = interaction.interaction
281:         seq = inter_dict[self.seq_field_name]
282:         seq_len = inter_dict[self.seq_len_field_name]
283:         emb_seq = self.profile_encoder.forward(seq, split_by)
284:         return emb_seq, seq_len
285: 
286: 
287: class EmbView:
288:     """
289:     轻量级特征视图：
290:     本身没有任何参数，只负责将请求转发给底层的 OmniEmbLayer，并自动绑定 target_sources。
291:     """
292:     def __init__(self,
293:                  omni_layer: 'OmniEmbLayer',
294:                  target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
295:                  include_fields: Iterable[str] = None,
296:                  exclude_fields: Iterable[str] = None):
297:         super().__init__()
298:         # 引用同一个巨无霸底座
299:         self.omni = omni_layer
300:         self.target_sources = target_sources
301:         self.include_fields = tuple(include_fields) if include_fields is not None else None
302:         self.exclude_fields = tuple(exclude_fields) if exclude_fields is not None else None
303: 
304:     def forward(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
305:         return self.omni(
306:             interaction,
307:             split_by=split_by,
308:             target_sources=self.target_sources,
309:             include_fields=self.include_fields,
310:             exclude_fields=self.exclude_fields,
311:         )
312: 
313:     def __call__(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[
314:         Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
315:         return self.omni(
316:             interaction,
317:             split_by=split_by,
318:             target_sources=self.target_sources,
319:             include_fields=self.include_fields,
320:             exclude_fields=self.exclude_fields,
321:         )
322: 
323:     @property
324:     def embedding_dim(self) -> int:
325:         return self.omni.get_output_dim(
326:             target_sources=self.target_sources,
327:             include_fields=self.include_fields,
328:             exclude_fields=self.exclude_fields,
329:         )
330: 
331:     @property
332:     def selected_settings(self) -> tuple[EmbSetting, ...]:
333:         return tuple(
334:             self.omni.get_settings(
335:                 target_sources=self.target_sources,
336:                 include_fields=self.include_fields,
337:                 exclude_fields=self.exclude_fields,
338:             )
339:         )
340: 
341:     @property
342:     def field_segments(self) -> tuple[tuple[str, int, int], ...]:
343:         cursor = 0
344:         segments = []
345:         for setting in self.selected_settings:
346:             start = cursor
347:             cursor += int(setting.embedding_dim)
348:             segments.append((setting.field_name, start, cursor))
349:         return tuple(segments)
350: 
351:     @property
352:     def field_slices(self) -> Dict[str, slice]:
353:         return {
354:             field_name: slice(start, stop)
355:             for field_name, start, stop in self.field_segments
356:         }
357: 
358: 
359: class SeqGroupView(EmbView):
360:     """
361:     专为序列组设计的强化版视图：
362:     不仅能获取序列特征，还能自动获取严格对齐的 Target 特征和序列长度。
363:     """
364: 
365:     def __init__(self, omni_layer, group_name: str, include_fields: Iterable[str]):
366:         super().__init__(omni_layer, include_fields=include_fields)
367:         self.group_name = group_name
368: 
369:         # 预加载对齐的 element 设置与 seq_len 字段
370:         self.target_settings = []
371:         self.seq_len_field = None
372:         self.time_field = None
373:         self._target_forward_valid = True
374: 
375:         for field in self.include_fields:
376:             setting = next((s for s in self.omni.settings if s.field_name == field), None)
377:             if setting is None:
378:                 continue
379: 
380:             if hasattr(setting, 'element_setting'):
381:                 self.target_settings.append(setting.element_setting)
382:             else:
383:                 self._target_forward_valid = False
384: 
385:             if self.seq_len_field is None and hasattr(setting, 'seq_len_field_name'):
386:                 self.seq_len_field = setting.seq_len_field_name
387:             if hasattr(setting, 'time_field_name'):
388:                 setting_time_field = setting.time_field_name
389:                 if setting_time_field is None:
390:                     continue
391:                 if self.time_field is None:
392:                     self.time_field = setting_time_field
393:                 elif self.time_field != setting_time_field:
394:                     raise ValueError(
395:                         f"Seq group '{self.group_name}' has inconsistent time_field_name: "
396:                         f"{self.time_field} vs {setting_time_field}"
397:                     )
398: 
399:     def forward_target(self, interaction, split_by: str = "none"):
400:         # 实际上所有 setting 都应暴露 element_setting，forward_target 才不会有歧义
401:         if split_by == "source":
402:             raise ValueError("Target sequence view strictly supports split_by='none' or 'name'.")
403: 
404:         computed_results = []
405:         for target_s in self.target_settings:
406:             emb = target_s.compute_tensor(interaction, self.omni.emb_modules)
407:             computed_results.append((target_s, emb))
408:         return self.omni.format_output(computed_results, split_by)
409: 
410:     def fetch_all(self, interaction, include_time: bool = False):
411:         """默认返回 (seq_emb, target_emb, seq_len)，需要时间时返回四元组。"""
412:         seq_emb = self.forward(interaction, split_by="none")
413:         tar_emb = (
414:             self.forward_target(interaction, split_by="none")
415:             if self._target_forward_valid
416:             else ValueError("存在部分 setting 没有 element_setting，函数中断")
417:         )
418:         seq_len = interaction[self.seq_len_field] if self.seq_len_field else None
419:         hist_time = interaction[self.time_field] if self.time_field else None
420:         if include_time:
421:             return seq_emb, tar_emb, seq_len, hist_time
422:         return seq_emb, tar_emb, seq_len
423: 
424: class OmniEmbLayer(nn.Module):
425:     """
426:     巨无霸统一 Embedding 层：
427:     内部纳管所有 settings。forward 时可通过 target_sources 动态指定范围。
428:     彻底取代 SideEmb、UserSideEmb、ItemSideEmb 等一众子类。
429:     """
430:     def __init__(self, emb_settings: Iterable[EmbSetting] = None, manager: 'SchemaManager' = None, init_method='normal', init_std=1e-3):
431:         super(OmniEmbLayer, self).__init__()
432:         if manager is not None:
433:             emb_settings = manager.settings
434:         if emb_settings is None:
435:             raise ValueError("OmniEmbLayer requires manager or emb_settings.")
436: 
437:         self.manager = manager
438:         self.settings: List[EmbSetting] = list(emb_settings)
439:         self.emb_modules = nn.ModuleDict()
440:         self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}
441: 
442:         self.group2fields = {}
443:         self._init_embedding(init_method=init_method, init_std=init_std)
444: 
445:         self.seq_groups: Dict[str, SeqGroupView] = {}
446:         for g_name, fields in self.group2fields.items():
447:             self.seq_groups[g_name] = SeqGroupView(self, group_name=g_name, include_fields=fields)
448: 
449:         self.whole = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
450:                                                    FeatureSource.ITEM_ID, FeatureSource.ITEM,
451:                                                    FeatureSource.INTERACTION))
452:         self.user_all = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER))
453:         self.item_all = EmbView(self, target_sources=(FeatureSource.ITEM_ID, FeatureSource.ITEM))
454:         self.inter = EmbView(self, target_sources=(FeatureSource.INTERACTION,))
455: 
456:         self.user_id = EmbView(self, target_sources=(FeatureSource.USER_ID,))
457:         self.item_id = EmbView(self, target_sources=(FeatureSource.ITEM_ID,))
458: 
459: 
460:         domain_fields = tuple(manager.domain_fields) if manager is not None else tuple()
461:         domain_field = manager.domain_field if manager is not None else None
462:         self.domain = EmbView(self, include_fields=domain_fields)
463:         self.domain_id = EmbView(
464:             self,
465:             include_fields=((domain_field,) if domain_field is not None else tuple()),
466:         )
467:         self.whole_without_domain = EmbView(
468:             self,
469:             target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
470:                             FeatureSource.ITEM_ID, FeatureSource.ITEM,
471:                             FeatureSource.INTERACTION),
472:             exclude_fields=domain_fields,
473:         )
474:         self.inter_without_domain = EmbView(
475:             self,
476:             target_sources=(FeatureSource.INTERACTION,),
477:             exclude_fields=domain_fields,
478:         )
479: 
480:     def _init_embedding(self, init_method='normal', init_std=1e-3):
481:         for setting in self.settings:
482:             # A. 维护 Source 映射
483:             self.source2settings.setdefault(setting.source, []).append(setting)
484: 
485:             # B. 确定哪些特征需要物理权重 (nn.Embedding)
486:             if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
487:                 self.emb_modules[setting.embedding_field_name] = RecEmbedding(
488:                     num_embeddings=setting.num_embeddings,
489:                     embedding_dim=setting.embedding_dim,
490:                     padding_idx={True: 0, False: None}[setting.padding_zero],
491:                     init_method=init_method, init_std=init_std
492:                 )
493: 
494:             # C. 自动收集带有组名标签的序列特征
495:             if hasattr(setting, 'group_name') and setting.group_name:
496:                 self.group2fields.setdefault(setting.group_name, []).append(setting.field_name)
497: 
498:     def reinitialize_large_vocab_embeddings(
499:             self,
500:             vocab_size_threshold: int,
501:             *,
502:             init_method: Optional[str] = None,
503:             init_std: Optional[float] = None,
504:     ) -> List[Dict[str, Any]]:
505:         """
506:         Reinitialize embedding modules whose linked setting vocab size exceeds the threshold.
507: 
508:         Notes:
509:         - This method is opt-in only. Nothing in the default training flow calls it.
510:         - When multiple settings share one embedding table, the table is reinitialized at most once.
511:         - The threshold compares `setting.vocab_size`, not `num_embeddings`, so padding/OOV slots
512:           do not affect the trigger.
513:         """
514:         vocab_size_threshold = int(vocab_size_threshold)
515:         embedding_meta: Dict[str, Dict[str, Any]] = {}
516: 
517:         for setting in self.settings:
518:             if not setting.requires_embedding_module:
519:                 continue
520: 
521:             emb_field_name = setting.embedding_field_name
522:             meta = embedding_meta.setdefault(
523:                 emb_field_name,
524:                 {
525:                     "embedding_field_name": emb_field_name,
526:                     "field_names": [],
527:                     "vocab_size": 0,
528:                     "num_embeddings": int(setting.num_embeddings),
529:                 },
530:             )
531:             if setting.field_name not in meta["field_names"]:
532:                 meta["field_names"].append(setting.field_name)
533:             meta["vocab_size"] = max(int(meta["vocab_size"]), int(setting.vocab_size))
534:             meta["num_embeddings"] = max(int(meta["num_embeddings"]), int(setting.num_embeddings))
535: 
536:         reinitialized = []
537:         for emb_field_name, meta in embedding_meta.items():
538:             if meta["vocab_size"] <= vocab_size_threshold:
539:                 continue
540: 
541:             if emb_field_name not in self.emb_modules:
542:                 continue
543:             emb_module = self.emb_modules[emb_field_name]
544: 
545:             emb_module.reinitialize(init_method=init_method, init_std=init_std)
546:             reinitialized.append(meta)
547: 
548:         return sorted(reinitialized, key=lambda item: item["vocab_size"], reverse=True)
549: 
550: 
551:     def forward(
552:             self,
553:             interaction,
554:             split_by: SPLIT_METHODS = "none",
555:             target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
556:             include_fields: Iterable[str] = None,
557:             exclude_fields: Iterable[str] = None,
558:     ) -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
559: 
560:         split_by = split_by.lower() if isinstance(split_by, str) else None
561:         if split_by not in ("source", "name", "none", None):
562:             raise ValueError(f"Invalid split_by: {split_by}")
563: 
564:         # 1. 路由过滤：找出本次 forward 需要处理的所有 Setting
565:         valid_settings = self._filter_settings(target_sources, include_fields, exclude_fields, split_by)
566:         computed = [(s, s.compute_tensor(interaction, self.emb_modules)) for s in valid_settings]
567:         return self.format_output(computed, split_by)
568: 
569: 
570:     def _filter_settings(self, target_sources, include_fields, exclude_fields, split_by) -> List[EmbSetting]:
571:         """职责一：专门负责解析 target_sources 和 include/exclude_fields，返回合法的 Setting 列表"""
572:         target_sources = set(target_sources) if target_sources else None
573:         include_fields = set(include_fields) if include_fields else None
574:         exclude_fields = set(exclude_fields) if exclude_fields else None
575:         valid_settings = []
576:         for source, settings in self.source2settings.items():
577:             if target_sources and source not in target_sources:
578:                 continue
579:             for setting in settings:
580:                 if include_fields and setting.field_name not in include_fields:
581:                     continue
582:                 if exclude_fields and setting.field_name in exclude_fields:
583:                     continue
584: 
585:                 if (split_by == "none" or split_by is None) and setting.is_sequence_setting:
586:                     if not (include_fields and setting.field_name in include_fields):
587:                         continue
588: 
589:                 valid_settings.append(setting)
590:         return valid_settings
591: 
592:     def format_output(self, computed_results: List[tuple], split_by: str):
593:         """职责三：根据 split_by 策略打包最终输出的 Tensor 或 Dict"""
594:         if not computed_results:
595:             return None
596: 
597:         if split_by == "name":
598:             return {setting.field_name: emb for setting, emb in computed_results}
599: 
600:         if split_by == "source":
601:             output_dict = {}
602:             for setting, emb in computed_results:
603:                 output_dict.setdefault(setting.source, []).append(emb)
604:             return {source: torch.cat(embs, dim=-1) for source, embs in output_dict.items()}
605: 
606:         all_embs = [emb for _, emb in computed_results]
607:         return torch.cat(all_embs, dim=-1)
608: 
609:     def get_output_dim(self,
610:                        target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
611:                        include_fields: Iterable[str] = None,
612:                        exclude_fields: Iterable[str] = None) -> int:
613:         """
614:         获取指定 target_sources 下拼接后 Embedding 的总维度大小。
615:         这是为了弥补移除 SideEmb 后，下游网络无法直接获取 self.embedding_size 的问题。
616:         """
617:         valid_settings = self.get_settings(
618:             target_sources=target_sources,
619:             include_fields=include_fields,
620:             exclude_fields=exclude_fields,
621:         )
622:         return sum(setting.embedding_dim for setting in valid_settings)
623: 
624:     def get_settings(
625:             self,
626:             target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
627:             include_fields: Iterable[str] = None,
628:             exclude_fields: Iterable[str] = None,
629:     ) -> tuple[EmbSetting, ...]:
630:         return tuple(
631:             self._filter_settings(
632:                 target_sources=target_sources,
633:                 include_fields=include_fields,
634:                 exclude_fields=exclude_fields,
635:                 split_by="none",
636:             )
637:         )
````

## File: src/rosen/emb/manager.py
````python
  1: # ==========================================
  2: # 2. 调度层 (Manager Layer) - 计算图统筹
  3: # ==========================================
  4: import gc
  5: import json
  6: import shutil
  7: import ctypes
  8: from pathlib import Path
  9: from typing import Any, Iterable, List, Literal, Mapping, Optional, Union
 10: import polars as pl
 11: 
 12: from rosen.emb.split import SplitConfig, SplitContext, create_split_strategy
 13: from rosen.emb.schema import EmbSetting, EmbType, MultiSparseSetting, SparseEmbSetting
 14: from rosen.core.enum_type import FeatureSource
 15: 
 16: from rosen.utils.sort import sort_parquet_inplace
 17: from rosen.utils.shuffle import write_shuffled_parquet
 18: 
 19: 
 20: def ensure_list(value: Any) -> List[Any]:
 21:     if value is None:
 22:         return []
 23:     if isinstance(value, (str, bytes)):
 24:         return [value]
 25:     if isinstance(value, Iterable):
 26:         return list(value)
 27:     return [value]
 28: 
 29: 
 30: def collect_lazy(lf: pl.LazyFrame, streaming: bool = False) -> pl.DataFrame:
 31:     if not streaming:
 32:         return lf.collect()
 33:     try:
 34:         return lf.collect(streaming=True)
 35:     except TypeError:
 36:         return lf.collect(engine="streaming")
 37:     except ValueError:
 38:         return lf.collect()
 39: 
 40: 
 41: def trim_process_memory():
 42:     gc.collect()
 43:     try:
 44:         ctypes.CDLL("libc.so.6").malloc_trim(0)
 45:     except Exception:
 46:         pass
 47: 
 48: 
 49: VOCAB_DIR_NAME = "vocab"
 50: VOCAB_VALUE_COL = "value"
 51: VOCAB_ID_COL = "id"
 52: 
 53: 
 54: class SchemaManager:
 55:     ITEM_PROFILE_NAME = "item_profile.parquet"
 56:     USER_PROFILE_NAME = "user_profile.parquet"
 57:     SCHEMA_META_NAME = "feature_meta.json"
 58:     WHOLE_DATA_NAME = "whole_dataframe.parquet"
 59:     def __init__(self, settings_list: List[EmbSetting], schema_dir: str,
 60:                  time_field=None,
 61:                  label_fields: Union[str, Iterable[str]]=None,
 62:                  domain_fields: Union[str, Iterable[str]]=None
 63:                  ):
 64:         self.settings = settings_list
 65: 
 66:         ## path & dir
 67:         self.schema_dir = Path(schema_dir)
 68:         self.schema_dir.mkdir(parents=True, exist_ok=True)
 69:         self.meta_filepath = self.schema_dir / self.SCHEMA_META_NAME
 70: 
 71:         ## field_name
 72:         self.uid_field = next(
 73:             (s.field_name for s in self.settings if s.source == FeatureSource.USER_ID),
 74:             None
 75:         )
 76:         self.iid_field =next(
 77:             (s.field_name for s in self.settings if s.source == FeatureSource.ITEM_ID),
 78:             None
 79:         )
 80:         self.time_field = time_field
 81:         self._label_fields = ensure_list(label_fields)
 82:         self._domain_fields = ensure_list(domain_fields)
 83: 
 84:     @property
 85:     def label_field(self):
 86:         return next(iter(self._label_fields), None)
 87:     @property
 88:     def label_fields(self) -> tuple:
 89:         return tuple(self._label_fields)
 90:     @property
 91:     def domain_field(self):
 92:         return next(iter(self._domain_fields), None)
 93:     @property
 94:     def domain_fields(self) -> tuple:
 95:         return tuple(self._domain_fields)
 96: 
 97: 
 98:     def _parse_output_dir(self, output_dir=None):
 99:         if output_dir is None:
100:             output_dir = self.schema_dir
101:         output_dir = Path(output_dir)
102:         output_dir.mkdir(exist_ok=True)
103:         return output_dir
104: 
105: 
106:     def prepare_data(self, lazy_df: pl.LazyFrame, output_dir: Union[Path, str]=None, redo=False):
107:         """
108:         不建议使用，全自动化流程：自动判断输入文件类型，自动构建计算图
109:         """
110:         output_parquet = self._parse_output_dir(output_dir) / self.WHOLE_DATA_NAME
111:         if self.meta_filepath.exists() and output_parquet.exists() and not redo:
112:             print("[Info] 检测到已存在元数据和目标文件，跳过处理直接加载 Schema。")
113:             self.load_schema()
114:             return pl.scan_parquet(output_parquet)
115: 
116:         print("[*] 正在执行计算图截断与中间态落盘，请稍候...")
117:         lazy_df = self._make_checkpoint(lazy_df)
118: 
119:         print("[*] 阶段一：构建 Fit 计算图...")
120:         fit_exprs = []
121:         for setting in self.settings:
122:             if not setting.is_fitted:
123:                 fit_exprs.extend(setting.get_fit_exprs())
124: 
125:         if fit_exprs:
126:             print(f"[*] 启动单次全表扫描获取统计量，包含 {len(fit_exprs)} 个计算节点...")
127:             fit_result = lazy_df.select(fit_exprs).collect()
128:             for setting in self.settings:
129:                 if not setting.is_fitted:
130:                     setting.parse_fit_result(fit_result)
131:             print("[+] 统计量计算与映射表构建完成。")
132:         else:
133:             print("[+] 所有特征已就绪，跳过 Fit 扫描。")
134:         print("[*] 阶段二：构建 Transform 计算图...")
135:         transform_exprs = self._build_transform_exprs()
136:         print(f"[*] 启动流式转换计算，引擎正在将数据写入 {output_parquet} ...")
137:         lazy_df.with_columns(transform_exprs) \
138:             .sink_parquet(output_parquet)
139:         self.save_schema()
140:         print("[+] 预处理全流程结束！")
141:         return pl.scan_parquet(output_parquet)
142: 
143: 
144:     def fit(self, train_raw_lf: pl.LazyFrame, low_memory: bool = False):
145:         """
146:         阶段一：仅使用训练集拟合统计量和词表。
147: 
148:         :param train_raw_lf: Polars LazyFrame 格式的训练数据
149:         :param low_memory: 是否开启低内存模式。
150:                            - False (默认，推荐): 单次全表扫描，所有特征并行计算。速度极快，但内存开销较大。
151:                            - True (内存受限时): 逐个特征扫描。内存占用极低，但 I/O 放大，耗时成倍增加。
152:         """
153:         if self.meta_filepath.exists():
154:             self.load_schema()
155:             print("[+] 所有特征已就绪，跳过 Fit 扫描。")
156:             return
157:         unfitted_settings = [s for s in self.settings if not s.is_fitted]
158:         if not unfitted_settings:
159:             print("[+] 所有特征已就绪，跳过 Fit 扫描。")
160:             self.save_schema()
161:             return
162: 
163:         if low_memory:
164:             print(f"[*] 启动极低内存 Fit 模式，将对 {len(unfitted_settings)} 个特征进行逐列扫描 (耗时较长)...")
165:             for i, setting in enumerate(unfitted_settings):
166:                 fit_exprs = setting.get_fit_exprs()
167:                 if not fit_exprs:
168:                     continue
169:                 print(f"    -> 正在拟合 [{i + 1}/{len(unfitted_settings)}]: {setting.field_name}")
170:                 col_fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
171:                 setting.parse_fit_result(col_fit_result)
172:                 del col_fit_result # 显式释放哈希表内存
173:                 trim_process_memory()
174:             print("[+] 逐列统计量与词表构建完成。")
175:         else:
176:             print(f"[*] 启动高性能并行 Fit 模式，单次扫描全表计算 {len(unfitted_settings)} 个统计量...")
177:             fit_exprs = [expr for s in unfitted_settings for expr in s.get_fit_exprs()]
178:             if fit_exprs:
179:                 # 即使是并行，也建议开启 streaming=True，让 Polars 底层自己尽力优化和 Spilling
180:                 fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
181:                 for setting in unfitted_settings:
182:                     setting.parse_fit_result(fit_result)
183:         self.save_schema()
184: 
185:     def _checkpoint_path(self, file_name: Union[str, Path]) -> Path:
186:         path = Path(file_name)
187:         if not path.is_absolute():
188:             path = self.schema_dir / path
189:         path.parent.mkdir(parents=True, exist_ok=True)
190:         return path
191: 
192:     def make_checkpoint(self, lf, file_name="custom_checkpoint.parquet", redo=True):
193:         path = self._checkpoint_path(file_name)
194:         if not path.exists() or redo:
195:             self._make_checkpoint(lf, file_name)
196:             return pl.scan_parquet(path)
197:         else:
198:             return pl.scan_parquet(path)
199: 
200:     def make_sorted_checkpoint(
201:         self,
202:         lf,
203:         file_name="custom_sorted_checkpoint.parquet",
204:         sort_by=None,
205:         descending=False,
206:         redo=True,
207:         memory_limit: str = "84GB",
208:     ):
209:         if sort_by is None:
210:             raise ValueError("sort_by is required for make_sorted_checkpoint.")
211:         path = self._checkpoint_path(file_name)
212:         if not path.exists() or redo:
213:             self._make_checkpoint(lf, file_name)
214:             sort_parquet_inplace(
215:                 path,
216:                 sort_by,
217:                 descending=descending,
218:                 memory_limit=memory_limit,
219:                 temp_dir=path.parent / f"_{path.stem}_sort_tmp",
220:             )
221:         return pl.scan_parquet(path)
222: 
223:     def make_shuffled_checkpoint(
224:         self,
225:         lf,
226:         file_name="custom_shuffled_checkpoint.parquet",
227:         seed: int = 2026,
228:         bucket_count: int = 128,
229:         batch_size: int = 65536,
230:         redo: bool = True,
231:         exact: bool = False,
232:         mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
233:         compression: str = "zstd",
234:     ):
235:         path = self._checkpoint_path(file_name)
236:         if path.exists() and not redo:
237:             return pl.scan_parquet(path)
238: 
239:         source_path = path.parent / f"_{path.stem}_unshuffled.parquet"
240:         if source_path.exists():
241:             source_path.unlink()
242:         try:
243:             lf.sink_parquet(source_path, compression=compression)
244:             write_shuffled_parquet(
245:                 [source_path],
246:                 path,
247:                 seed=seed,
248:                 bucket_count=bucket_count,
249:                 batch_size=batch_size,
250:                 compression=compression,
251:                 redo=True,
252:                 exact=exact,
253:                 mode=mode,
254:                 temp_dir=path.parent / f"_{path.stem}_shuffle_tmp",
255:             )
256:         finally:
257:             if source_path.exists():
258:                 source_path.unlink()
259:         return pl.scan_parquet(path)
260: 
261:     def transform(self, raw_lf: pl.LazyFrame) -> pl.LazyFrame:
262:         """
263:         阶段二：根据已固化的 Schema 进行流式转换
264:         """
265:         transform_exprs = self._build_transform_exprs()
266:         return raw_lf.with_columns(transform_exprs)
267: 
268:     def transform_csv_partitioned(
269:         self,
270:         input_paths: Union[str, Path, Iterable[Union[str, Path]]],
271:         output_dir: Union[str, Path],
272:         uid_field: Optional[str] = None,
273:         n_parts: int = 100,
274:         batch_size: int = 50000,
275:         batches_per_read: int = 1,
276:         buffer_flush_rows: int = 500000,
277:         read_csv_kwargs: Optional[Mapping[str, Any]] = None,
278:         redo: bool = True,
279:     ) -> List[Path]:
280:         """
281:         分批读取 CSV，边 transform 边按原始 uid hash 分区落盘。
282: 
283:         输出结构:
284:             output_dir/p000/c0000.parquet
285:             output_dir/p000/c0001.parquet
286:             output_dir/p001/c0000.parquet
287:             ...
288:         """
289:         uid_field = uid_field or self.uid_field
290:         if uid_field is None:
291:             raise ValueError("uid_field is required when manager.uid_field is not set.")
292:         if n_parts <= 0:
293:             raise ValueError("n_parts must be positive.")
294:         if batch_size <= 0:
295:             raise ValueError("batch_size must be positive.")
296:         if batches_per_read <= 0:
297:             raise ValueError("batches_per_read must be positive.")
298:         if buffer_flush_rows <= 0:
299:             raise ValueError("buffer_flush_rows must be positive.")
300: 
301:         output_dir = Path(output_dir)
302:         if output_dir.exists() and redo:
303:             shutil.rmtree(output_dir)
304:         output_dir.mkdir(parents=True, exist_ok=True)
305: 
306:         part_dirs = [output_dir / f"p{pid:03d}" for pid in range(n_parts)]
307:         for part_dir in part_dirs:
308:             part_dir.mkdir(parents=True, exist_ok=True)
309: 
310:         existing = [path for part_dir in part_dirs for path in part_dir.glob("*.parquet")]
311:         if existing and not redo:
312:             return part_dirs
313: 
314:         transform_exprs = self._build_transform_exprs()
315:         part_col = "__bb_part"
316:         buffers = {pid: [] for pid in range(n_parts)}
317:         buffer_rows = {pid: 0 for pid in range(n_parts)}
318:         file_counts = {pid: 0 for pid in range(n_parts)}
319: 
320:         def flush(pid: int):
321:             if not buffers[pid]:
322:                 return
323:             out = pl.concat(buffers[pid], how="vertical")
324:             path = part_dirs[pid] / f"c{file_counts[pid]:04d}.parquet"
325:             out.write_parquet(path)
326:             file_counts[pid] += 1
327:             buffers[pid] = []
328:             buffer_rows[pid] = 0
329: 
330:         csv_kwargs = dict(read_csv_kwargs or {})
331:         csv_kwargs["batch_size"] = batch_size
332:         total_rows = 0
333: 
334:         for input_path in ensure_list(input_paths):
335:             input_path = Path(input_path)
336:             print(f"[*] transform csv partitioned: {input_path}")
337:             reader = pl.read_csv_batched(input_path, **csv_kwargs)
338: 
339:             while True:
340:                 batches = reader.next_batches(batches_per_read)
341:                 if batches is None:
342:                     break
343: 
344:                 for raw_df in batches:
345:                     if raw_df.height == 0:
346:                         continue
347:                     if uid_field not in raw_df.columns:
348:                         raise ValueError(f"uid_field '{uid_field}' not found in {input_path}")
349:                     if part_col in raw_df.columns:
350:                         raise ValueError(f"reserved column '{part_col}' already exists in {input_path}")
351: 
352:                     df = (
353:                         raw_df.lazy()
354:                         .with_columns(
355:                             (pl.col(uid_field).hash(seed=0) % n_parts)
356:                             .cast(pl.UInt32)
357:                             .alias(part_col)
358:                         )
359:                         .with_columns(transform_exprs)
360:                         .pipe(collect_lazy, streaming=True)
361:                     )
362: 
363:                     for key, part_df in df.partition_by(
364:                         part_col,
365:                         maintain_order=False,
366:                         include_key=True,
367:                         as_dict=True,
368:                     ).items():
369:                         pid = int(key[0])
370:                         part_df = part_df.drop(part_col)
371:                         buffers[pid].append(part_df)
372:                         buffer_rows[pid] += part_df.height
373:                         if buffer_rows[pid] >= buffer_flush_rows:
374:                             flush(pid)
375: 
376:                     total_rows += df.height
377:                     del raw_df, df
378: 
379:             print(f"    -> done, cumulative rows={total_rows:,}")
380: 
381:         for pid in range(n_parts):
382:             flush(pid)
383: 
384:         print(f"[+] partitioned transform done: rows={total_rows:,}, output={output_dir}")
385:         return part_dirs
386: 
387:     def extract_sequence_partitions(
388:         self,
389:         flat_part_dirs: Iterable[Union[str, Path]],
390:         output_dir: Union[str, Path],
391:         max_seq_len: int,
392:         user_col: str,
393:         time_col: str,
394:         feature_mapping: Mapping[str, str],
395:         seq_len_col: str = "seq_len",
396:         label_col: Optional[str] = None,
397:         positive_label: Any = 1,
398:         select_fields: Optional[Iterable[str]] = None,
399:         redo: bool = True,
400:     ) -> List[Path]:
401:         from rosen.utils.sequential import extract_history_sequences
402: 
403:         output_dir = Path(output_dir)
404:         if output_dir.exists() and redo:
405:             shutil.rmtree(output_dir)
406:         output_dir.mkdir(parents=True, exist_ok=True)
407: 
408:         outputs: List[Path] = []
409:         fields = list(select_fields) if select_fields is not None else None
410: 
411:         for idx, part in enumerate(flat_part_dirs):
412:             part = Path(part)
413:             out_path = output_dir / f"{part.name}.parquet"
414:             outputs.append(out_path)
415:             if out_path.exists() and not redo:
416:                 continue
417: 
418:             if part.is_dir():
419:                 files = sorted(part.glob("*.parquet"))
420:             else:
421:                 files = [part] if part.exists() else []
422: 
423:             if not files:
424:                 continue
425: 
426:             seq_lf = extract_history_sequences(
427:                 pl.scan_parquet([str(path) for path in files]),
428:                 max_seq_len=max_seq_len,
429:                 user_col=user_col,
430:                 time_col=time_col,
431:                 feature_mapping=dict(feature_mapping),
432:                 seq_len_col=seq_len_col,
433:                 label_col=label_col,
434:                 positive_label=positive_label,
435:             )
436:             if fields is not None:
437:                 seq_lf = seq_lf.select(fields)
438:             seq_lf.sink_parquet(out_path)
439:             print(f"    -> seq part {idx + 1}: {out_path}")
440: 
441:         return outputs
442: 
443:     def _build_transform_exprs(self) -> List[pl.Expr]:
444:         transform_exprs: List[pl.Expr] = []
445:         seen_output_names = set()
446:         for setting in self.settings:
447:             if setting.emb_type == EmbType.UNKNOWN:
448:                 continue
449:             expr = setting.get_transform_expr()
450:             expr_list = expr if isinstance(expr, list) else [expr]
451:             for item in expr_list:
452:                 try:
453:                     output_name = item.meta.output_name()
454:                 except Exception:
455:                     output_name = None
456:                 if output_name is not None and output_name in seen_output_names:
457:                     continue
458:                 if output_name is not None:
459:                     seen_output_names.add(output_name)
460:                 transform_exprs.append(item)
461:         return transform_exprs
462: 
463:     def _make_checkpoint(self, lazy_df, file_name="temp_checkpoint.parquet"):
464:         temp_checkpoint_path = self._checkpoint_path(file_name)
465:         lazy_df.sink_parquet(temp_checkpoint_path)
466:         lazy_df = pl.scan_parquet(temp_checkpoint_path)
467:         return lazy_df
468: 
469:     # NOTE 不推荐使用
470:     def generate_profiles(self, lazy_df: pl.LazyFrame, output_dir: str=None, redo=False):
471:         """
472:         利用 FeatureSource 统筹提取并保存静态画像表 (Profile)
473:         """
474:         item_profile_path = self._parse_output_dir(output_dir) / self.ITEM_PROFILE_NAME
475:         user_profile_path = self._parse_output_dir(output_dir) / self.USER_PROFILE_NAME
476: 
477:         print("[*] 阶段三：开始提取静态特征查找表 (Profiles)...")
478:         # 1. Item Profile
479:         item_features = [setting.field_name for setting in self.settings
480:                          if setting.source == FeatureSource.ITEM]
481:         if self.iid_field and (not item_profile_path.exists() or redo):
482:             # 确保主键在提取列表中
483:             if self.iid_field not in item_features:
484:                 item_features.insert(0, self.iid_field)
485: 
486:             print(f"[*] 正在提取 Item 画像，包含特征: {item_features}")
487:             (
488:                 lazy_df
489:                 .select(item_features)
490:                 .drop_nulls(subset=[self.iid_field])  # 剔除由于 left join 产生的空物品
491:                 .unique(subset=[self.iid_field])  # 按物品 ID 去重
492:                 .sort(self.iid_field)  # 【极其关键】按 ID 从小到大排序！
493:                 .sink_parquet(item_profile_path)
494:             )
495:         # 2. 提取 User Profile (同理)
496:         user_features = [
497:             setting.field_name
498:             for setting in self.settings
499:             if setting.source == FeatureSource.USER
500:         ]
501: 
502:         if self.uid_field and (not user_profile_path.exists() or redo):
503:             if self.uid_field not in user_features:
504:                 user_features.insert(0, self.uid_field)
505: 
506:             print(f"[*] 正在提取 User 画像，包含特征: {user_features}")
507:             (
508:                 lazy_df
509:                 .select(user_features)
510:                 .drop_nulls(subset=[self.uid_field])
511:                 .unique(subset=[self.uid_field])
512:                 .sort(self.uid_field)
513:                 .sink_parquet(user_profile_path)
514:             )
515:         print("[+] 静态查找表提取完成！")
516: 
517:     def split_dataset(
518:         self,
519:         lf: pl.LazyFrame,
520:         strategy: Literal['loo', 'time', 'sequential_ratio', 'random_ratio'] = "random_ratio",
521:         output_dir: str = None,
522:         redo: bool = False,
523:         config: Optional[SplitConfig] = None,
524:         **kwargs,
525:     ):
526:         output_dir = self._parse_output_dir(output_dir)
527:         split_ctx = SplitContext(
528:             uid_field=self.uid_field,
529:             iid_field=self.iid_field,
530:             time_field=self.time_field,
531:             checkpoint_fn=self._make_checkpoint  # 把方法当做变量传递进去 (Callback)
532:         )
533:         splitter = create_split_strategy(strategy=strategy, context=split_ctx, config=config, **kwargs)
534:         return splitter.split(lf, output_dir, redo)
535: 
536:     def save_as_dataset(self, train_lf, valid_lf=None, test_lf=None, output_dir: str = None, redo: bool = False):
537:         """
538:         如果预设的split_dataset不和你的想法，那么把三个lf传过来，这边直接保存。利好Kuairand等预设test的数据集
539:         """
540:         output_dir = self._parse_output_dir(output_dir)
541:         train_path = output_dir / "train.parquet"
542:         valid_path = output_dir / "valid.parquet"
543:         test_path = output_dir / "test.parquet"
544:         split_paths = str(train_path), str(valid_path), str(test_path)
545:         if not redo and (train_path.exists() or valid_path.exists() or test_path.exists()):
546:             return split_paths
547:         gc.collect()
548:         for lf, save_path, name in zip([train_lf, valid_lf, test_lf], split_paths, ["Train", "Valid", "Test"]):
549:             if lf is not None:
550:                 print(f"    -> 正在写入 {name} 集合到 {save_path}")
551:                 lf.sink_parquet(str(save_path),)
552:         print("[+] 所有数据落盘完成！")
553:         return split_paths
554: 
555:     def save_schema(self):
556:         vocab_dir = self.meta_filepath.parent / VOCAB_DIR_NAME
557:         vocab_dir.mkdir(parents=True, exist_ok=True)
558:         meta_data = []
559:         for setting in self.settings:
560:             data = setting.to_dict()
561:             for state in (data, data.get("element_setting_state")):
562:                 if not isinstance(state, dict) or not isinstance(state.get("vocab"), dict) or not state["vocab"]:
563:                     continue
564:                 ref = f"{VOCAB_DIR_NAME}/{state['field_name']}.parquet"
565:                 vocab = state["vocab"]
566:                 rows = sorted(vocab.items(), key=lambda item: int(item[1]))
567:                 pl.DataFrame(
568:                     {
569:                         VOCAB_VALUE_COL: [str(key) for key, _ in rows],
570:                         VOCAB_ID_COL: [int(value) for _, value in rows],
571:                     }
572:                 ).write_parquet(self.meta_filepath.parent / ref)
573:                 state["vocab"] = ref
574:             meta_data.append(data)
575:         with open(self.meta_filepath, 'w') as f:
576:             json.dump(meta_data, f, indent=4)
577: 
578:     def load_schema(self):
579:         if not self.meta_filepath.exists():
580:             print(f"[Warning] 找不到元数据文件 {self.meta_filepath}，无法加载 schema。")
581:             return
582: 
583:         with open(self.meta_filepath, 'r', encoding='utf-8') as f:
584:             meta_data_list = json.load(f)
585: 
586:         for data in meta_data_list:
587:             for state in (data, data.get("element_setting_state")):
588:                 if not isinstance(state, dict) or not isinstance(state.get("vocab"), str):
589:                     continue
590:                 vocab_df = pl.read_parquet(
591:                     self.meta_filepath.parent / state["vocab"],
592:                     columns=[VOCAB_VALUE_COL, VOCAB_ID_COL],
593:                 )
594:                 state["vocab"] = {
595:                     str(row[VOCAB_VALUE_COL]): int(row[VOCAB_ID_COL])
596:                     for row in vocab_df.iter_rows(named=True)
597:                 }
598: 
599:         meta_dict = {item["field_name"]: item for item in meta_data_list}
600:         for setting in self.settings:
601:             if setting.field_name in meta_dict:
602:                 saved_state = meta_dict[setting.field_name]
603:                 saved_type = saved_state.get("type")
604:                 if saved_type not in setting.compatible_type_names:
605:                     raise ValueError(
606:                         f"特征 {setting.field_name} 的类型不匹配: "
607:                         f"期望 {sorted(setting.compatible_type_names)}, JSON 中为 {saved_type}"
608:                     )
609: 
610:                 setting.load_state(saved_state)
611:                 print(f"[-] 成功恢复特征状态: {setting.field_name}")
612:             else:
613:                 print(f"[Warning] 特征 {setting.field_name} 在 JSON 元数据中未找到，将保持默认/未拟合状态。")
614: 
615:     def source2emb_dim(self, *sources: FeatureSource):
616:         emb_dim = 0
617:         for setting in self.settings:
618:             if setting.source in sources:
619:                 emb_dim += setting.embedding_dim
620:         return emb_dim
621: 
622:     def get_setting(self, field_name):
623:         for setting in self.settings:
624:             if setting.field_name == field_name:
625:                 return setting
626:         return None
627: 
628:     def fields(self):
629:         fields = []
630:         for setting in self.settings:
631:             for field_name in setting.get_output_field_names():
632:                 if field_name not in fields:
633:                     fields.append(field_name)
634: 
635:         for ctx_field in (self.time_field, *self.label_fields, *self.domain_fields):
636:             if ctx_field is not None and ctx_field not in fields:
637:                 fields.append(ctx_field)
638:         return fields
639: 
640: 
641: # ==========================================
642: # 3. 使用示例 (假设使用 KuaiRand)
643: # ==========================================
644: if __name__ == "__main__":
645:     # 假设特征注册
646:     user_id_setting = SparseEmbSetting("user_id", FeatureSource.USER)
647: 
648:     # 高级序列特征映射：无论原数据是 "1,5,6" 还是 ["1", "5", "6"]，瞬间处理完毕
649:     tags = MultiSparseSetting(
650:         field_name="tag",
651:         source=FeatureSource.ITEM,
652:         is_string_format=True,
653:         separator=","
654:     )
655: 
656:     manager = SchemaManager([user_id_setting, tags])
657: 
658:     from rosen.datasets.kuairand import KuaiRand
659:     item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
660:     inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
661:     user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)
662: 
663:     whole_lf = inter_lf.join(item_lf, on="video_id", how="left")
664:     whole_lf = whole_lf.join(user_lf, on="user_id", how="left")
665:     # 只需要喂入路径，不用 pd.read_csv。即使 kuairand_log.csv 有 50GB 也能在普通笔记本上跑完
666:     # manager.prepare_data("kuairand_log.csv", "kuairand_processed.parquet")
667:     manager.prepare_data(whole_lf, "kuairand_processed.parquet", redo=True)
668:     print("架构编译成功，可供调用。")
669: 
670:     import pandas as pd
671:     df = pd.read_parquet("kuairand_processed.parquet")
672:     print(df.head(3).to_string())
````

## File: src/rosen/models/msr/base.py
````python
 1: import inspect
 2: import warnings
 3: 
 4: import torch
 5: 
 6: from rosen.emb import SchemaManager
 7: from rosen.models.base import BaseModel
 8: 
 9: 
10: class MSRModel(BaseModel):
11:     def __init__(self, manager: SchemaManager, num_domains: int, **kwargs):
12:         super(MSRModel, self).__init__(manager)
13:         self.num_domains = num_domains
14: 
15:     def forward(self, x: torch.Tensor, domain_ids: torch.Tensor):
16:         raise NotImplementedError
17: 
18:     @classmethod
19:     def from_manager(cls, manager: SchemaManager, num_domains: int, **kwargs) -> 'MSRModel':
20:         init_signature = inspect.signature(cls.__init__)
21:         parameters = init_signature.parameters
22:         accepts_var_kwargs = any(
23:             parameter.kind == inspect.Parameter.VAR_KEYWORD
24:             for parameter in parameters.values()
25:         )
26:         if accepts_var_kwargs:
27:             return cls(manager, num_domains, **kwargs)
28: 
29:         valid_kwargs = {}
30:         ignored_keys = []
31:         for key, value in kwargs.items():
32:             if key in parameters and key not in {"self", "manager", "num_domains"}:
33:                 valid_kwargs[key] = value
34:             else:
35:                 ignored_keys.append(key)
36: 
37:         if ignored_keys:
38:             warnings.warn(
39:                 f"{cls.__name__}.from_manager ignored unsupported kwargs: {sorted(ignored_keys)}",
40:                 stacklevel=2,
41:             )
42:         return cls(manager, num_domains, **valid_kwargs)
````

## File: src/rosen/models/base.py
````python
 1: import numpy as np
 2: from torch import nn
 3: 
 4: from rosen.core.interaction import Interaction
 5: from rosen.core.train.loss import LossPort
 6: from rosen.emb import SchemaManager
 7: from rosen.emb.emblayer import OmniEmbLayer
 8: from rosen.utils.observatory.port import ObservationPort
 9: 
10: 
11: class BaseModel(nn.Module):
12:     def __init__(self, manager: SchemaManager):
13:         super(BaseModel, self).__init__()
14:         self.manager: SchemaManager = manager
15:         self.omni_embedding: OmniEmbLayer = OmniEmbLayer(manager=manager)
16:         self.obp: ObservationPort = ObservationPort(namespace=type(self).__name__)
17:         self.lsp: LossPort = LossPort()
18: 
19:     def apply_xavier_initialization(self) -> None:
20:         """Optionally reset linear layers to Xavier normal with zero bias."""
21:         def _reset(module: nn.Module) -> None:
22:             if isinstance(module, nn.Linear):
23:                 nn.init.xavier_normal_(module.weight)
24:                 if module.bias is not None:
25:                     module.bias.data.zero_()
26: 
27:         self.apply(_reset)
28: 
29:     def calculate_loss(self, interaction: Interaction):
30:         raise NotImplementedError
31: 
32:     def predict(self, interaction: Interaction):
33:         raise NotImplementedError
34: 
35:     def full_sort_predict(self, interaction: Interaction):
36:         raise NotImplementedError
37: 
38:     def __str__(self):
39:         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
40:         params = sum([np.prod(p.size()) for p in model_parameters])
41:         return (
42:             super().__str__()
43:             + f": {params}"
44:         )
````

## File: src/rosen/utils/optimize.py
````python
  1: import torch.nn as nn
  2: from typing import Any, Dict, Iterable, List, Tuple, Union
  3: 
  4: 
  5: DEFAULT_NO_DECAY_KEYWORDS = (
  6:     "position_ids",
  7:     # Free parameters with embedding-like semantics.  These are not always
  8:     # owned by nn.Embedding, so module-type inspection alone is insufficient.
  9:     "seed",
 10:     "shift_begin",
 11:     "shift_end",
 12:     "boundary_flag",
 13:     "_vectors",
 14: )
 15: 
 16: def split_params_by_decay(
 17:         named_params: Iterable[Tuple[str, nn.Parameter]],  # 修改这里
 18:         weight_decay: float = 0.01,
 19:         no_decay_keywords: Union[tuple, list] = ("embedding", "position_ids")
 20: ) -> List[Dict[str, Any]]:
 21:     """
 22:     基于张量维度和关键字自动分组优化器参数。
 23:     """
 24:     decay_params = []
 25:     no_decay_params = []
 26: 
 27:     for name, param in named_params: # 修改这里
 28:         if not param.requires_grad:
 29:             continue
 30: 
 31:         if param.ndim < 2 or any(nd in name.lower() for nd in no_decay_keywords):
 32:             no_decay_params.append(param)
 33:         else:
 34:             decay_params.append(param)
 35: 
 36:     return [
 37:         {"params": decay_params, "weight_decay": weight_decay},
 38:         {"params": no_decay_params, "weight_decay": 0.0}
 39:     ]
 40: 
 41: 
 42: def split_model_params_by_decay(
 43:     model: nn.Module,
 44:     weight_decay: float = 1.0e-6,
 45:     no_decay_keywords: Union[tuple, list] = DEFAULT_NO_DECAY_KEYWORDS,
 46: ) -> List[Dict[str, Any]]:
 47:     """Build optimizer groups while never decaying embedding-like parameters.
 48: 
 49:     Matrix/tensor weights receive ``weight_decay``. Biases, normalization
 50:     parameters, scalar/vector parameters, parameters owned directly by an
 51:     ``nn.Embedding`` module, and explicitly named embedding-like free
 52:     parameters receive zero decay. A Linear nested under a container whose
 53:     name happens to contain ``embedding`` is still decayed.
 54: 
 55:     Inspecting module ownership matters: an embedding table may be named
 56:     ``lookup.weight`` and therefore cannot be identified reliably from the
 57:     parameter name alone.
 58:     """
 59:     embedding_parameter_ids = {
 60:         id(parameter)
 61:         for module in model.modules()
 62:         if isinstance(module, nn.Embedding)
 63:         for parameter in module.parameters(recurse=False)
 64:     }
 65:     lowered_keywords = tuple(str(keyword).lower() for keyword in no_decay_keywords)
 66:     decay_params = []
 67:     no_decay_params = []
 68: 
 69:     for name, parameter in model.named_parameters():
 70:         if not parameter.requires_grad:
 71:             continue
 72:         lowered_name = name.lower()
 73:         no_decay = (
 74:             parameter.ndim < 2
 75:             or id(parameter) in embedding_parameter_ids
 76:             or any(keyword in lowered_name for keyword in lowered_keywords)
 77:         )
 78:         (no_decay_params if no_decay else decay_params).append(parameter)
 79: 
 80:     groups = []
 81:     if decay_params:
 82:         groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
 83:     if no_decay_params:
 84:         groups.append({"params": no_decay_params, "weight_decay": 0.0})
 85:     return groups
 86: 
 87: def create_complex_optimizer_groups(
 88:         model: nn.Module,
 89:         decay_dict: Dict[str, float],
 90:         default_decay: float = 0.01
 91: ) -> List[Dict[str, Any]]:
 92:     """
 93:     根据传入的 decay_dict 灵活配置不同网络层的前缀参数衰减。
 94: 
 95:     参数示例:
 96:     decay_dict = {
 97:         "embeddings": 0.0,       # 推荐系统 Embedding
 98:         "moe_router": 0.05,      # 假设 MoE 门控网络需要更强的正则化
 99:         "bias": 0.0,             # 偏置项
100:         "norm": 0.0              # 归一化层
101:     }
102:     """
103:     # 构建一个按 weight_decay 分组的字典
104:     groups = {}
105: 
106:     for name, param in model.named_parameters():
107:         if not param.requires_grad:
108:             continue
109: 
110:         # 默认衰减值
111:         current_decay = default_decay
112: 
113:         # 遍历字典进行匹配（可以根据需求换成正则 re.search）
114:         for key, decay_value in decay_dict.items():
115:             if key in name.lower():
116:                 current_decay = decay_value
117:                 break  # 匹配到第一个优先级最高的就跳出
118: 
119:         # 归入对应的 decay 组
120:         if current_decay not in groups:
121:             groups[current_decay] = []
122:         groups[current_decay].append(param)
123: 
124:     # 转换为 Optimizer 需要的格式
125:     return [
126:         {"params": params, "weight_decay": wd}
127:         for wd, params in groups.items()
128:     ]
````

## File: src/rosen/utils/shuffle.py
````python
  1: from __future__ import annotations
  2: 
  3: import random
  4: import shutil
  5: from pathlib import Path
  6: from typing import Iterable, Literal, Optional
  7: 
  8: import polars as pl
  9: import pyarrow as pa
 10: import pyarrow.parquet as pq
 11: 
 12: 
 13: def _resolve_input_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
 14:     if isinstance(paths, (str, Path)):
 15:         candidates = [Path(paths)]
 16:     else:
 17:         candidates = [Path(path) for path in paths]
 18: 
 19:     resolved: list[Path] = []
 20:     for path in candidates:
 21:         if path.is_dir():
 22:             resolved.extend(sorted(path.glob("*.parquet")))
 23:         elif path.exists():
 24:             resolved.append(path)
 25: 
 26:     if not resolved:
 27:         raise FileNotFoundError("No input parquet files found for shuffle checkpoint.")
 28:     return resolved
 29: 
 30: 
 31: def _parquet_columns(paths: list[Path]) -> list[str]:
 32:     return pq.ParquetFile(paths[0]).schema_arrow.names
 33: 
 34: 
 35: def _filter_existing_fields(paths: list[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
 36:     if fields is None:
 37:         return None
 38:     existing = set(_parquet_columns(paths))
 39:     return [field for field in fields if field in existing]
 40: 
 41: 
 42: def _empty_table(paths: list[Path], fields: Optional[list[str]]) -> pa.Table:
 43:     schema = pq.ParquetFile(paths[0]).schema_arrow
 44:     if fields is not None:
 45:         schema = pa.schema([field for field in schema if field.name in fields])
 46:     arrays = [pa.array([], type=field.type) for field in schema]
 47:     return pa.Table.from_arrays(arrays, schema=schema)
 48: 
 49: 
 50: def _append_parquet_batches(
 51:     input_paths: list[Path],
 52:     out_path: Path,
 53:     *,
 54:     batch_size: int,
 55:     compression: str,
 56:     rand_col: str,
 57:     shuffle_rows: bool,
 58:     seed: int,
 59: ) -> None:
 60:     writer = None
 61:     wrote_rows = False
 62:     try:
 63:         chunk_idx = 0
 64:         for path in input_paths:
 65:             parquet_file = pq.ParquetFile(path)
 66:             for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
 67:                 df = pl.from_arrow(batch)
 68:                 if df.height == 0:
 69:                     continue
 70:                 if shuffle_rows:
 71:                     row_seed = (int(seed) + chunk_idx * 1000003 + batch_idx * 9176) & 0xFFFFFFFF
 72:                     df = df.sample(fraction=1.0, shuffle=True, seed=row_seed)
 73:                 if rand_col in df.columns:
 74:                     df = df.drop(rand_col)
 75:                 table = df.to_arrow()
 76:                 if writer is None:
 77:                     writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
 78:                 writer.write_table(table)
 79:                 wrote_rows = True
 80:             chunk_idx += 1
 81:     finally:
 82:         if writer is not None:
 83:             writer.close()
 84: 
 85:     if not wrote_rows:
 86:         table = _empty_table(input_paths, None)
 87:         pq.write_table(table, out_path, compression=compression)
 88: 
 89: 
 90: class _ParquetWriterGroup:
 91:     def __init__(self, compression: str):
 92:         self.compression = compression
 93:         self._writers: dict[int, pq.ParquetWriter] = {}
 94: 
 95:     def write(self, key: int, path: Path, df: pl.DataFrame) -> int:
 96:         if df.height == 0:
 97:             return 0
 98:         table = df.to_arrow()
 99:         if key not in self._writers:
100:             path.parent.mkdir(parents=True, exist_ok=True)
101:             self._writers[key] = pq.ParquetWriter(str(path), table.schema, compression=self.compression)
102:         self._writers[key].write_table(table)
103:         return int(df.height)
104: 
105:     def close(self) -> None:
106:         first_error = None
107:         for writer in self._writers.values():
108:             try:
109:                 writer.close()
110:             except Exception as exc:  # pragma: no cover - best-effort cleanup
111:                 first_error = first_error or exc
112:         self._writers.clear()
113:         if first_error is not None:
114:             raise first_error
115: 
116: 
117: def _is_power_of_two(value: int) -> bool:
118:     return value > 0 and (value & (value - 1)) == 0
119: 
120: 
121: def _hash_range_target_expr(rand_col: str, target_count: int) -> pl.Expr:
122:     if target_count == 1:
123:         return pl.lit(0, dtype=pl.UInt32)
124:     if _is_power_of_two(target_count):
125:         shift = 64 - (target_count.bit_length() - 1)
126:         step = 1 << shift
127:         return (pl.col(rand_col) // step).cast(pl.UInt32)
128:     return (
129:         ((pl.col(rand_col).cast(pl.Float64) / float(1 << 64)) * target_count)
130:         .floor()
131:         .clip(0, target_count - 1)
132:         .cast(pl.UInt32)
133:     )
134: 
135: 
136: def _write_target_sort_shuffled_parquet(
137:     input_paths: list[Path],
138:     out_path: Path,
139:     *,
140:     selected_fields: Optional[list[str]],
141:     filter_expr: Optional[pl.Expr],
142:     seed: int,
143:     target_count: int,
144:     batch_size: int,
145:     compression: str,
146:     temp_dir: str | Path | None,
147: ) -> None:
148:     """External random-key shuffle.
149: 
150:     Rows are first range-partitioned by the high bits of a stable row hash, then
151:     each target is sorted by the full hash. Concatenating targets in target id
152:     order is equivalent to sorting the full dataset by the random key, without
153:     materializing the full table in memory.
154:     """
155: 
156:     rand_col = "__bb_shuffle_id"
157:     target_col = "__bb_shuffle_target"
158:     tmp_cols = ["__bb_file_idx", "__bb_row_idx", rand_col, target_col]
159:     existing_cols = set(_parquet_columns(input_paths))
160:     conflict_cols = sorted(existing_cols.intersection(tmp_cols))
161:     if conflict_cols:
162:         raise ValueError(f"Input parquet contains reserved shuffle columns: {conflict_cols}")
163: 
164:     tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
165:     if tmp_dir.exists():
166:         shutil.rmtree(tmp_dir)
167:     staging_root = tmp_dir / "stage"
168:     final_root = tmp_dir / "final"
169:     staging_root.mkdir(parents=True, exist_ok=True)
170:     final_root.mkdir(parents=True, exist_ok=True)
171: 
172:     staging_paths = [
173:         staging_root / f"part-{target_id:05d}.parquet"
174:         for target_id in range(target_count)
175:     ]
176:     read_columns = None if filter_expr is not None else selected_fields
177:     writers = _ParquetWriterGroup(compression=compression)
178: 
179:     try:
180:         for file_idx, path in enumerate(input_paths):
181:             row_offset = 0
182:             parquet_file = pq.ParquetFile(path)
183:             for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
184:                 df = pl.from_arrow(batch)
185:                 if filter_expr is not None:
186:                     df = df.filter(filter_expr)
187:                     if df.height == 0:
188:                         row_offset += batch.num_rows
189:                         continue
190:                     if selected_fields is not None:
191:                         df = df.select(selected_fields)
192:                 elif selected_fields is not None:
193:                     df = df.select(selected_fields)
194: 
195:                 if df.height > 0:
196:                     df = (
197:                         df
198:                         .with_row_index("__bb_row_idx", offset=row_offset)
199:                         .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
200:                         .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
201:                         .with_columns(_hash_range_target_expr(rand_col, target_count).alias(target_col))
202:                     )
203: 
204:                     for key, target_df in df.partition_by(
205:                         target_col,
206:                         maintain_order=False,
207:                         include_key=True,
208:                         as_dict=True,
209:                     ).items():
210:                         target_id = int(key[0] if isinstance(key, tuple) else key)
211:                         writers.write(target_id, staging_paths[target_id], target_df)
212: 
213:                 row_offset += batch.num_rows
214:             print(f"    -> shuffle targetize {file_idx + 1}/{len(input_paths)}")
215:     finally:
216:         writers.close()
217: 
218:     try:
219:         target_paths: list[Path] = []
220:         for target_id, staging_path in enumerate(staging_paths):
221:             if not staging_path.exists():
222:                 continue
223:             final_path = final_root / f"part-{target_id:05d}.parquet"
224:             (
225:                 pl.scan_parquet(staging_path)
226:                 .sort(rand_col)
227:                 .drop(tmp_cols)
228:                 .sink_parquet(final_path, compression=compression)
229:             )
230:             target_paths.append(final_path)
231: 
232:         if out_path.exists():
233:             out_path.unlink()
234:         if target_paths:
235:             _append_parquet_batches(
236:                 target_paths,
237:                 out_path,
238:                 batch_size=batch_size,
239:                 compression=compression,
240:                 rand_col=rand_col,
241:                 shuffle_rows=False,
242:                 seed=seed,
243:             )
244:         else:
245:             pq.write_table(_empty_table(input_paths, selected_fields), out_path, compression=compression)
246:     finally:
247:         shutil.rmtree(tmp_dir, ignore_errors=True)
248: 
249: 
250: def write_shuffled_parquet(
251:     input_paths: str | Path | Iterable[str | Path],
252:     out_path: str | Path,
253:     *,
254:     fields: Optional[Iterable[str]] = None,
255:     filter_expr: Optional[pl.Expr] = None,
256:     seed: int = 2026,
257:     bucket_count: int = 64,
258:     batch_size: int = 65536,
259:     compression: str = "zstd",
260:     redo: bool = True,
261:     exact: bool = False,
262:     mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
263:     temp_dir: str | Path | None = None,
264: ) -> Path:
265:     """Write a physically shuffled parquet checkpoint.
266: 
267:     The default target_sort mode performs an external random-key sort by
268:     range-partitioning rows into power-of-two targets, sorting each target by
269:     the stable row hash, and concatenating targets in key-range order.
270:     Set mode="bucket_chunk" to use the older low-memory chunk shuffle.
271:     Set exact=True only when a single in-memory/logical full sort is required.
272:     """
273: 
274:     paths = _resolve_input_paths(input_paths)
275:     out_path = Path(out_path)
276:     if out_path.exists() and not redo:
277:         return out_path
278:     out_path.parent.mkdir(parents=True, exist_ok=True)
279: 
280:     selected_fields = _filter_existing_fields(paths, fields)
281:     rand_col = "__bb_rand"
282: 
283:     if exact:
284:         lf = pl.scan_parquet([str(path) for path in paths])
285:         if filter_expr is not None:
286:             lf = lf.filter(filter_expr)
287:         if selected_fields is not None:
288:             lf = lf.select(selected_fields)
289:         (
290:             lf
291:             .with_row_index("__bb_row_idx")
292:             .with_columns(pl.struct(["__bb_row_idx"]).hash(seed=seed).alias(rand_col))
293:             .sort(rand_col)
294:             .drop(["__bb_row_idx", rand_col])
295:             .sink_parquet(out_path, compression=compression)
296:         )
297:         return out_path
298: 
299:     if bucket_count <= 0:
300:         raise ValueError("bucket_count must be positive.")
301:     if batch_size <= 0:
302:         raise ValueError("batch_size must be positive.")
303: 
304:     if mode not in {"target_sort", "bucket_chunk"}:
305:         raise ValueError("mode must be either 'target_sort' or 'bucket_chunk'.")
306: 
307:     if mode == "target_sort":
308:         _write_target_sort_shuffled_parquet(
309:             paths,
310:             out_path,
311:             selected_fields=selected_fields,
312:             filter_expr=filter_expr,
313:             seed=seed,
314:             target_count=bucket_count,
315:             batch_size=batch_size,
316:             compression=compression,
317:             temp_dir=temp_dir,
318:         )
319:         return out_path
320: 
321:     # TODO: This bucket/chunk path is not equivalent to a true global shuffle.
322:     # If upstream files are sorted by user/time and downstream training reads
323:     # output shards sequentially, large local order blocks can remain. Add a
324:     # round-robin/random shard reader or make exact hash-sort the default for
325:     # no-sequence datasets where the memory cost is acceptable.
326:     tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
327:     if tmp_dir.exists():
328:         shutil.rmtree(tmp_dir)
329:     bucket_root = tmp_dir / "buckets"
330:     bucket_root.mkdir(parents=True, exist_ok=True)
331:     for bucket_id in range(bucket_count):
332:         (bucket_root / f"b{bucket_id:03d}").mkdir(parents=True, exist_ok=True)
333: 
334:     file_counts = {bucket_id: 0 for bucket_id in range(bucket_count)}
335:     read_columns = None if filter_expr is not None else selected_fields
336: 
337:     try:
338:         for file_idx, path in enumerate(paths):
339:             row_offset = 0
340:             parquet_file = pq.ParquetFile(path)
341:             for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
342:                 df = pl.from_arrow(batch)
343:                 if filter_expr is not None:
344:                     df = df.filter(filter_expr)
345:                     if df.height == 0:
346:                         row_offset += batch.num_rows
347:                         continue
348:                     if selected_fields is not None:
349:                         df = df.select(selected_fields)
350: 
351:                 df = (
352:                     df
353:                     .with_row_index("__bb_row_idx", offset=row_offset)
354:                     .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
355:                     .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
356:                     .with_columns((pl.col(rand_col) % bucket_count).cast(pl.UInt32).alias("__bb_bucket"))
357:                 )
358: 
359:                 for key, bucket_df in df.partition_by(
360:                     "__bb_bucket",
361:                     maintain_order=False,
362:                     include_key=True,
363:                     as_dict=True,
364:                 ).items():
365:                     bucket_id = int(key[0] if isinstance(key, tuple) else key)
366:                     chunk_path = (
367:                         bucket_root
368:                         / f"b{bucket_id:03d}"
369:                         / f"c{file_counts[bucket_id]:06d}.parquet"
370:                     )
371:                     (
372:                         bucket_df
373:                         .drop(["__bb_bucket", "__bb_file_idx", "__bb_row_idx"])
374:                         .write_parquet(chunk_path, compression=compression)
375:                     )
376:                     file_counts[bucket_id] += 1
377: 
378:                 row_offset += batch.num_rows
379:             print(f"    -> shuffle bucketize {file_idx + 1}/{len(paths)}")
380: 
381:         rng = random.Random(seed)
382:         bucket_order = list(range(bucket_count))
383:         rng.shuffle(bucket_order)
384: 
385:         shuffled_chunks: list[Path] = []
386:         for bucket_id in bucket_order:
387:             bucket_files = sorted((bucket_root / f"b{bucket_id:03d}").glob("*.parquet"))
388:             rng.shuffle(bucket_files)
389:             shuffled_chunks.extend(bucket_files)
390: 
391:         if out_path.exists():
392:             out_path.unlink()
393:         if shuffled_chunks:
394:             _append_parquet_batches(
395:                 shuffled_chunks,
396:                 out_path,
397:                 batch_size=batch_size,
398:                 compression=compression,
399:                 rand_col=rand_col,
400:                 shuffle_rows=True,
401:                 seed=seed,
402:             )
403:         else:
404:             pq.write_table(_empty_table(paths, selected_fields), out_path, compression=compression)
405:     finally:
406:         shutil.rmtree(tmp_dir, ignore_errors=True)
407: 
408:     return out_path
409: 
410: 
411: def shuffle_parquet_files(*args, **kwargs) -> Path:
412:     return write_shuffled_parquet(*args, **kwargs)
````

## File: src/rosen/utils/sort.py
````python
  1: import os
  2: import shutil
  3: 
  4: import duckdb
  5: from typing import Union, List
  6: from pathlib import Path
  7: 
  8: 
  9: def duckdb_sort_parquet(
 10:         input_path: Union[str, Path],
 11:         output_path: Union[str, Path],
 12:         sort_by: Union[str, List[str]],
 13:         descending: Union[bool, List[bool]] = False,
 14:         memory_limit: str = "84GB",  # 物理防爆盾：强制最高内存限制
 15:         temp_dir: str = "./workspace/tmp"  # 磁盘防爆盾：防止把系统 C 盘写满
 16: ):
 17:     input_path = str(input_path)
 18:     output_path = str(output_path)
 19: 
 20:     # 确保安全的临时目录存在
 21:     os.makedirs(temp_dir, exist_ok=True)
 22: 
 23:     # 统一转换 sort_by 和 descending 为列表，对齐 Polars 体验
 24:     if isinstance(sort_by, str):
 25:         sort_by = [sort_by]
 26:     if isinstance(descending, bool):
 27:         descending = [descending] * len(sort_by)
 28:     elif len(descending) != len(sort_by):
 29:         raise ValueError("descending 的长度必须与 sort_by 一致")
 30: 
 31:     # 构建 SQL 的 ORDER BY 语句
 32:     order_clauses = []
 33:     for col, desc in zip(sort_by, descending):
 34:         direction = "DESC" if desc else "ASC"
 35:         order_clauses.append(f'"{col}" {direction}')
 36:     order_by_str = ", ".join(order_clauses)
 37: 
 38:     print(f"\n[DuckDB] 🚄 启动核外大表物理排序...")
 39:     print(f"    -> 输入: {input_path}")
 40:     print(f"    -> 输出: {output_path}")
 41:     print(f"    -> 规则: ORDER BY {order_by_str}")
 42:     print(f"    -> 资源限制: 内存不超过 {memory_limit}, 临时缓冲: {temp_dir}")
 43: 
 44:     # 建立 DuckDB 连接并注入资源防爆指令
 45:     con = duckdb.connect()
 46:     con.execute(f"PRAGMA memory_limit='{memory_limit}'")
 47:     con.execute(f"PRAGMA temp_directory='{temp_dir}'")
 48: 
 49:     # 执行查询并流式落盘，开启高压缩比
 50:     query = f"""
 51:         COPY (
 52:             SELECT * FROM read_parquet('{input_path}')
 53:             ORDER BY {order_by_str}
 54:         ) TO '{output_path}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
 55:     """
 56:     con.execute(query)
 57:     con.close()
 58: 
 59: def sort_parquet_inplace(
 60:         file_path: Union[str, Path],
 61:         sort_by: Union[str, List[str]],
 62:         descending: Union[bool, List[bool]] = False,
 63:         memory_limit: str = "64GB",
 64:         temp_dir = "./tmp"
 65: ):
 66:     """
 67:     使用 DuckDB 对 Parquet 文件进行极低内存的“原地排序”。
 68:     机制：将排序结果输出到 ./tmp 目录下，完成后直接覆盖原文件，绝对安全。
 69:     """
 70:     file_path = Path(file_path).absolute()
 71:     if not file_path.exists():
 72:         raise FileNotFoundError(f"找不到目标文件: {file_path}")
 73: 
 74:     # 创建安全的临时目录
 75:     temp_dir_path = Path(temp_dir).absolute()
 76:     temp_dir_path.mkdir(parents=True, exist_ok=True)
 77:     temp_output_path = temp_dir_path / f"sorted_{file_path.name}"
 78: 
 79:     # 统一转换 sort_by 和 descending 为列表
 80:     if isinstance(sort_by, str):
 81:         sort_by = [sort_by]
 82:     if isinstance(descending, bool):
 83:         descending = [descending] * len(sort_by)
 84:     elif len(descending) != len(sort_by):
 85:         raise ValueError("descending 的长度必须与 sort_by 一致")
 86: 
 87:     # 构建 SQL 的 ORDER BY 语句
 88:     order_clauses = [
 89:         f'"{col}" {"DESC" if desc else "ASC"}'
 90:         for col, desc in zip(sort_by, descending)
 91:     ]
 92:     order_by_str = ", ".join(order_clauses)
 93: 
 94:     print(f"\n[DuckDB] 🚄 启动大表原地物理排序 (In-place Sort)...")
 95:     print(f"    -> 目标文件: {file_path}")
 96:     print(f"    -> 排序规则: ORDER BY {order_by_str}")
 97:     print(f"    -> 资源限制: 内存 <= {memory_limit}, 临时引擎目录: {temp_dir_path}")
 98: 
 99:     try:
100:         # 建立 DuckDB 连接并注入资源限制
101:         con = duckdb.connect()
102:         # 注意：DuckDB 接收路径最好用 POSIX 格式 (特别是 Windows 下)，避免转义坑
103:         con.execute(f"PRAGMA memory_limit='{memory_limit}'")
104:         con.execute(f"PRAGMA temp_directory='{temp_dir_path.as_posix()}'")
105: 
106:         # 执行查询并流式落盘到 temp_output_path
107:         query = f"""
108:             COPY (
109:                 SELECT * FROM read_parquet('{file_path.as_posix()}')
110:                 ORDER BY {order_by_str}
111:             ) TO '{temp_output_path.as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
112:         """
113:         con.execute(query)
114:         con.close()
115: 
116:         # ✨ 核心：安全地替换原文件
117:         # 使用 shutil.move 可以跨磁盘移动，并且覆盖原文件
118:         shutil.move(str(temp_output_path), str(file_path))
119:         print(f"[DuckDB] ✅ 排序完成！已成功覆写原文件: {file_path.name}\n")
120: 
121:     except Exception as e:
122:         print(f"\n[DuckDB] ❌ 排序过程中发生错误: {e}")
123:         if temp_output_path.exists():
124:             temp_output_path.unlink()
125:         raise e
````

## File: workspace/kr_time_budget/data/v3/schema/feature_meta.json
````json
  1: [
  2:     {
  3:         "type": "SPARSE",
  4:         "field_name": "user_id",
  5:         "embedding_size": 16,
  6:         "num_embeddings": 1003,
  7:         "vocab_size": 1000,
  8:         "feature_source": "USER_ID",
  9:         "is_fitted": true,
 10:         "padding_zero": true,
 11:         "use_oov": true,
 12:         "vocab": "vocab/user_id.parquet",
 13:         "oov_idx": 1002,
 14:         "min_freq": 10,
 15:         "use_null": true,
 16:         "null_idx": 1001,
 17:         "fill_null": "null"
 18:     },
 19:     {
 20:         "type": "SPARSE",
 21:         "field_name": "user_active_degree",
 22:         "embedding_size": 8,
 23:         "num_embeddings": 10,
 24:         "vocab_size": 7,
 25:         "feature_source": "USER",
 26:         "is_fitted": true,
 27:         "padding_zero": true,
 28:         "use_oov": true,
 29:         "vocab": "vocab/user_active_degree.parquet",
 30:         "oov_idx": 9,
 31:         "min_freq": 10,
 32:         "use_null": true,
 33:         "null_idx": 8,
 34:         "fill_null": "null"
 35:     },
 36:     {
 37:         "type": "SPARSE",
 38:         "field_name": "is_live_streamer",
 39:         "embedding_size": 8,
 40:         "num_embeddings": 5,
 41:         "vocab_size": 2,
 42:         "feature_source": "USER",
 43:         "is_fitted": true,
 44:         "padding_zero": true,
 45:         "use_oov": true,
 46:         "vocab": "vocab/is_live_streamer.parquet",
 47:         "oov_idx": 4,
 48:         "min_freq": 10,
 49:         "use_null": true,
 50:         "null_idx": 3,
 51:         "fill_null": "null"
 52:     },
 53:     {
 54:         "type": "SPARSE",
 55:         "field_name": "is_video_author",
 56:         "embedding_size": 8,
 57:         "num_embeddings": 5,
 58:         "vocab_size": 2,
 59:         "feature_source": "USER",
 60:         "is_fitted": true,
 61:         "padding_zero": true,
 62:         "use_oov": true,
 63:         "vocab": "vocab/is_video_author.parquet",
 64:         "oov_idx": 4,
 65:         "min_freq": 10,
 66:         "use_null": true,
 67:         "null_idx": 3,
 68:         "fill_null": "null"
 69:     },
 70:     {
 71:         "type": "SPARSE",
 72:         "field_name": "friend_user_num_range",
 73:         "embedding_size": 8,
 74:         "num_embeddings": 10,
 75:         "vocab_size": 7,
 76:         "feature_source": "USER",
 77:         "is_fitted": true,
 78:         "padding_zero": true,
 79:         "use_oov": true,
 80:         "vocab": "vocab/friend_user_num_range.parquet",
 81:         "oov_idx": 9,
 82:         "min_freq": 10,
 83:         "use_null": true,
 84:         "null_idx": 8,
 85:         "fill_null": "null"
 86:     },
 87:     {
 88:         "type": "SPARSE",
 89:         "field_name": "register_days_range",
 90:         "embedding_size": 8,
 91:         "num_embeddings": 10,
 92:         "vocab_size": 7,
 93:         "feature_source": "USER",
 94:         "is_fitted": true,
 95:         "padding_zero": true,
 96:         "use_oov": true,
 97:         "vocab": "vocab/register_days_range.parquet",
 98:         "oov_idx": 9,
 99:         "min_freq": 10,
100:         "use_null": true,
101:         "null_idx": 8,
102:         "fill_null": "null"
103:     },
104:     {
105:         "type": "SPARSE",
106:         "field_name": "follow_user_num_range",
107:         "embedding_size": 8,
108:         "num_embeddings": 11,
109:         "vocab_size": 8,
110:         "feature_source": "USER",
111:         "is_fitted": true,
112:         "padding_zero": true,
113:         "use_oov": true,
114:         "vocab": "vocab/follow_user_num_range.parquet",
115:         "oov_idx": 10,
116:         "min_freq": 10,
117:         "use_null": true,
118:         "null_idx": 9,
119:         "fill_null": "null"
120:     },
121:     {
122:         "type": "SPARSE",
123:         "field_name": "fans_user_num_range",
124:         "embedding_size": 8,
125:         "num_embeddings": 11,
126:         "vocab_size": 8,
127:         "feature_source": "USER",
128:         "is_fitted": true,
129:         "padding_zero": true,
130:         "use_oov": true,
131:         "vocab": "vocab/fans_user_num_range.parquet",
132:         "oov_idx": 10,
133:         "min_freq": 10,
134:         "use_null": true,
135:         "null_idx": 9,
136:         "fill_null": "null"
137:     },
138:     {
139:         "type": "SPARSE",
140:         "field_name": "onehot_feat0",
141:         "embedding_size": 8,
142:         "num_embeddings": 5,
143:         "vocab_size": 2,
144:         "feature_source": "USER",
145:         "is_fitted": true,
146:         "padding_zero": true,
147:         "use_oov": true,
148:         "vocab": "vocab/onehot_feat0.parquet",
149:         "oov_idx": 4,
150:         "min_freq": 10,
151:         "use_null": true,
152:         "null_idx": 3,
153:         "fill_null": "null"
154:     },
155:     {
156:         "type": "SPARSE",
157:         "field_name": "onehot_feat1",
158:         "embedding_size": 8,
159:         "num_embeddings": 10,
160:         "vocab_size": 7,
161:         "feature_source": "USER",
162:         "is_fitted": true,
163:         "padding_zero": true,
164:         "use_oov": true,
165:         "vocab": "vocab/onehot_feat1.parquet",
166:         "oov_idx": 9,
167:         "min_freq": 10,
168:         "use_null": true,
169:         "null_idx": 8,
170:         "fill_null": "null"
171:     },
172:     {
173:         "type": "SPARSE",
174:         "field_name": "onehot_feat2",
175:         "embedding_size": 8,
176:         "num_embeddings": 26,
177:         "vocab_size": 23,
178:         "feature_source": "USER",
179:         "is_fitted": true,
180:         "padding_zero": true,
181:         "use_oov": true,
182:         "vocab": "vocab/onehot_feat2.parquet",
183:         "oov_idx": 25,
184:         "min_freq": 10,
185:         "use_null": true,
186:         "null_idx": 24,
187:         "fill_null": "null"
188:     },
189:     {
190:         "type": "SPARSE",
191:         "field_name": "onehot_feat3",
192:         "embedding_size": 8,
193:         "num_embeddings": 397,
194:         "vocab_size": 394,
195:         "feature_source": "USER",
196:         "is_fitted": true,
197:         "padding_zero": true,
198:         "use_oov": true,
199:         "vocab": "vocab/onehot_feat3.parquet",
200:         "oov_idx": 396,
201:         "min_freq": 10,
202:         "use_null": true,
203:         "null_idx": 395,
204:         "fill_null": "null"
205:     },
206:     {
207:         "type": "SPARSE",
208:         "field_name": "onehot_feat4",
209:         "embedding_size": 8,
210:         "num_embeddings": 17,
211:         "vocab_size": 14,
212:         "feature_source": "USER",
213:         "is_fitted": true,
214:         "padding_zero": true,
215:         "use_oov": true,
216:         "vocab": "vocab/onehot_feat4.parquet",
217:         "oov_idx": 16,
218:         "min_freq": 10,
219:         "use_null": true,
220:         "null_idx": 15,
221:         "fill_null": "null"
222:     },
223:     {
224:         "type": "SPARSE",
225:         "field_name": "onehot_feat5",
226:         "embedding_size": 8,
227:         "num_embeddings": 8,
228:         "vocab_size": 5,
229:         "feature_source": "USER",
230:         "is_fitted": true,
231:         "padding_zero": true,
232:         "use_oov": true,
233:         "vocab": "vocab/onehot_feat5.parquet",
234:         "oov_idx": 7,
235:         "min_freq": 10,
236:         "use_null": true,
237:         "null_idx": 6,
238:         "fill_null": "null"
239:     },
240:     {
241:         "type": "SPARSE",
242:         "field_name": "onehot_feat6",
243:         "embedding_size": 8,
244:         "num_embeddings": 6,
245:         "vocab_size": 3,
246:         "feature_source": "USER",
247:         "is_fitted": true,
248:         "padding_zero": true,
249:         "use_oov": true,
250:         "vocab": "vocab/onehot_feat6.parquet",
251:         "oov_idx": 5,
252:         "min_freq": 10,
253:         "use_null": true,
254:         "null_idx": 4,
255:         "fill_null": "null"
256:     },
257:     {
258:         "type": "SPARSE",
259:         "field_name": "onehot_feat7",
260:         "embedding_size": 8,
261:         "num_embeddings": 40,
262:         "vocab_size": 37,
263:         "feature_source": "USER",
264:         "is_fitted": true,
265:         "padding_zero": true,
266:         "use_oov": true,
267:         "vocab": "vocab/onehot_feat7.parquet",
268:         "oov_idx": 39,
269:         "min_freq": 10,
270:         "use_null": true,
271:         "null_idx": 38,
272:         "fill_null": "null"
273:     },
274:     {
275:         "type": "SPARSE",
276:         "field_name": "onehot_feat8",
277:         "embedding_size": 8,
278:         "num_embeddings": 286,
279:         "vocab_size": 283,
280:         "feature_source": "USER",
281:         "is_fitted": true,
282:         "padding_zero": true,
283:         "use_oov": true,
284:         "vocab": "vocab/onehot_feat8.parquet",
285:         "oov_idx": 285,
286:         "min_freq": 10,
287:         "use_null": true,
288:         "null_idx": 284,
289:         "fill_null": "null"
290:     },
291:     {
292:         "type": "SPARSE",
293:         "field_name": "onehot_feat9",
294:         "embedding_size": 8,
295:         "num_embeddings": 10,
296:         "vocab_size": 7,
297:         "feature_source": "USER",
298:         "is_fitted": true,
299:         "padding_zero": true,
300:         "use_oov": true,
301:         "vocab": "vocab/onehot_feat9.parquet",
302:         "oov_idx": 9,
303:         "min_freq": 10,
304:         "use_null": true,
305:         "null_idx": 8,
306:         "fill_null": "null"
307:     },
308:     {
309:         "type": "SPARSE",
310:         "field_name": "onehot_feat10",
311:         "embedding_size": 8,
312:         "num_embeddings": 8,
313:         "vocab_size": 5,
314:         "feature_source": "USER",
315:         "is_fitted": true,
316:         "padding_zero": true,
317:         "use_oov": true,
318:         "vocab": "vocab/onehot_feat10.parquet",
319:         "oov_idx": 7,
320:         "min_freq": 10,
321:         "use_null": true,
322:         "null_idx": 6,
323:         "fill_null": "null"
324:     },
325:     {
326:         "type": "SPARSE",
327:         "field_name": "onehot_feat11",
328:         "embedding_size": 8,
329:         "num_embeddings": 6,
330:         "vocab_size": 3,
331:         "feature_source": "USER",
332:         "is_fitted": true,
333:         "padding_zero": true,
334:         "use_oov": true,
335:         "vocab": "vocab/onehot_feat11.parquet",
336:         "oov_idx": 5,
337:         "min_freq": 10,
338:         "use_null": true,
339:         "null_idx": 4,
340:         "fill_null": "null"
341:     },
342:     {
343:         "type": "SPARSE",
344:         "field_name": "onehot_feat12",
345:         "embedding_size": 8,
346:         "num_embeddings": 6,
347:         "vocab_size": 3,
348:         "feature_source": "USER",
349:         "is_fitted": true,
350:         "padding_zero": true,
351:         "use_oov": true,
352:         "vocab": "vocab/onehot_feat12.parquet",
353:         "oov_idx": 5,
354:         "min_freq": 10,
355:         "use_null": true,
356:         "null_idx": 4,
357:         "fill_null": "null"
358:     },
359:     {
360:         "type": "SPARSE",
361:         "field_name": "onehot_feat13",
362:         "embedding_size": 8,
363:         "num_embeddings": 6,
364:         "vocab_size": 3,
365:         "feature_source": "USER",
366:         "is_fitted": true,
367:         "padding_zero": true,
368:         "use_oov": true,
369:         "vocab": "vocab/onehot_feat13.parquet",
370:         "oov_idx": 5,
371:         "min_freq": 10,
372:         "use_null": true,
373:         "null_idx": 4,
374:         "fill_null": "null"
375:     },
376:     {
377:         "type": "SPARSE",
378:         "field_name": "onehot_feat14",
379:         "embedding_size": 8,
380:         "num_embeddings": 6,
381:         "vocab_size": 3,
382:         "feature_source": "USER",
383:         "is_fitted": true,
384:         "padding_zero": true,
385:         "use_oov": true,
386:         "vocab": "vocab/onehot_feat14.parquet",
387:         "oov_idx": 5,
388:         "min_freq": 10,
389:         "use_null": true,
390:         "null_idx": 4,
391:         "fill_null": "null"
392:     },
393:     {
394:         "type": "SPARSE",
395:         "field_name": "onehot_feat15",
396:         "embedding_size": 8,
397:         "num_embeddings": 6,
398:         "vocab_size": 3,
399:         "feature_source": "USER",
400:         "is_fitted": true,
401:         "padding_zero": true,
402:         "use_oov": true,
403:         "vocab": "vocab/onehot_feat15.parquet",
404:         "oov_idx": 5,
405:         "min_freq": 10,
406:         "use_null": true,
407:         "null_idx": 4,
408:         "fill_null": "null"
409:     },
410:     {
411:         "type": "SPARSE",
412:         "field_name": "onehot_feat16",
413:         "embedding_size": 8,
414:         "num_embeddings": 6,
415:         "vocab_size": 3,
416:         "feature_source": "USER",
417:         "is_fitted": true,
418:         "padding_zero": true,
419:         "use_oov": true,
420:         "vocab": "vocab/onehot_feat16.parquet",
421:         "oov_idx": 5,
422:         "min_freq": 10,
423:         "use_null": true,
424:         "null_idx": 4,
425:         "fill_null": "null"
426:     },
427:     {
428:         "type": "SPARSE",
429:         "field_name": "onehot_feat17",
430:         "embedding_size": 8,
431:         "num_embeddings": 6,
432:         "vocab_size": 3,
433:         "feature_source": "USER",
434:         "is_fitted": true,
435:         "padding_zero": true,
436:         "use_oov": true,
437:         "vocab": "vocab/onehot_feat17.parquet",
438:         "oov_idx": 5,
439:         "min_freq": 10,
440:         "use_null": true,
441:         "null_idx": 4,
442:         "fill_null": "null"
443:     },
444:     {
445:         "type": "SPARSE",
446:         "field_name": "tab",
447:         "embedding_size": 8,
448:         "num_embeddings": 6,
449:         "vocab_size": 5,
450:         "feature_source": "INTERACTION",
451:         "is_fitted": true,
452:         "padding_zero": false,
453:         "use_oov": true,
454:         "vocab": "vocab/tab.parquet",
455:         "oov_idx": 5,
456:         "min_freq": 1,
457:         "use_null": false,
458:         "null_idx": -1,
459:         "fill_null": "null"
460:     },
461:     {
462:         "type": "SPARSE",
463:         "field_name": "day_of_week",
464:         "embedding_size": 8,
465:         "num_embeddings": 10,
466:         "vocab_size": 7,
467:         "feature_source": "INTERACTION",
468:         "is_fitted": true,
469:         "padding_zero": true,
470:         "use_oov": true,
471:         "vocab": "vocab/day_of_week.parquet",
472:         "oov_idx": 9,
473:         "min_freq": 1,
474:         "use_null": true,
475:         "null_idx": 8,
476:         "fill_null": "null"
477:     },
478:     {
479:         "type": "SPARSE",
480:         "field_name": "is_weekend",
481:         "embedding_size": 8,
482:         "num_embeddings": 5,
483:         "vocab_size": 2,
484:         "feature_source": "INTERACTION",
485:         "is_fitted": true,
486:         "padding_zero": true,
487:         "use_oov": true,
488:         "vocab": "vocab/is_weekend.parquet",
489:         "oov_idx": 4,
490:         "min_freq": 1,
491:         "use_null": true,
492:         "null_idx": 3,
493:         "fill_null": "null"
494:     },
495:     {
496:         "type": "SPARSE",
497:         "field_name": "start_hour",
498:         "embedding_size": 8,
499:         "num_embeddings": 27,
500:         "vocab_size": 24,
501:         "feature_source": "INTERACTION",
502:         "is_fitted": true,
503:         "padding_zero": true,
504:         "use_oov": true,
505:         "vocab": "vocab/start_hour.parquet",
506:         "oov_idx": 26,
507:         "min_freq": 1,
508:         "use_null": true,
509:         "null_idx": 25,
510:         "fill_null": "null"
511:     },
512:     {
513:         "type": "SPARSE",
514:         "field_name": "is_first_req",
515:         "embedding_size": 4,
516:         "num_embeddings": 5,
517:         "vocab_size": 2,
518:         "feature_source": "INTERACTION",
519:         "is_fitted": true,
520:         "padding_zero": true,
521:         "use_oov": true,
522:         "vocab": "vocab/is_first_req.parquet",
523:         "oov_idx": 4,
524:         "min_freq": 1,
525:         "use_null": true,
526:         "null_idx": 3,
527:         "fill_null": "null"
528:     },
529:     {
530:         "type": "SPARSE",
531:         "field_name": "slate_has_rand",
532:         "embedding_size": 4,
533:         "num_embeddings": 5,
534:         "vocab_size": 2,
535:         "feature_source": "INTERACTION",
536:         "is_fitted": true,
537:         "padding_zero": true,
538:         "use_oov": true,
539:         "vocab": "vocab/slate_has_rand.parquet",
540:         "oov_idx": 4,
541:         "min_freq": 1,
542:         "use_null": true,
543:         "null_idx": 3,
544:         "fill_null": "null"
545:     },
546:     {
547:         "type": "MULTI_SPARSE",
548:         "field_name": "hist_tags",
549:         "embedding_size": 16,
550:         "num_embeddings": 61,
551:         "vocab_size": 59,
552:         "feature_source": "ITEM",
553:         "is_fitted": true,
554:         "padding_zero": true,
555:         "use_oov": true,
556:         "vocab": "vocab/hist_tags.parquet",
557:         "oov_idx": 60,
558:         "min_freq": 1,
559:         "max_tag_len": 15,
560:         "is_string_format": true,
561:         "separator": ",",
562:         "agg": "sum"
563:     },
564:     {
565:         "type": "MULTI_SPARSE",
566:         "field_name": "slate_tags",
567:         "embedding_size": 16,
568:         "num_embeddings": 61,
569:         "vocab_size": 59,
570:         "feature_source": "ITEM",
571:         "is_fitted": true,
572:         "padding_zero": true,
573:         "use_oov": true,
574:         "vocab": "vocab/slate_tags.parquet",
575:         "oov_idx": 60,
576:         "min_freq": 1,
577:         "max_tag_len": 12,
578:         "is_string_format": true,
579:         "separator": ",",
580:         "agg": "sum"
581:     },
582:     {
583:         "type": "MULTI_SPARSE",
584:         "field_name": "slate_video_types",
585:         "embedding_size": 8,
586:         "num_embeddings": 5,
587:         "vocab_size": 3,
588:         "feature_source": "ITEM",
589:         "is_fitted": true,
590:         "padding_zero": true,
591:         "use_oov": true,
592:         "vocab": "vocab/slate_video_types.parquet",
593:         "oov_idx": 4,
594:         "min_freq": 1,
595:         "max_tag_len": 6,
596:         "is_string_format": true,
597:         "separator": ",",
598:         "agg": "sum"
599:     },
600:     {
601:         "type": "MULTI_SPARSE",
602:         "field_name": "slate_music_types",
603:         "embedding_size": 8,
604:         "num_embeddings": 9,
605:         "vocab_size": 7,
606:         "feature_source": "ITEM",
607:         "is_fitted": true,
608:         "padding_zero": true,
609:         "use_oov": true,
610:         "vocab": "vocab/slate_music_types.parquet",
611:         "oov_idx": 8,
612:         "min_freq": 1,
613:         "max_tag_len": 6,
614:         "is_string_format": true,
615:         "separator": ",",
616:         "agg": "sum"
617:     },
618:     {
619:         "type": "DENSE",
620:         "field_name": "user_hist_session_min",
621:         "embedding_size": 1,
622:         "num_embeddings": -1,
623:         "vocab_size": 0,
624:         "feature_source": "INTERACTION",
625:         "is_fitted": true,
626:         "padding_zero": false,
627:         "use_oov": false,
628:         "vocab": {},
629:         "oov_idx": -1,
630:         "min_val": 0.0,
631:         "max_val": 1.0
632:     },
633:     {
634:         "type": "DENSE",
635:         "field_name": "user_hist_req_cnt",
636:         "embedding_size": 1,
637:         "num_embeddings": -1,
638:         "vocab_size": 0,
639:         "feature_source": "INTERACTION",
640:         "is_fitted": true,
641:         "padding_zero": false,
642:         "use_oov": false,
643:         "vocab": {},
644:         "oov_idx": -1,
645:         "min_val": 0.0,
646:         "max_val": 1.0
647:     },
648:     {
649:         "type": "DENSE",
650:         "field_name": "day_prev_sess_cnt",
651:         "embedding_size": 1,
652:         "num_embeddings": -1,
653:         "vocab_size": 0,
654:         "feature_source": "INTERACTION",
655:         "is_fitted": true,
656:         "padding_zero": false,
657:         "use_oov": false,
658:         "vocab": {},
659:         "oov_idx": -1,
660:         "min_val": 0.0,
661:         "max_val": 1.0
662:     },
663:     {
664:         "type": "DENSE",
665:         "field_name": "day_prev_elapsed_min",
666:         "embedding_size": 1,
667:         "num_embeddings": -1,
668:         "vocab_size": 0,
669:         "feature_source": "INTERACTION",
670:         "is_fitted": true,
671:         "padding_zero": false,
672:         "use_oov": false,
673:         "vocab": {},
674:         "oov_idx": -1,
675:         "min_val": 0.0,
676:         "max_val": 1.0
677:     },
678:     {
679:         "type": "DENSE",
680:         "field_name": "elapsed_min",
681:         "embedding_size": 1,
682:         "num_embeddings": -1,
683:         "vocab_size": 0,
684:         "feature_source": "INTERACTION",
685:         "is_fitted": true,
686:         "padding_zero": false,
687:         "use_oov": false,
688:         "vocab": {},
689:         "oov_idx": -1,
690:         "min_val": 0.0,
691:         "max_val": 1.0
692:     },
693:     {
694:         "type": "DENSE",
695:         "field_name": "req_idx",
696:         "embedding_size": 1,
697:         "num_embeddings": -1,
698:         "vocab_size": 0,
699:         "feature_source": "INTERACTION",
700:         "is_fitted": true,
701:         "padding_zero": false,
702:         "use_oov": false,
703:         "vocab": {},
704:         "oov_idx": -1,
705:         "min_val": 0.0,
706:         "max_val": 1.0
707:     },
708:     {
709:         "type": "DENSE",
710:         "field_name": "prev1_click_cnt",
711:         "embedding_size": 1,
712:         "num_embeddings": -1,
713:         "vocab_size": 0,
714:         "feature_source": "INTERACTION",
715:         "is_fitted": true,
716:         "padding_zero": false,
717:         "use_oov": false,
718:         "vocab": {},
719:         "oov_idx": -1,
720:         "min_val": 0.0,
721:         "max_val": 1.0
722:     },
723:     {
724:         "type": "DENSE",
725:         "field_name": "prev1_longview_cnt",
726:         "embedding_size": 1,
727:         "num_embeddings": -1,
728:         "vocab_size": 0,
729:         "feature_source": "INTERACTION",
730:         "is_fitted": true,
731:         "padding_zero": false,
732:         "use_oov": false,
733:         "vocab": {},
734:         "oov_idx": -1,
735:         "min_val": 0.0,
736:         "max_val": 1.0
737:     },
738:     {
739:         "type": "DENSE",
740:         "field_name": "prev1_pos_cnt",
741:         "embedding_size": 1,
742:         "num_embeddings": -1,
743:         "vocab_size": 0,
744:         "feature_source": "INTERACTION",
745:         "is_fitted": true,
746:         "padding_zero": false,
747:         "use_oov": false,
748:         "vocab": {},
749:         "oov_idx": -1,
750:         "min_val": 0.0,
751:         "max_val": 1.0
752:     },
753:     {
754:         "type": "DENSE",
755:         "field_name": "sess_cum_click_rate",
756:         "embedding_size": 1,
757:         "num_embeddings": -1,
758:         "vocab_size": 0,
759:         "feature_source": "INTERACTION",
760:         "is_fitted": true,
761:         "padding_zero": false,
762:         "use_oov": false,
763:         "vocab": {},
764:         "oov_idx": -1,
765:         "min_val": 0.0,
766:         "max_val": 1.0
767:     },
768:     {
769:         "type": "DENSE",
770:         "field_name": "sess_cum_longview_rate",
771:         "embedding_size": 1,
772:         "num_embeddings": -1,
773:         "vocab_size": 0,
774:         "feature_source": "INTERACTION",
775:         "is_fitted": true,
776:         "padding_zero": false,
777:         "use_oov": false,
778:         "vocab": {},
779:         "oov_idx": -1,
780:         "min_val": 0.0,
781:         "max_val": 1.0
782:     },
783:     {
784:         "type": "DENSE",
785:         "field_name": "sess_cum_pos_rate",
786:         "embedding_size": 1,
787:         "num_embeddings": -1,
788:         "vocab_size": 0,
789:         "feature_source": "INTERACTION",
790:         "is_fitted": true,
791:         "padding_zero": false,
792:         "use_oov": false,
793:         "vocab": {},
794:         "oov_idx": -1,
795:         "min_val": 0.0,
796:         "max_val": 1.0
797:     },
798:     {
799:         "type": "DENSE",
800:         "field_name": "sess_cum_hate_cnt",
801:         "embedding_size": 1,
802:         "num_embeddings": -1,
803:         "vocab_size": 0,
804:         "feature_source": "INTERACTION",
805:         "is_fitted": true,
806:         "padding_zero": false,
807:         "use_oov": false,
808:         "vocab": {},
809:         "oov_idx": -1,
810:         "min_val": 0.0,
811:         "max_val": 1.0
812:     },
813:     {
814:         "type": "DENSE",
815:         "field_name": "slate_prior_ctr_mean",
816:         "embedding_size": 1,
817:         "num_embeddings": -1,
818:         "vocab_size": 0,
819:         "feature_source": "INTERACTION",
820:         "is_fitted": true,
821:         "padding_zero": false,
822:         "use_oov": false,
823:         "vocab": {},
824:         "oov_idx": -1,
825:         "min_val": 0.0,
826:         "max_val": 1.0
827:     },
828:     {
829:         "type": "DENSE",
830:         "field_name": "slate_prior_longview_mean",
831:         "embedding_size": 1,
832:         "num_embeddings": -1,
833:         "vocab_size": 0,
834:         "feature_source": "INTERACTION",
835:         "is_fitted": true,
836:         "padding_zero": false,
837:         "use_oov": false,
838:         "vocab": {},
839:         "oov_idx": -1,
840:         "min_val": 0.0,
841:         "max_val": 1.0
842:     },
843:     {
844:         "type": "DENSE",
845:         "field_name": "slate_dur_mean_min",
846:         "embedding_size": 1,
847:         "num_embeddings": -1,
848:         "vocab_size": 0,
849:         "feature_source": "INTERACTION",
850:         "is_fitted": true,
851:         "padding_zero": false,
852:         "use_oov": false,
853:         "vocab": {},
854:         "oov_idx": -1,
855:         "min_val": 0.0,
856:         "max_val": 1.0
857:     },
858:     {
859:         "type": "DENSE",
860:         "field_name": "slate_unique_tag_cnt",
861:         "embedding_size": 1,
862:         "num_embeddings": -1,
863:         "vocab_size": 0,
864:         "feature_source": "INTERACTION",
865:         "is_fitted": true,
866:         "padding_zero": false,
867:         "use_oov": false,
868:         "vocab": {},
869:         "oov_idx": -1,
870:         "min_val": 0.0,
871:         "max_val": 1.0
872:     },
873:     {
874:         "type": "DENSE",
875:         "field_name": "slate_unique_author_cnt",
876:         "embedding_size": 1,
877:         "num_embeddings": -1,
878:         "vocab_size": 0,
879:         "feature_source": "INTERACTION",
880:         "is_fitted": true,
881:         "padding_zero": false,
882:         "use_oov": false,
883:         "vocab": {},
884:         "oov_idx": -1,
885:         "min_val": 0.0,
886:         "max_val": 1.0
887:     }
888: ]
````

## File: workspace/kr_time_budget/data/v3/data_audit.json
````json
  1: {
  2:   "splits": [
  3:     {
  4:       "split": "train",
  5:       "rows": 1721745,
  6:       "sessions": 109368,
  7:       "users": 1000,
  8:       "first_day": 20220408,
  9:       "last_day": 20220506,
 10:       "continue_rate": 0.9364783983690965
 11:     },
 12:     {
 13:       "split": "test",
 14:       "rows": 48898,
 15:       "sessions": 3281,
 16:       "users": 855,
 17:       "first_day": 20220508,
 18:       "last_day": 20220508,
 19:       "continue_rate": 0.9329011411509673
 20:     },
 21:     {
 22:       "split": "valid",
 23:       "rows": 57872,
 24:       "sessions": 3822,
 25:       "users": 894,
 26:       "first_day": 20220507,
 27:       "last_day": 20220507,
 28:       "continue_rate": 0.933957699751175
 29:     }
 30:   ],
 31:   "session_split_conflicts": 0,
 32:   "raw_feature_nulls": {
 33:     "request_id": 0,
 34:     "user_id": 0,
 35:     "session_id": 0,
 36:     "req_idx": 0,
 37:     "slate_items": 0,
 38:     "slate_tags": 0,
 39:     "slate_video_types": 0,
 40:     "slate_music_types": 0,
 41:     "slate_dur_mean_min": 0,
 42:     "slate_unique_author_cnt": 0,
 43:     "slate_has_rand": 0,
 44:     "elapsed_min": 0,
 45:     "prev1_click_cnt": 0,
 46:     "prev1_longview_cnt": 0,
 47:     "prev1_pos_cnt": 0,
 48:     "sess_cum_click_rate": 0,
 49:     "sess_cum_longview_rate": 0,
 50:     "sess_cum_pos_rate": 0,
 51:     "sess_cum_hate_cnt": 0,
 52:     "is_first_req": 0,
 53:     "hist_tags": 0,
 54:     "y_continue": 0,
 55:     "slate_unique_tag_cnt": 0,
 56:     "session_date": 0,
 57:     "end_date": 0,
 58:     "start_hour": 0,
 59:     "tab": 0,
 60:     "session_min": 0,
 61:     "session_req_cnt": 0,
 62:     "session_n": 0,
 63:     "first_session": 0,
 64:     "last_session": 0,
 65:     "day_prev_sess_cnt": 0,
 66:     "day_prev_elapsed_min": 0,
 67:     "split": 0,
 68:     "user_active_degree": 0,
 69:     "is_live_streamer": 0,
 70:     "is_video_author": 0,
 71:     "friend_user_num_range": 0,
 72:     "register_days_range": 0,
 73:     "follow_user_num_range": 0,
 74:     "fans_user_num_range": 0,
 75:     "onehot_feat0": 0,
 76:     "onehot_feat1": 0,
 77:     "onehot_feat2": 0,
 78:     "onehot_feat3": 0,
 79:     "onehot_feat4": 0,
 80:     "onehot_feat5": 0,
 81:     "onehot_feat6": 0,
 82:     "onehot_feat7": 0,
 83:     "onehot_feat8": 0,
 84:     "onehot_feat9": 0,
 85:     "onehot_feat10": 0,
 86:     "onehot_feat11": 0,
 87:     "onehot_feat12": 0,
 88:     "onehot_feat13": 0,
 89:     "onehot_feat14": 0,
 90:     "onehot_feat15": 0,
 91:     "onehot_feat16": 0,
 92:     "onehot_feat17": 0,
 93:     "user_id_raw": 0,
 94:     "tab_raw": 0,
 95:     "req_idx_raw": 0,
 96:     "intra_step_group": 0,
 97:     "day_of_week": 0,
 98:     "is_weekend": 0,
 99:     "active_step_group": 0,
100:     "user_prior_n": 0,
101:     "user_hist_session_min": 0,
102:     "user_hist_req_cnt": 0,
103:     "slate_prior_ctr_mean": 0,
104:     "slate_prior_longview_mean": 0,
105:     "item_prior_support_mean": 0,
106:     "item_prior_cold_fraction": 0
107:   },
108:   "clock_decreases": 63080,
109:   "feedback_bounds": {
110:     "prev1_click_cnt": 6,
111:     "prev1_longview_cnt": 6,
112:     "prev1_pos_cnt": 6,
113:     "sess_cum_pos_rate": 1.0,
114:     "slate_unique_tag_cnt": 12
115:   },
116:   "scenarios": [
117:     {
118:       "split": "test",
119:       "tab": 0,
120:       "len": 5645
121:     },
122:     {
123:       "split": "test",
124:       "tab": 1,
125:       "len": 40208
126:     },
127:     {
128:       "split": "test",
129:       "tab": 2,
130:       "len": 2118
131:     },
132:     {
133:       "split": "test",
134:       "tab": 3,
135:       "len": 309
136:     },
137:     {
138:       "split": "test",
139:       "tab": 4,
140:       "len": 618
141:     },
142:     {
143:       "split": "train",
144:       "tab": 0,
145:       "len": 222544
146:     },
147:     {
148:       "split": "train",
149:       "tab": 1,
150:       "len": 1403752
151:     },
152:     {
153:       "split": "train",
154:       "tab": 2,
155:       "len": 58549
156:     },
157:     {
158:       "split": "train",
159:       "tab": 3,
160:       "len": 16062
161:     },
162:     {
163:       "split": "train",
164:       "tab": 4,
165:       "len": 20838
166:     },
167:     {
168:       "split": "valid",
169:       "tab": 0,
170:       "len": 7040
171:     },
172:     {
173:       "split": "valid",
174:       "tab": 1,
175:       "len": 47147
176:     },
177:     {
178:       "split": "valid",
179:       "tab": 2,
180:       "len": 2188
181:     },
182:     {
183:       "split": "valid",
184:       "tab": 3,
185:       "len": 344
186:     },
187:     {
188:       "split": "valid",
189:       "tab": 4,
190:       "len": 1153
191:     }
192:   ],
193:   "source": {
194:     "sources": [
195:       {
196:         "path": "/Users/lxogri/Developer/pyprojects/time-budget/datasource/KuaiRand/KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv",
197:         "rows": 43028,
198:         "bytes": 3086015
199:       },
200:       {
201:         "path": "/Users/lxogri/Developer/pyprojects/time-budget/datasource/KuaiRand/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv",
202:         "rows": 5055984,
203:         "bytes": 373670312
204:       },
205:       {
206:         "path": "/Users/lxogri/Developer/pyprojects/time-budget/datasource/KuaiRand/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv",
207:         "rows": 6657061,
208:         "bytes": 492303531
209:       }
210:     ],
211:     "raw_rows": 11756073,
212:     "deduplicated_rows": 11661848,
213:     "dedup": "all 19 raw columns"
214:   },
215:   "requests_not_six_events": 0,
216:   "duplicate_event_memberships": 0
217: }
````

## File: workspace/kr_time_budget/data/v3/manifest.json
````json
  1: {
  2:   "version": 3,
  3:   "features": [
  4:     "user_id",
  5:     "user_active_degree",
  6:     "is_live_streamer",
  7:     "is_video_author",
  8:     "friend_user_num_range",
  9:     "register_days_range",
 10:     "follow_user_num_range",
 11:     "fans_user_num_range",
 12:     "onehot_feat0",
 13:     "onehot_feat1",
 14:     "onehot_feat2",
 15:     "onehot_feat3",
 16:     "onehot_feat4",
 17:     "onehot_feat5",
 18:     "onehot_feat6",
 19:     "onehot_feat7",
 20:     "onehot_feat8",
 21:     "onehot_feat9",
 22:     "onehot_feat10",
 23:     "onehot_feat11",
 24:     "onehot_feat12",
 25:     "onehot_feat13",
 26:     "onehot_feat14",
 27:     "onehot_feat15",
 28:     "onehot_feat16",
 29:     "onehot_feat17",
 30:     "user_hist_session_min",
 31:     "user_hist_req_cnt",
 32:     "tab",
 33:     "day_of_week",
 34:     "is_weekend",
 35:     "start_hour",
 36:     "day_prev_sess_cnt",
 37:     "day_prev_elapsed_min",
 38:     "elapsed_min",
 39:     "req_idx",
 40:     "is_first_req",
 41:     "prev1_click_cnt",
 42:     "prev1_longview_cnt",
 43:     "prev1_pos_cnt",
 44:     "sess_cum_click_rate",
 45:     "sess_cum_longview_rate",
 46:     "sess_cum_pos_rate",
 47:     "sess_cum_hate_cnt",
 48:     "hist_tags",
 49:     "slate_tags",
 50:     "slate_video_types",
 51:     "slate_music_types",
 52:     "slate_prior_ctr_mean",
 53:     "slate_prior_longview_mean",
 54:     "slate_dur_mean_min",
 55:     "slate_unique_tag_cnt",
 56:     "slate_unique_author_cnt",
 57:     "slate_has_rand"
 58:   ],
 59:   "splits": {
 60:     "train": {
 61:       "rows": 1721745,
 62:       "users": 1000,
 63:       "positive_rate": 0.9364783983690965,
 64:       "request_order_sha256": "bf59bf8492385cb4358c89ba06781528f18f81541d89b04742facaf13f01b51a"
 65:     },
 66:     "valid": {
 67:       "rows": 57872,
 68:       "users": 894,
 69:       "positive_rate": 0.933957699751175,
 70:       "request_order_sha256": "5bfb9d2af0f10bbaba102934b1e969d3e6cd203ea775fc33a191f49877eb4917"
 71:     },
 72:     "test": {
 73:       "rows": 48898,
 74:       "users": 855,
 75:       "positive_rate": 0.9329011411509673,
 76:       "request_order_sha256": "46ac632ce48e70f2ddb933866fe135b0cdf8d53f1178cc61ebf88e52329aeffc"
 77:     }
 78:   },
 79:   "audit": {
 80:     "train": {
 81:       "unseen_users": 0,
 82:       "user_history_coverage": 0.9704311614089194,
 83:       "item_cold_fraction": 0.5748582591111536
 84:     },
 85:     "valid": {
 86:       "unseen_users": 0,
 87:       "user_history_coverage": 1.0,
 88:       "item_cold_fraction": 0.507435950603631
 89:     },
 90:     "test": {
 91:       "unseen_users": 0,
 92:       "user_history_coverage": 1.0,
 93:       "item_cold_fraction": 0.6148172386055326
 94:     }
 95:   },
 96:   "tags": {
 97:     "slate_tags": {
 98:       "cap": 12,
 99:       "truncated_fraction": 0.0001853963462153715
100:     },
101:     "hist_tags": {
102:       "cap": 15,
103:       "truncated_fraction": 0.0011003464560039157
104:     }
105:   },
106:   "protocol": {
107:     "train": "session ends <=20220506",
108:     "valid": "whole session on 20220507",
109:     "test": "whole session on 20220508",
110:     "session_tab": "first event",
111:     "top_tabs": "top5 train sessions",
112:     "train_priors": "strictly earlier calendar days; users completed sessions; items all observed exposures",
113:     "eval_priors": "frozen through 20220506",
114:     "user_smoothing": 5,
115:     "item_smoothing": 20,
116:     "positive": "any like/comment/follow/forward per event",
117:     "tags": "repetitions retained; chronological order; head truncation; sum aggregation; independent hist/slate vocab and tables",
118:     "head_remainder": "chunk0 history only; participates in previous-chunk stats and last2chunk tags",
119:     "duration_clip_ms": [
120:       3000,
121:       600000
122:     ],
123:     "clock": "Leader screen end; delta1min",
124:     "content_elapsed": "full-vector content path; excluded only from budget parameter net"
125:   },
126:   "source": {
127:     "sources": [
128:       {
129:         "path": "/Users/lxogri/Developer/pyprojects/time-budget/datasource/KuaiRand/KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv",
130:         "rows": 43028,
131:         "bytes": 3086015
132:       },
133:       {
134:         "path": "/Users/lxogri/Developer/pyprojects/time-budget/datasource/KuaiRand/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv",
135:         "rows": 5055984,
136:         "bytes": 373670312
137:       },
138:       {
139:         "path": "/Users/lxogri/Developer/pyprojects/time-budget/datasource/KuaiRand/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv",
140:         "rows": 6657061,
141:         "bytes": 492303531
142:       }
143:     ],
144:     "raw_rows": 11756073,
145:     "deduplicated_rows": 11661848,
146:     "dedup": "all 19 raw columns"
147:   },
148:   "structure": {
149:     "top_tabs": [
150:       1,
151:       0,
152:       2,
153:       4,
154:       3
155:     ],
156:     "sessions_total": 147066,
157:     "sessions_retained": 116471,
158:     "boundary_sessions_excluded": 268,
159:     "requests": 1828515
160:   }
161: }
````

## File: workspace/kr_time_budget/pipeline/audit_feature_categories.py
````python
 1: """Audit the 30+6+18 contract against existing KR v3 data, without rebuilding it."""
 2: import hashlib
 3: import json
 4: from pathlib import Path
 5: import polars as pl
 6: from torch.utils.data import DataLoader
 7: from rosen.data.dataset import ParquetStreamDataset
 8: from workspace.kr_time_budget.pipeline.kr_request_schema import (
 9:     request_manager, FEATURES, EXTRA_COLUMNS, FEATURE_CATEGORIES, USER_PROFILE_FIELDS,
10:     SESSION_CONTEXT_FIELDS, BUDGET_PRIOR_FIELDS, SESSION_CLOCK_FIELDS,
11:     CONTENT_HISTORY_FIELDS, CONTENT_SLATE_FIELDS, BUDGET_PARAMETER_FIELDS,
12:     BUDGET_SCOPE_FIELDS, CONTENT_TOWER_FIELDS, CONTENT_MASK_FIELDS, LOG1P_FIELDS,
13: )
14: 
15: ROOT = Path(__file__).resolve().parents[1]
16: 
17: 
18: def main():
19:     data = ROOT/'data/v3'
20:     manifest_path = data/'manifest.json'
21:     manifest = json.loads(manifest_path.read_text())
22:     assert manifest['features'] == FEATURES
23:     manager = request_manager(data/'schema', load=True)
24:     assert {s.field_name for s in manager.settings} == set(FEATURES)
25:     assert all(manager.get_setting(c).min_val == 0 and manager.get_setting(c).max_val == 1
26:                for c in ['elapsed_min', 'req_idx'])
27:     output = {
28:         'contract': 'shared30_budget6_content18', 'data': 'workspace/kr_time_budget/data/v3',
29:         'dataset_rebuild_required': False, 'dataset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
30:         'categories': FEATURE_CATEGORIES,
31:         'subgroups': {'shared_user': USER_PROFILE_FIELDS, 'shared_context': SESSION_CONTEXT_FIELDS,
32:                      'budget_priors': BUDGET_PRIOR_FIELDS, 'clocks': SESSION_CLOCK_FIELDS,
33:                      'content_history': CONTENT_HISTORY_FIELDS, 'content_slate': CONTENT_SLATE_FIELDS},
34:         'target_routes': {'budget_parameters': BUDGET_PARAMETER_FIELDS, 'budget_scope': BUDGET_SCOPE_FIELDS,
35:                           'content': CONTENT_TOWER_FIELDS, 'content_mask': sorted(CONTENT_MASK_FIELDS), 'fusion': FEATURES},
36:         'existing_v5_budget_formula': {'elapsed_min': 'consumed, minutes, delta=1 minute',
37:                                      'req_idx': 'not consumed; dual-clock formula pending'},
38:         'model_status': 'classification complete; historical v5 routing unchanged; no new full training',
39:         'log1p_fields': LOG1P_FIELDS, 'splits': {},
40:     }
41:     raw = pl.scan_parquet(data/'features_raw.parquet')
42:     stable = raw.group_by('session_id').agg(*[pl.col(c).n_unique().alias(c) for c in BUDGET_PARAMETER_FIELDS])
43:     maxima = stable.select(*[pl.col(c).max().alias(c) for c in BUDGET_PARAMETER_FIELDS]).collect().row(0, named=True)
44:     assert all(value == 1 for value in maxima.values()), maxima
45:     output['budget_parameter_fields_constant_within_session'] = maxima
46:     for split in ['train', 'valid', 'test']:
47:         scan = pl.scan_parquet(data/f'{split}.parquet')
48:         stored = set(scan.collect_schema().names())
49:         assert set(FEATURES) <= stored
50:         ids = scan.select('request_id').collect()['request_id']
51:         expected = manifest['splits'][split]
52:         order_hash = hashlib.sha256(ids.to_numpy().tobytes()).hexdigest()
53:         assert len(ids) == expected['rows'] and order_hash == expected['request_order_sha256']
54:         nulls = scan.select(*[pl.col(c).null_count().alias(c) for c in FEATURES]).collect().row(0, named=True)
55:         assert not any(nulls.values()), nulls
56:         # Separate metadata categories from actual runtime decoding: one native batch per split.
57:         batch = next(iter(DataLoader(ParquetStreamDataset(data/f'{split}.parquet', manager,
58:                      batch_size=8, shuffle=False, drop_last=False, extra_col_names=EXTRA_COLUMNS), batch_size=None)))
59:         assert all(c in batch.interaction for c in FEATURES)
60:         output['splits'][split] = {'rows': len(ids), 'request_order_sha256': order_hash,
61:                                   'feature_count': len(FEATURES), 'null_features': 0,
62:                                   'native_batch_rows': len(batch['request_id'])}
63:         print('VERIFIED', split, len(ids), 'rows; 54 fields; original order preserved', flush=True)
64:     target = ROOT/'feature_categories.json'
65:     target.write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
66:     print('COMPLETE', target, 'v3 reused; req_idx formula explicitly pending', flush=True)
67: 
68: 
69: if __name__ == '__main__':
70:     main()
````

## File: workspace/kr_time_budget/pipeline/build_requests_v3.py
````python
  1: """Ordered raw events -> requests -> temporal features -> native rosen schema."""
  2: import argparse
  3: import gc
  4: import hashlib
  5: import json
  6: from pathlib import Path
  7: import polars as pl
  8: from rosen.datasets.kuairand import KuaiRandDataset
  9: from workspace.kr_time_budget.pipeline.build_requests import sessionize
 10: from workspace.kr_time_budget.pipeline.features import past_daily
 11: from workspace.kr_time_budget.pipeline.kr_request_schema import (request_manager, FEATURES, USER_CATEGORY_FIELDS, LOG1P_FIELDS, EXTRA_COLUMNS)
 12: 
 13: ROOT = Path('workspace/kr_time_budget/data/v3')
 14: TRAIN_END = 20220506
 15: 
 16: 
 17: def build_sessions(events):
 18:     sessions = events.group_by(['user_id', 'session_id'], maintain_order=True).agg(
 19:         pl.col('date').first().alias('session_date'), pl.col('date').max().alias('end_date'),
 20:         (pl.col('hourmin').first() // 100).alias('start_hour'), pl.col('tab').first().alias('tab'),
 21:         ((pl.col('time_ms').last() - pl.col('time_ms').first() + pl.col('play_clip').last()) / 60000).alias('session_min'),
 22:         (pl.len() // 6).cast(pl.Float64).alias('session_req_cnt'), pl.len().alias('session_n'),
 23:         pl.col('first_session').first(), pl.col('last_session').first(),
 24:     ).sort('session_id')
 25:     return sessions.with_columns(
 26:         (pl.col('session_id').cum_count().over(['user_id','session_date']) - 1).alias('day_prev_sess_cnt'),
 27:         (pl.col('session_min').cum_sum().over(['user_id','session_date']) - pl.col('session_min')).alias('day_prev_elapsed_min'))
 28: 
 29: 
 30: def aggregate(events):
 31:     """Head remainder is chunk 0: history only, never a supervised sample."""
 32:     g = events.group_by(['user_id','session_id','req_idx'], maintain_order=True).agg(
 33:         pl.len().alias('n'), pl.col('video_id').alias('slate_items'),
 34:         pl.col('tag').str.join(',').alias('slate_tags'),
 35:         pl.col('tag').filter((pl.col('is_click') == 1) | (pl.col('long_view') == 1)).str.join(',').alias('engaged_tags'),
 36:         pl.col('video_type').str.join(',').alias('slate_video_types'),
 37:         pl.col('music_type').str.join(',').alias('slate_music_types'),
 38:         (pl.col('duration_ms').clip(3000,600000).mean() / 60000).alias('slate_dur_mean_min'),
 39:         pl.col('author_id').filter(pl.col('author_id') >= 0).n_unique().alias('slate_unique_author_cnt'),
 40:         pl.col('is_rand').max().alias('slate_has_rand'),
 41:         pl.col('is_click').sum().alias('click_n'), pl.col('long_view').sum().alias('long_n'),
 42:         pl.col('positive').sum().alias('pos_n'), pl.col('is_hate').sum().alias('hate_n'),
 43:         ((pl.col('time_ms').last() - pl.col('session_start_ms').first() + pl.col('play_clip').last()) / 60000).alias('elapsed_min'),
 44:     ).sort(['session_id','req_idx'])
 45:     for col in ['n','click_n','long_n','pos_n','hate_n']:
 46:         g = g.with_columns((pl.col(col).cum_sum().over('session_id')-pl.col(col)).alias('past_'+col))
 47:     g = g.with_columns(
 48:         *[pl.col(c).shift(1).over('session_id').fill_null(0).alias(out) for c,out in
 49:           [('click_n','prev1_click_cnt'),('long_n','prev1_longview_cnt'),('pos_n','prev1_pos_cnt')]],
 50:         *[(pl.col('past_'+c)/pl.col('past_n').clip(lower_bound=1)).alias(out) for c,out in
 51:           [('click_n','sess_cum_click_rate'),('long_n','sess_cum_longview_rate'),('pos_n','sess_cum_pos_rate')]],
 52:         pl.col('past_hate_n').alias('sess_cum_hate_cnt'), (pl.col('req_idx')==1).cast(pl.Int8).alias('is_first_req'),
 53:         pl.concat_str([pl.col('engaged_tags').shift(2).over('session_id').fill_null(''),
 54:                        pl.col('engaged_tags').shift(1).over('session_id').fill_null('')],separator=',').alias('hist_tags'),
 55:         (pl.col('req_idx') < pl.col('req_idx').max().over('session_id')).cast(pl.Int8).alias('y_continue'),
 56:         pl.col('slate_tags').str.split(',').list.eval(pl.element().filter((pl.element()!='') & (pl.element()!='__MISSING__'))).list.n_unique().alias('slate_unique_tag_cnt'),
 57:     )
 58:     return g.filter(pl.col('req_idx')>0).drop(['engaged_tags','n','click_n','long_n','pos_n','hate_n']+['past_'+c for c in ['n','click_n','long_n','pos_n','hate_n']])
 59: 
 60: 
 61: def user_priors(query, sessions):
 62:     obs = sessions.select('user_id',pl.col('end_date').alias('date'),'session_min','session_req_cnt')
 63:     own = past_daily(query,obs,['user_id'],['session_min','session_req_cnt'])
 64:     glob = past_daily(query.select('request_id','date'),obs,[],['session_min','session_req_cnt']).select('request_id',
 65:         *[(pl.col('sum_'+c)/pl.col('prior_n').clip(lower_bound=1)).alias('global_'+c) for c in ['session_min','session_req_cnt']])
 66:     return own.join(glob,on='request_id',validate='1:1').select('request_id',pl.col('prior_n').alias('user_prior_n'),
 67:         *[((pl.col('sum_'+c)+5*pl.col('global_'+c))/(pl.col('prior_n')+5)).alias(out) for c,out in
 68:           [('session_min','user_hist_session_min'),('session_req_cnt','user_hist_req_cnt')]])
 69: 
 70: 
 71: def item_priors(query, obs):
 72:     vals=['is_click','long_view']
 73:     per = past_daily(query.explode('slate_items').rename({'slate_items':'video_id'}),obs,['video_id'],vals)
 74:     glob = past_daily(query.select('request_id','date'),obs,[],vals).select('request_id',
 75:         *[(pl.col('sum_'+c)/pl.col('prior_n').clip(lower_bound=1)).alias('global_'+c) for c in vals])
 76:     per=per.join(glob,on='request_id').with_columns(
 77:         *[((pl.col('sum_'+c)+20*pl.col('global_'+c))/(pl.col('prior_n')+20)).alias('quality_'+c) for c in vals])
 78:     return per.group_by('request_id').agg(
 79:         pl.col('quality_is_click').mean().alias('slate_prior_ctr_mean'),
 80:         pl.col('quality_long_view').mean().alias('slate_prior_longview_mean'),
 81:         pl.col('prior_n').mean().alias('item_prior_support_mean'),
 82:         (pl.col('prior_n')==0).mean().alias('item_prior_cold_fraction'))
 83: 
 84: 
 85: def encode(raw, out):
 86:     raw=raw.with_columns(*[pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32) for c in LOG1P_FIELDS])
 87:     mgr=request_manager(out/'schema')
 88:     mgr.fit(raw.filter(pl.col('split')=='train').lazy())
 89:     keep=list(dict.fromkeys(mgr.fields()+EXTRA_COLUMNS))
 90:     stats={}
 91:     for split in ['train','valid','test']:
 92:         frame=mgr.transform(raw.filter(pl.col('split')==split).lazy()).select(keep).collect()
 93:         if split=='train': frame=frame.sample(fraction=1,shuffle=True,seed=2026)
 94:         assert frame.null_count().sum_horizontal().item()==0
 95:         frame.write_parquet(out/f'{split}.parquet',row_group_size=65536)
 96:         stats[split]={'rows':frame.height,'users':frame['user_id_raw'].n_unique(),
 97:                       'positive_rate':frame['y_continue'].mean(),
 98:                       'request_order_sha256':hashlib.sha256(frame['request_id'].to_numpy().tobytes()).hexdigest()}
 99:         print('ENCODED',split,stats[split],flush=True)
100:     return stats
101: 
102: 
103: def main():
104:     ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=KuaiRandDataset.DATA_KUAI_1K);ap.add_argument('--output',type=Path,default=ROOT)
105:     args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
106:     if (out/'manifest.json').exists():
107:         print('Already complete',out,flush=True);return
108:     if not (out/'events.parquet').exists():
109:         frames=[];sources=[]
110:         for i,p in enumerate(sorted(args.source.glob('log_*.csv'))):
111:             f=pl.read_csv(p).with_row_index('source_row').with_columns(pl.lit(i).alias('source_order'))
112:             frames.append(f);sources.append({'path':str(p),'rows':f.height,'bytes':p.stat().st_size})
113:             print('READ',p.name,f.height,flush=True)
114:         events=pl.concat(frames);del frames;cols=[c for c in events.columns if c not in ['source_row','source_order']]
115:         raw_n=events.height
116:         events=events.unique(subset=cols,maintain_order=True)
117:         events=sessionize(events).with_row_index('event_id')
118:         events.write_parquet(out/'events.parquet',row_group_size=65536)
119:         (out/'source_manifest.json').write_text(json.dumps({'sources':sources,'raw_rows':raw_n,'deduplicated_rows':events.height,'dedup':'all 19 raw columns'},indent=2))
120:         print('EVENTS',events.height,flush=True);del events;gc.collect()
121:     if not (out/'requests_base.parquet').exists():
122:         events=pl.read_parquet(out/'events.parquet');sessions=build_sessions(events)
123:         sessions.write_parquet(out/'sessions.parquet')
124:         usable=sessions.filter((pl.col('session_id')!=pl.col('first_session'))&(pl.col('session_id')!=pl.col('last_session'))&(pl.col('session_n')>=6))
125:         # Choose scenarios using training-range sessions only, never test counts.
126:         top=usable.filter(pl.col('end_date')<=TRAIN_END).group_by('tab').len().sort(['len','tab'],descending=[True,False]).head(5)['tab'].to_list()
127:         usable=usable.filter(pl.col('tab').is_in(top)).with_columns(
128:             pl.when(pl.col('end_date')<=TRAIN_END).then(pl.lit('train'))
129:             .when((pl.col('session_date')==20220507)&(pl.col('end_date')==20220507)).then(pl.lit('valid'))
130:             .when((pl.col('session_date')==20220508)&(pl.col('end_date')==20220508)).then(pl.lit('test')).otherwise(pl.lit('boundary_excluded')).alias('split'))
131:         excluded=usable.filter(pl.col('split')=='boundary_excluded').height
132:         usable=usable.filter(pl.col('split')!='boundary_excluded')
133:         events=events.join(usable.select('session_id'),on='session_id',how='semi').sort(['session_id','event_idx'])
134:         video=pl.read_csv(args.source/'video_features_basic_1k.csv',columns=['video_id','author_id','tag','video_type','music_type'],schema_overrides={'tag':pl.String})
135:         events=events.join(video,on='video_id',how='left',validate='m:1',maintain_order='left').with_columns(
136:             pl.col('tag').fill_null('__MISSING__'),pl.col('video_type').fill_null('__MISSING__'),
137:             pl.col('music_type').cast(pl.Int64).cast(pl.String).fill_null('__MISSING__'),pl.col('author_id').fill_null(-1),
138:             pl.any_horizontal(*[pl.col(c)==1 for c in ['is_like','is_comment','is_follow','is_forward']]).cast(pl.Int8).alias('positive'))
139:         req=aggregate(events).join(usable,on=['user_id','session_id'],how='left',validate='m:1').with_row_index('request_id')
140:         # Persist event membership, including request-0 remainder, without duplicating event payloads.
141:         events.select('event_id','user_id','session_id','req_idx','event_idx').join(req.select('session_id','req_idx','request_id'),on=['session_id','req_idx'],how='left').write_parquet(out/'request_events.parquet')
142:         del events,video;gc.collect()
143:         users=pl.read_csv(args.source/'user_features_1k.csv').select('user_id',*USER_CATEGORY_FIELDS).with_columns(
144:             pl.col('is_live_streamer').eq(1).cast(pl.Int8),
145:             *[pl.col(f'onehot_feat{i}').cast(pl.Int64).fill_null(-1) for i in range(18)])
146:         req=req.join(users,on='user_id',how='left',validate='m:1').with_columns(
147:             pl.col('user_id').alias('user_id_raw'),pl.col('tab').alias('tab_raw'),pl.col('req_idx').alias('req_idx_raw'),
148:             pl.col('req_idx').alias('intra_step_group'),
149:             pl.col('session_date').cast(pl.String).str.strptime(pl.Date,'%Y%m%d').dt.weekday().alias('day_of_week'),
150:             (pl.col('session_date').cast(pl.String).str.strptime(pl.Date,'%Y%m%d').dt.weekday()>=6).cast(pl.Int8).alias('is_weekend'),
151:         ).with_columns(pl.struct('user_active_degree','req_idx').rank('dense').alias('active_step_group'))
152:         req.write_parquet(out/'requests_base.parquet')
153:         (out/'structure_manifest.json').write_text(json.dumps({'top_tabs':top,'sessions_total':sessions.height,'sessions_retained':usable.height,'boundary_sessions_excluded':excluded,'requests':req.height},indent=2))
154:         print('REQUESTS',req.height,'sessions',usable.height,'top',top,flush=True);del req;gc.collect()
155:     req=pl.read_parquet(out/'requests_base.parquet');sessions=pl.read_parquet(out/'sessions.parquet')
156:     # All held-out priors are frozen at May 6 end; train queries exclude the entire query day.
157:     query=req.select('request_id','user_id',pl.when(pl.col('split')=='train').then(pl.col('session_date')).otherwise(pl.lit(20220507)).alias('date'))
158:     req=req.join(user_priors(query,sessions),on='request_id',validate='1:1')
159:     print('USER PRIORS complete',flush=True)
160:     obs=pl.read_parquet(out/'events.parquet',columns=['video_id','date','is_click','long_view']).filter(pl.col('date')<=TRAIN_END)
161:     item_query=query.select('request_id','date').join(req.select('request_id','slate_items'),on='request_id',validate='1:1')
162:     req=req.join(item_priors(item_query,obs),on='request_id',validate='1:1')
163:     del obs;gc.collect();print('ITEM PRIORS complete',flush=True)
164:     req.write_parquet(out/'features_raw.parquet',row_group_size=65536)
165:     splits=encode(req,out)
166:     train_users=req.filter(pl.col('split')=='train')['user_id'].unique()
167:     audit={s:{'unseen_users':req.filter((pl.col('split')==s)&~pl.col('user_id').is_in(train_users.implode()))['user_id'].n_unique(),
168:               'user_history_coverage':req.filter(pl.col('split')==s).select((pl.col('user_prior_n')>0).mean()).item(),
169:               'item_cold_fraction':req.filter(pl.col('split')==s)['item_prior_cold_fraction'].mean()} for s in ['train','valid','test']}
170:     tags={c:{'cap':cap,'truncated_fraction':req.select((pl.col(c).str.split(',').list.eval(pl.element().filter(pl.element()!='')).list.len()>cap).mean()).item()} for c,cap in [('slate_tags',12),('hist_tags',15)]}
171:     meta={'version':3,'features':FEATURES,'splits':splits,'audit':audit,'tags':tags,
172:           'protocol':{'train':'session ends <=20220506','valid':'whole session on 20220507','test':'whole session on 20220508','session_tab':'first event','top_tabs':'top5 train sessions',
173:                       'train_priors':'strictly earlier calendar days; users completed sessions; items all observed exposures','eval_priors':'frozen through 20220506','user_smoothing':5,'item_smoothing':20,
174:                       'positive':'any like/comment/follow/forward per event','tags':'repetitions retained; chronological order; head truncation; sum aggregation; independent hist/slate vocab and tables',
175:                       'head_remainder':'chunk0 history only; participates in previous-chunk stats and last2chunk tags','duration_clip_ms':[3000,600000],'clock':'Leader screen end; delta1min',
176:                       'content_elapsed':'full-vector content path; excluded only from budget parameter net'},
177:           'source':json.loads((out/'source_manifest.json').read_text()),'structure':json.loads((out/'structure_manifest.json').read_text())}
178:     (out/'manifest.json').write_text(json.dumps(meta,indent=2));print('DATA READY',json.dumps(meta['splits']),flush=True)
179: 
180: if __name__=='__main__':main()
````

## File: workspace/kr_time_budget/pipeline/build_requests.py
````python
  1: """KR session/request reconstruction. datasource is read-only."""
  2: import argparse
  3: import json
  4: from pathlib import Path
  5: import polars as pl
  6: import numpy as np
  7: from sklearn.model_selection import GroupKFold
  8: from rosen.datasets.kuairand import KuaiRandDataset
  9: 
 10: ROOT = Path('workspace/kr_time_budget/data/v1')
 11: SOURCE = KuaiRandDataset.DATA_KUAI_1K
 12: 
 13: 
 14: def sessionize(df: pl.DataFrame) -> pl.DataFrame:
 15:     df = df.sort(['user_id', 'time_ms', 'source_order', 'source_row'])
 16:     df = df.with_columns((pl.col('play_time_ms').clip(0, 300000)).alias('play_clip'))
 17:     gap = pl.col('time_ms') - (pl.col('time_ms').shift(1).over('user_id') + pl.col('play_clip').shift(1).over('user_id'))
 18:     df = df.with_columns((gap.is_null() | (gap > 1800000)).cast(pl.Int64).alias('new_session'))
 19:     df = df.with_columns(pl.col('new_session').cum_sum().alias('session_id'))
 20:     return df.with_columns(
 21:         pl.len().over('session_id').alias('session_n'),
 22:         pl.int_range(pl.len()).over('session_id').alias('event_idx'),
 23:         pl.col('time_ms').first().over('session_id').alias('session_start_ms'),
 24:         pl.col('session_id').first().over('user_id').alias('first_session'),
 25:         pl.col('session_id').last().over('user_id').alias('last_session'),
 26:     ).with_columns(
 27:         ((pl.col('event_idx').cast(pl.Int64) - (pl.col('session_n') % 6).cast(pl.Int64)) // 6 + 1).alias('req_idx')
 28:     )
 29: 
 30: 
 31: def aggregate_requests(events: pl.DataFrame) -> pl.DataFrame:
 32:     """req_idx=0 holds the head remainder and participates only in history."""
 33:     df = events.group_by(['user_id', 'session_id', 'req_idx'], maintain_order=True).agg(
 34:         pl.len().alias('n_items'),
 35:         pl.col('video_id').alias('slate_items'), pl.col('tag').alias('slate_tags'),
 36:         pl.col('dur_s').alias('slate_durations'),
 37:         pl.col('dur_s').sum().alias('slate_dur_sum_s'),
 38:         pl.col('tag').n_unique().alias('slate_unique_tag_cnt'),
 39:         pl.col('author_id').n_unique().alias('slate_unique_author_cnt'),
 40:         pl.col('is_rand').max().alias('slate_has_rand'),
 41:         pl.col('time_ms').first().alias('request_start_ms'),
 42:         ((pl.col('time_ms').last() - pl.col('session_start_ms').first() + pl.col('play_clip').last()) / 1000).alias('elapsed_end_s_DIAGNOSTIC'),
 43:         ((pl.col('time_ms').first() - pl.col('session_start_ms').first()) / 1000).alias('elapsed_time_s'),
 44:         (pl.col('play_time_ms').clip(lower_bound=0).sum() / 1000).alias('curr_watch_s'),
 45:         pl.col('comp').sum().alias('comp_sum'), pl.col('long_view').sum().alias('long_sum'),
 46:         pl.col('pos').sum().alias('pos_sum'), pl.col('is_hate').sum().alias('hate_sum'),
 47:         *[pl.col(c).first() for c in ['date', 'session_date', 'session_hour', 'session_tab', 'session_start_ms', 'session_wall_s', 'session_requests', 'day_prev_sess_cnt', 'day_prev_time_s']],
 48:     ).sort(['session_id', 'req_idx'])
 49:     for c in ['curr_watch_s', 'comp_sum', 'pos_sum', 'hate_sum', 'n_items']:
 50:         df = df.with_columns((pl.col(c).cum_sum().over('session_id') - pl.col(c)).alias('past_' + c))
 51:     df = df.with_columns(
 52:         pl.col('curr_watch_s').shift(1).over('session_id').fill_null(0).alias('prev1_play_s'),
 53:         (pl.col('comp_sum') / pl.col('n_items')).shift(1).over('session_id').fill_null(0).alias('prev1_comp_rate'),
 54:         (pl.col('long_sum') / pl.col('n_items')).shift(1).over('session_id').fill_null(0).alias('prev1_long_view_rate'),
 55:         pl.col('pos_sum').shift(1).over('session_id').fill_null(0).alias('prev1_pos_cnt'),
 56:         (pl.col('past_n_items') == 0).cast(pl.Int8).alias('is_first_req'),
 57:         (pl.col('past_comp_sum') / pl.col('past_n_items').clip(lower_bound=1)).alias('sess_cum_comp_rate'),
 58:         (pl.col('past_pos_sum') / pl.col('past_n_items').clip(lower_bound=1)).alias('sess_cum_pos_rate'),
 59:         pl.col('past_hate_sum').alias('sess_cum_hate_cnt'),
 60:         pl.col('curr_watch_s').reverse().cum_sum().reverse().over('session_id').alias('remaining_watch_s'),
 61:         (pl.col('req_idx') < pl.col('req_idx').max().over('session_id')).cast(pl.Int8).alias('y_continue'),
 62:     )
 63:     # The previous two chunks, including a possible head remainder; <=12 events.
 64:     for raw, out in [('slate_items', 'history_items'), ('slate_tags', 'history_tags')]:
 65:         df = df.with_columns(pl.concat_list([
 66:             pl.col(raw).shift(2).over('session_id').fill_null([]),
 67:             pl.col(raw).shift(1).over('session_id').fill_null([]),
 68:         ]).alias(out))
 69:     return df.filter(pl.col('req_idx') > 0).with_columns(
 70:         (pl.col('remaining_watch_s') - pl.col('curr_watch_s')).alias('future_watch_s')
 71:     ).drop(['comp_sum', 'long_sum', 'pos_sum', 'hate_sum'] + ['past_' + c for c in ['curr_watch_s', 'comp_sum', 'pos_sum', 'hate_sum', 'n_items']])
 72: 
 73: 
 74: def main():
 75:     ap = argparse.ArgumentParser()
 76:     ap.add_argument('--source', type=Path, default=SOURCE)
 77:     ap.add_argument('--output', type=Path, default=ROOT)
 78:     args = ap.parse_args()
 79:     out = args.output
 80:     out.mkdir(parents=True, exist_ok=True)
 81:     if (out / 'requests.parquet').exists() or (out / 'manifest.json').exists():
 82:         raise FileExistsError(f'{out} already contains a dataset; use a fresh --output to avoid stale fold schemas')
 83:     paths = sorted(args.source.glob('log_*.csv'))
 84:     cols = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'play_time_ms', 'duration_ms', 'tab', 'is_like', 'is_comment', 'is_forward', 'is_follow', 'is_hate', 'long_view', 'is_rand']
 85:     frames = []
 86:     for i, p in enumerate(paths):
 87:         f = pl.read_csv(p, columns=cols).with_row_index('source_row').with_columns(pl.lit(i).alias('source_order'))
 88:         print(p.name, f.height, flush=True)
 89:         frames.append(f)
 90:     df = pl.concat(frames)
 91:     del frames
 92:     raw_rows = df.height
 93:     # Deduplicate the modeled raw projection; omitted raw feedback columns are
 94:     # not part of this key. Record this rule rather than calling it byte-exact.
 95:     df = df.unique(subset=cols, maintain_order=True)
 96:     dedup_rows = df.height
 97:     df = sessionize(df)
 98:     sessions = df.group_by(['user_id', 'session_id'], maintain_order=True).agg(
 99:         pl.col('date').first().alias('session_date'),
100:         (pl.col('hourmin').first() // 100).alias('session_hour'),
101:         pl.col('tab').first().alias('session_tab'),
102:         ((pl.col('time_ms').last() - pl.col('time_ms').first() + pl.col('play_clip').last()) / 1000).alias('session_wall_s'),
103:         (pl.len() // 6).alias('session_requests'),
104:         pl.col('date').max().cast(pl.String).str.strptime(pl.Date, '%Y%m%d').dt.offset_by('1d').dt.strftime('%Y%m%d').cast(pl.Int64).alias('prior_available_date'),
105:         pl.col('first_session').first(), pl.col('last_session').first(), pl.len().alias('session_n'),
106:     ).sort('session_id')
107:     sessions = sessions.with_columns(
108:         (pl.col('session_id').cum_count().over(['user_id', 'session_date']) - 1).alias('day_prev_sess_cnt'),
109:         (pl.col('session_wall_s').cum_sum().over(['user_id', 'session_date']) - pl.col('session_wall_s')).alias('day_prev_time_s'),
110:     )
111:     sessions.write_parquet(out / 'sessions.parquet')
112:     print('sessions', sessions.height, 'events', df.height, flush=True)
113:     df = df.join(sessions.drop(['session_n', 'first_session', 'last_session']), on=['user_id', 'session_id'], how='left', validate='m:1')
114:     video = pl.read_csv(args.source / 'video_features_basic_1k.csv', columns=['video_id', 'author_id', 'tag'], schema_overrides={'tag': pl.String})
115:     assert video['video_id'].n_unique() == video.height
116:     df = df.join(video, on='video_id', how='left', validate='m:1').with_columns(
117:         pl.col('tag').fill_null('__MISSING__'), pl.col('author_id').fill_null(-1),
118:         (pl.col('duration_ms') / 1000).clip(3, 600).alias('dur_s'),
119:         pl.sum_horizontal('is_like', 'is_comment', 'is_forward', 'is_follow').alias('pos'),
120:     ).with_columns((pl.col('play_clip') / (pl.col('dur_s') * 1000)).clip(0, 1).alias('comp'))
121:     # Retain all observed events for strictly earlier-day, training-user priors.
122:     df.select(['user_id', 'video_id', 'date', 'session_id', 'req_idx', 'comp', 'long_view', 'is_like']).write_parquet(out / 'prior_events.parquet')
123:     # First/last observed sessions can be left/right censored at observation edges.
124:     usable = df.filter((pl.col('session_id') != pl.col('first_session')) & (pl.col('session_id') != pl.col('last_session')) & (pl.col('session_n') >= 6))
125:     req = aggregate_requests(usable)
126:     del df, usable
127:     users = pl.read_csv(args.source / 'user_features_1k.csv')
128:     keep = ['user_id', 'user_active_degree', 'is_live_streamer', 'is_video_author'] + [f'onehot_feat{i}' for i in range(18)] + ['register_days', 'fans_user_num', 'follow_user_num', 'friend_user_num']
129:     users = users.select(keep).with_columns(
130:         pl.col('is_live_streamer').eq(1).cast(pl.Int8),
131:         *[pl.col(f'onehot_feat{i}').fill_null(-1 if i < 12 else 0) for i in range(18)],
132:     )
133:     req = req.join(users, on='user_id', how='left', validate='m:1')
134:     user_values = req['user_id'].to_numpy()
135:     folds = np.empty(req.height, dtype=np.int8)
136:     for i, (_, idx) in enumerate(GroupKFold(5).split(user_values, groups=user_values)):
137:         folds[idx] = i
138:     req = req.with_columns(pl.Series('fold', folds)).with_row_index('request_id')
139:     assert req.null_count().sum_horizontal().item() == 0
140:     assert req.filter(pl.col('n_items') != 6).height == 0
141:     req.write_parquet(out / 'requests.parquet', row_group_size=65536)
142:     stats = {
143:         'raw_rows': raw_rows, 'deduplicated_rows': dedup_rows, 'all_sessions': sessions.height,
144:         'retained_sessions': req['session_id'].n_unique(), 'request_rows': req.height,
145:         'users': req['user_id'].n_unique(), 'y_continue_mean': req['y_continue'].mean(),
146:         'fold_rows': req.group_by('fold').len().sort('fold').to_dicts(),
147:         'sources': [{'path': str(p.resolve()), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths],
148:         'protocol': '30min idle gap after clipped preceding play; M=6 reverse chunks; head remainder history; first/last session per user excluded; pre-exposure wall clock; end-clock diagnostic only',
149:         'history_cap': 12, 'duration_clip_s': [3, 600], 'play_clip_ms_for_boundary_and_completion': [0, 300000],
150:         'watch_labels': 'sum raw nonnegative play_time_ms / 1000; no upper clip',
151:     }
152:     (out / 'manifest.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
153:     print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
154: 
155: if __name__ == '__main__':
156:     main()
````

## File: workspace/kr_time_budget/pipeline/features.py
````python
 1: """Fold-local, strictly earlier-calendar-day quality and budget priors."""
 2: import json
 3: from pathlib import Path
 4: import polars as pl
 5: from rosen.emb import SchemaManager
 6: from rosen.emb.schema import MinMaxDenseSetting, SparseEmbSetting, SequenceSetting, SeqGroupConfig, VectorDenseSetting
 7: from rosen.core.enum_type import FeatureSource as FS
 8: 
 9: B1 = ['register_days', 'fans_user_num', 'follow_user_num', 'friend_user_num', 'user_hist_session_s', 'user_hist_session_requests']
10: B2 = ['hour_sin', 'hour_cos', 'is_weekend', 'day_prev_sess_cnt', 'day_prev_time_s', 'elapsed_log', 'req_idx']
11: B3 = ['is_first_req', 'prev1_play_s', 'prev1_comp_rate', 'prev1_long_view_rate', 'prev1_pos_cnt', 'sess_cum_comp_rate', 'sess_cum_pos_rate', 'sess_cum_hate_cnt']
12: B4 = ['slate_dur_sum_s', 'slate_unique_tag_cnt', 'slate_unique_author_cnt', 'slate_has_rand'] + ['prior_' + c + '_' + a for c in ['comp', 'long_view', 'is_like'] for a in ['mean', 'min', 'max']]
13: CATS = ['user_active_degree', 'is_live_streamer', 'is_video_author'] + [f'onehot_feat{i}' for i in range(12)] + ['session_tab']
14: B1 += [f'onehot_feat{i}' for i in range(12, 18)]
15: NUMS = B1 + B2 + B3 + B4
16: LOG_COLS = ['register_days', 'fans_user_num', 'follow_user_num', 'friend_user_num', 'user_hist_session_s', 'user_hist_session_requests', 'day_prev_sess_cnt', 'day_prev_time_s', 'req_idx', 'prev1_play_s', 'prev1_pos_cnt', 'sess_cum_hate_cnt', 'slate_dur_sum_s']
17: EXTRAS = ['request_id', 'user_id', 'session_id', 'y_continue', 'elapsed_minutes', 'delta_minutes', 'curr_watch_s', 'future_watch_s', 'remaining_watch_s']
18: 
19: 
20: def past_daily(query, observations, keys, values):
21:     """Cumulative sums/count through the previous day; current day excluded."""
22:     by = keys + ['date']
23:     daily = observations.group_by(by).agg(pl.len().cast(pl.Float64).alias('prior_n'), *[pl.col(v).sum().alias('sum_' + v) for v in values]).sort(by)
24:     sums = ['prior_n'] + ['sum_' + v for v in values]
25:     if keys:
26:         daily = daily.with_columns(*[pl.col(c).cum_sum().over(keys).alias(c) for c in sums])
27:     else:
28:         daily = daily.with_columns(*[pl.col(c).cum_sum().alias(c) for c in sums])
29:     return query.sort('date').join_asof(daily.sort('date'), on='date', by=keys or None, strategy='backward', allow_exact_matches=False, check_sortedness=False).with_columns(pl.col(sums).fill_null(0))
30: 
31: 
32: def make_manager(path: Path, req=None):
33:     settings = [MinMaxDenseSetting(c, FS.INTERACTION) for c in NUMS]
34:     settings += [SparseEmbSetting(c, FS.USER, embedding_dim=4) for c in CATS]
35:     for name, cap in [('slate_items', 6), ('slate_tags', 6), ('history_items', 12), ('history_tags', 12)]:
36:         settings.append(SequenceSetting(name, SparseEmbSetting(name, FS.ITEM, embedding_dim=8, min_freq=5 if 'items' in name else 1), SeqGroupConfig(name, name + '_len', cap)))
37:     settings.append(VectorDenseSetting('slate_durations', FS.INTERACTION, embedding_dim=6))
38:     mgr = SchemaManager(settings, str(path), label_fields='y_continue')
39:     if req is None:
40:         mgr.load_schema()
41:     return mgr
42: 
43: 
44: def prepare_fold(root: Path, fold: int, seed=2026):
45:     out = root / f'fold_{fold}'
46:     out.mkdir(exist_ok=True)
47:     if (out / 'manifest.json').exists():
48:         saved = json.loads((out / 'manifest.json').read_text())
49:         if saved['seed'] != seed:
50:             raise ValueError('Fold physical shuffle uses another seed; use a fresh dataset directory')
51:         return make_manager(out / 'schema'), out
52:     req = pl.read_parquet(root / 'requests.parquet')
53:     train_users = req.filter(pl.col('fold') != fold).select('user_id').unique()
54:     test_users = req.filter(pl.col('fold') == fold)['user_id'].unique().to_list()
55:     obs = pl.read_parquet(root / 'prior_events.parquet').join(train_users, on='user_id', how='semi')
56:     # Priors for each slate video, using only training users and days < request day.
57:     q = req.select('request_id', 'date', 'slate_items').explode('slate_items').rename({'slate_items': 'video_id'})
58:     vals = ['comp', 'long_view', 'is_like']
59:     q = past_daily(q, obs, ['video_id'], vals)
60:     global_q = past_daily(req.select('request_id', 'date'), obs, [], vals).select('request_id', *[(pl.col('sum_' + v) / pl.col('prior_n').clip(lower_bound=1)).alias('global_' + v) for v in vals])
61:     q = q.join(global_q, on='request_id', how='left').with_columns(*[((pl.col('sum_' + v) + 20 * pl.col('global_' + v)) / (pl.col('prior_n') + 20)).alias('prior_' + v) for v in vals])
62:     agg = q.group_by('request_id').agg(*[getattr(pl.col('prior_' + v), a)().alias('prior_' + v + '_' + a) for v in vals for a in ['mean', 'min', 'max']])
63:     req = req.join(agg, on='request_id', how='left')
64:     del obs, q, agg
65:     # Training users: their own prior-day sessions; held-out users: training-population fallback.
66:     sess = pl.read_parquet(root / 'sessions.parquet').join(train_users, on='user_id', how='semi').rename({'prior_available_date': 'date'})
67:     vals = ['session_wall_s', 'session_requests']
68:     uq = past_daily(req.select('request_id', 'date', 'user_id'), sess, ['user_id'], vals)
69:     gq = past_daily(req.select('request_id', 'date'), sess, [], vals).select('request_id', *[(pl.col('sum_' + v) / pl.col('prior_n').clip(lower_bound=1)).alias('global_' + v) for v in vals])
70:     uq = uq.join(gq, on='request_id').select('request_id', *[pl.when(pl.col('prior_n') > 0).then(pl.col('sum_' + v) / pl.col('prior_n')).otherwise(pl.col('global_' + v)).alias(n) for v, n in zip(vals, ['user_hist_session_s', 'user_hist_session_requests'])])
71:     req = req.join(uq, on='request_id')
72:     req = req.with_columns(
73:         (pl.col('session_hour') * (2 * 3.141592653589793 / 24)).sin().alias('hour_sin'),
74:         (pl.col('session_hour') * (2 * 3.141592653589793 / 24)).cos().alias('hour_cos'),
75:         (pl.col('session_date').cast(pl.String).str.strptime(pl.Date, '%Y%m%d').dt.weekday() >= 6).cast(pl.Float32).alias('is_weekend'),
76:         pl.col('elapsed_time_s').log1p().alias('elapsed_log'),
77:         (pl.col('elapsed_time_s') / 60).cast(pl.Float32).alias('elapsed_minutes'),
78:         (pl.col('slate_dur_sum_s') / 60).cast(pl.Float32).alias('delta_minutes'),
79:         pl.when(pl.col('session_tab').is_in([0, 1, 2, 4, 6])).then(pl.col('session_tab')).otherwise(99).alias('session_tab'),
80:         pl.col('slate_durations').list.eval(pl.element().log1p() / 6.4),
81:     ).with_columns(*[pl.col(c).clip(lower_bound=0).log1p().alias(c) for c in LOG_COLS])
82:     mgr = make_manager(out / 'schema', req)
83:     mgr.fit(req.filter(pl.col('fold') != fold).lazy())
84:     keep = list(dict.fromkeys(mgr.fields() + EXTRAS))
85:     for split, pred in [('train', pl.col('fold') != fold), ('test', pl.col('fold') == fold)]:
86:         part = mgr.transform(req.filter(pred).lazy()).select(keep).collect()
87:         if split == 'train':
88:             part = part.sample(fraction=1, shuffle=True, seed=seed)
89:         part.write_parquet(out / f'{split}.parquet', row_group_size=65536)
90:         print('fold', fold, split, part.height, flush=True)
91:     meta = {'fold': fold, 'train_rows': req.filter(pl.col('fold') != fold).height, 'test_rows': req.filter(pl.col('fold') == fold).height, 'test_users': sorted(test_users), 'priors': 'training-users only; strictly earlier calendar days; item smoothing strength=20; unseen users use prior-day training population', 'features': {'block1': B1 + CATS[:-1], 'block2': B2 + ['session_tab'], 'block3': B3 + ['history_items', 'history_tags'], 'block4': B4 + ['slate_items', 'slate_tags', 'slate_durations']}, 'seed': seed}
92:     (out / 'manifest.json').write_text(json.dumps(meta, indent=2))
93:     return mgr, out
````

## File: workspace/kr_time_budget/pipeline/kr_request_schema.py
````python
  1: """KR Request 特征集与Rosen编码配置。
  2: 
  3: 特征分类：54字段 = 30共享背景 + 6预算与时钟 + 18内容。
  4:     用户画像26 + 开场上下文4：两塔共享，支持用户/场景与内容的交叉。
  5:     预算先验4：历史场均分钟数/屏数、当天开场前场次/耗时，仅供预算塔。
  6:     会话时钟2：elapsed_min（分钟）、req_idx（屏），仅归预算公式，不进内容塔。
  7:     内容历史9 + 当屏列表9：仅供内容塔；历史只使用当前Request之前的信息。
  8: 
  9: 目标路由：预算参数网34字段，内容塔48字段，Fusion使用全部54字段。
 10: 时钟归属不等于公式已接入：现有v5只在公式使用elapsed_min，req_idx尚未接入。
 11: BUDGET_SCOPE_FIELDS表示36字段的目标预算归属。历史模型仍显式保留自己的路由。
 12: 
 13: 处理理念：
 14:     1. 业务分类与编码类型分开；FeatureSource不决定字段进入哪座塔。
 15:     2. Schema仅在train上fit，valid/test加载同一词表；历史先验遵守时间边界。
 16:     3. 上游仅对LOG1P_FIELDS做log1p；当前数值Setting固定[0,1]，只透传Float32。
 17:        网络缩放与公式原始量必须分离；不能把对数分钟或屏数当作原始分钟。
 18:     4. 标签与审计列放EXTRA_COLUMNS，不因被Dataset读取而成为模型特征。
 19:     5. 多标签保留重复、按原顺序截断并sum聚合；空历史保留空列表、padding为0。
 20:     6. FEATURES及settings顺序属于现有数据/checkpoint契约，重命名不改变其顺序。
 21: """
 22: from rosen.core.enum_type import FeatureSource as FS
 23: from rosen.emb import SchemaManager
 24: from rosen.emb.schema import SparseEmbSetting, MultiSparseSetting, MinMaxDenseSetting
 25: 
 26: 
 27: # 共享背景：用户画像26 + 开场上下文4。
 28: USER_CATEGORY_FIELDS = [
 29:     'user_active_degree', 'is_live_streamer', 'is_video_author',
 30:     'friend_user_num_range', 'register_days_range',
 31:     'follow_user_num_range', 'fans_user_num_range',
 32: ] + [f'onehot_feat{i}' for i in range(18)]
 33: USER_PROFILE_FIELDS = ['user_id'] + USER_CATEGORY_FIELDS
 34: SESSION_CONTEXT_FIELDS = ['tab', 'day_of_week', 'is_weekend', 'start_hour']
 35: SHARED_FIELDS = USER_PROFILE_FIELDS + SESSION_CONTEXT_FIELDS
 36: 
 37: # 预算先验4与时钟2：两种时钟量纲分离，不传给内容塔。
 38: USER_BUDGET_PRIOR_FIELDS = ['user_hist_session_min', 'user_hist_req_cnt']
 39: DAY_BUDGET_PRIOR_FIELDS = ['day_prev_sess_cnt', 'day_prev_elapsed_min']
 40: BUDGET_PRIOR_FIELDS = USER_BUDGET_PRIOR_FIELDS + DAY_BUDGET_PRIOR_FIELDS
 41: SESSION_CLOCK_FIELDS = ['elapsed_min', 'req_idx']
 42: BUDGET_EXCLUSIVE_FIELDS = BUDGET_PRIOR_FIELDS + SESSION_CLOCK_FIELDS
 43: 
 44: # 前序内容历史9：反馈只取前序Request，hist_tags取前两屏。
 45: HISTORY_NUMERIC_FIELDS = [
 46:     'prev1_click_cnt', 'prev1_longview_cnt', 'prev1_pos_cnt',
 47:     'sess_cum_click_rate', 'sess_cum_longview_rate', 'sess_cum_pos_rate',
 48:     'sess_cum_hate_cnt',
 49: ]
 50: CONTENT_HISTORY_FIELDS = ['is_first_req'] + HISTORY_NUMERIC_FIELDS + ['hist_tags']
 51: 
 52: # 当屏候选列表9：静态属性和过去日物品先验，不含当屏真实反馈。
 53: SLATE_NUMERIC_FIELDS = [
 54:     'slate_prior_ctr_mean', 'slate_prior_longview_mean', 'slate_dur_mean_min',
 55:     'slate_unique_tag_cnt', 'slate_unique_author_cnt',
 56: ]
 57: CONTENT_SLATE_FIELDS = [
 58:     'slate_tags', 'slate_video_types', 'slate_music_types',
 59: ] + SLATE_NUMERIC_FIELDS + ['slate_has_rand']
 60: CONTENT_EXCLUSIVE_FIELDS = CONTENT_HISTORY_FIELDS + CONTENT_SLATE_FIELDS
 61: 
 62: # 业务分类与目标路由；历史Content-only基线使用CONTENT_EXCLUSIVE_FIELDS。
 63: FEATURE_CATEGORIES = {
 64:     'shared': SHARED_FIELDS,
 65:     'budget_exclusive': BUDGET_EXCLUSIVE_FIELDS,
 66:     'content_exclusive': CONTENT_EXCLUSIVE_FIELDS,
 67: }
 68: BUDGET_PARAMETER_FIELDS = SHARED_FIELDS + BUDGET_PRIOR_FIELDS
 69: BUDGET_SCOPE_FIELDS = BUDGET_PARAMETER_FIELDS + SESSION_CLOCK_FIELDS
 70: CONTENT_TOWER_FIELDS = SHARED_FIELDS + CONTENT_EXCLUSIVE_FIELDS
 71: CONTENT_MASK_FIELDS = frozenset(BUDGET_EXCLUSIVE_FIELDS)
 72: 
 73: # 兼容已有parquet/checkpoint的54字段顺序；Fusion直接使用FEATURES。
 74: FEATURES = (
 75:     USER_PROFILE_FIELDS + USER_BUDGET_PRIOR_FIELDS
 76:     + SESSION_CONTEXT_FIELDS + DAY_BUDGET_PRIOR_FIELDS + SESSION_CLOCK_FIELDS
 77:     + CONTENT_HISTORY_FIELDS + CONTENT_SLATE_FIELDS
 78: )
 79: LOG1P_FIELDS = BUDGET_PRIOR_FIELDS
 80: EXTRA_COLUMNS = [
 81:     'request_id', 'session_id', 'user_id_raw', 'tab_raw', 'req_idx_raw',
 82:     'y_continue', 'elapsed_min', 'intra_step_group', 'active_step_group',
 83: ]
 84: 
 85: assert [len(fields) for fields in FEATURE_CATEGORIES.values()] == [30, 6, 18]
 86: assert sum(len(fields) for fields in FEATURE_CATEGORIES.values()) == len(set(FEATURES)) == 54
 87: assert set().union(*(set(fields) for fields in FEATURE_CATEGORIES.values())) == set(FEATURES)
 88: assert len(BUDGET_PARAMETER_FIELDS) == 34 and len(CONTENT_TOWER_FIELDS) == 48
 89: assert set(BUDGET_SCOPE_FIELDS) & set(CONTENT_TOWER_FIELDS) == set(SHARED_FIELDS)
 90: 
 91: 
 92: def request_manager(path, load=False):
 93:     """按固定顺序构建54字段Schema；load=True时复用已有train-fit状态。"""
 94:     settings = [SparseEmbSetting('user_id', FS.USER_ID, 16, min_freq=10, use_oov=True)]
 95:     settings += [
 96:         SparseEmbSetting(name, FS.USER, 8, min_freq=10, use_oov=True)
 97:         for name in USER_CATEGORY_FIELDS
 98:     ]
 99:     settings += [
100:         SparseEmbSetting(name, FS.INTERACTION, 8, padding_zero=name != 'tab', use_null=name != 'tab')
101:         for name in SESSION_CONTEXT_FIELDS
102:     ]
103:     settings += [
104:         SparseEmbSetting(name, FS.INTERACTION, 4)
105:         for name in ['is_first_req', 'slate_has_rand']
106:     ]
107:     settings += [
108:         MultiSparseSetting(
109:             name, FS.ITEM, dim, max_tag_len=cap,
110:             is_string_format=True, separator=',', agg='sum',
111:         )
112:         for name, dim, cap in [
113:             ('hist_tags', 16, 15), ('slate_tags', 16, 12),
114:             ('slate_video_types', 8, 6), ('slate_music_types', 8, 6),
115:         ]
116:     ]
117:     # 固定[0,1]仅透传Float32；LOG1P_FIELDS的变换由上游执行。
118:     numeric_fields = LOG1P_FIELDS + SESSION_CLOCK_FIELDS + HISTORY_NUMERIC_FIELDS + SLATE_NUMERIC_FIELDS
119:     settings += [
120:         MinMaxDenseSetting(name, FS.INTERACTION, min_val=0., max_val=1.)
121:         for name in numeric_fields
122:     ]
123:     assert len(settings) == len(FEATURES) == 54
124:     manager = SchemaManager(settings, str(path), label_fields='y_continue', domain_fields='tab')
125:     if load:
126:         manager.load_schema()
127:     return manager
````

## File: workspace/kr_time_budget/pipeline/run_exit_v3.py
````python
  1: """Argon 4 date-split comparison using native rosen Trainer/TwoStageRunner."""
  2: import argparse
  3: import hashlib
  4: import json
  5: import math
  6: from pathlib import Path
  7: from types import SimpleNamespace
  8: import numpy as np
  9: import polars as pl
 10: import torch
 11: from torch.utils.data import DataLoader
 12: from rosen.core.train import EarlyStopper
 13: from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
 14: from rosen.core.trainers import BaseTrainer, TwoStageRunner
 15: from rosen.evaluate.evaluator import Evaluator
 16: from rosen.evaluate.manager import EvaluatorManager, DomainFilter
 17: from rosen.data.dataset import ParquetStreamDataset
 18: from rosen.utils.optimize import split_model_params_by_decay
 19: from workspace.kr_time_budget.pipeline.run_exit import state_hash, grouped_auc, on_device
 20: from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, EXTRA_COLUMNS
 21: from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3
 22: 
 23: 
 24: def load_batches(path,manager,size,smoke=False):
 25:     loader=DataLoader(ParquetStreamDataset(path,manager,batch_size=size,shuffle=False,drop_last=False,extra_col_names=EXTRA_COLUMNS),batch_size=None,num_workers=0)
 26:     batches=[]
 27:     for b in loader:
 28:         batches.append(b)
 29:         if smoke:break
 30:     return batches
 31: 
 32: 
 33: def budget_hash(model):
 34:     digest=hashlib.sha256()
 35:     for module in model.budget_modules:
 36:         digest.update(state_hash(module).encode())
 37:     return digest.hexdigest()
 38: 
 39: 
 40: class ExitTrainer(BaseTrainer):
 41:     """Keep core lifecycle; add within-epoch validation and restore the selected state."""
 42:     def __init__(self,*args,**kwargs):
 43:         super().__init__(*args,**kwargs)
 44:         self.output=Path(self.cfg.ckpt_dir);self.output.mkdir(parents=True,exist_ok=True)
 45:         self.best_loss=float('inf');self.selected=None;self.eval_step=0
 46:         self.marks={max(1,math.ceil(len(self.train_loader)*f)) for f in [.25,.5]}
 47: 
 48:     def calculate_train_loss(self,b,ctx):
 49:         loss=super().calculate_train_loss(b,ctx)
 50:         if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
 51:         return loss
 52: 
 53:     def train_step(self,b,ctx):
 54:         loss=super().train_step(b,ctx)
 55:         if ctx.batch_idx+1 in self.marks:
 56:             self.eval_step=self.global_step+1
 57:             self.evaluate_epoch();self.model.train()
 58:         return loss
 59: 
 60:     def evaluate_epoch(self):
 61:         metrics=super().evaluate_epoch()
 62:         step=max(self.eval_step,self.global_step)
 63:         overall=metrics['overall']
 64:         row={'step':step,'epoch':self.epoch,'stage':self.model.stage,'valid':metrics}
 65:         print('VALID',json.dumps(row),flush=True)
 66:         with (self.output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 67:         if overall['logloss']<self.best_loss:
 68:             self.best_loss=overall['logloss'];self.selected=row
 69:             torch.save({'state_dict':self.model.state_dict(),'selection':row},self.output/'best.pt')
 70:         return metrics
 71: 
 72:     def run(self,*,stop_condition=None):
 73:         # Product start is a valid stage-two checkpoint too.
 74:         if self.model.mode=='ours' and self.model.stage==2:self.evaluate_epoch()
 75:         super().run(stop_condition=stop_condition)
 76:         best=torch.load(self.output/'best.pt',map_location=self.cfg.device,weights_only=False)
 77:         self.model.load_state_dict(best['state_dict'])
 78:         (self.output/'selection.json').write_text(json.dumps({'selected':best['selection'],'optimizer_steps_executed':self.global_step},indent=2))
 79:         return best['selection']['valid']
 80: 
 81: 
 82: @torch.no_grad()
 83: def predict(model,batches,device):
 84:     model.eval();parts=[]
 85:     for batch in batches:
 86:         b=on_device(batch,device);p=model.predict(b)
 87:         if not torch.isfinite(p).all():raise FloatingPointError('nonfinite test predictions')
 88:         data={c:b[c].cpu().numpy() for c in ['request_id','session_id','user_id_raw','tab_raw','req_idx_raw','y_continue','elapsed_min']}
 89:         data['prediction']=p.cpu().numpy()
 90:         if model.mode in ['budget','ours']:
 91:             _,gamma,stay,s,log_s=model.components(b)
 92:             data.update(gamma=gamma.cpu().numpy(),content_stay=stay.cpu().numpy(),survival=s.cpu().numpy(),log_survival=log_s.cpu().numpy())
 93:         parts.append(pl.DataFrame(data))
 94:     return pl.concat(parts)
 95: 
 96: 
 97: def scores(df):
 98:     from sklearn.metrics import roc_auc_score,log_loss
 99:     y,p=df['y_continue'].to_numpy(),df['prediction'].to_numpy()
100:     r={'overall.auc':float(roc_auc_score(y,p)),'user_gauc':grouped_auc(y,p,df['user_id_raw'].to_numpy()),
101:        'logloss':float(log_loss(y,p,labels=[0,1])),'rows':df.height,
102:        'intra_step_gauc':grouped_auc(y,p,df['req_idx_raw'].to_numpy())}
103:     r['domains']={str(tab):scores_domain(part) for tab,part in df.partition_by('tab_raw',as_dict=True).items()}
104:     return r
105: 
106: 
107: def scores_domain(df):
108:     from sklearn.metrics import roc_auc_score,log_loss
109:     y,p=df['y_continue'].to_numpy(),df['prediction'].to_numpy()
110:     return {'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
111:             'gauc':grouped_auc(y,p,df['user_id_raw'].to_numpy()),'logloss':float(log_loss(y,p,labels=[0,1])),'rows':df.height}
112: 
113: 
114: def main():
115:     ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('workspace/kr_time_budget/data/v3'))
116:     ap.add_argument('--output',type=Path,default=Path('workspace/kr_time_budget/runs/exit_v3'))
117:     ap.add_argument('--models',nargs='+',choices=['content','budget','fusion','ours'],default=['content','budget','fusion','ours'])
118:     ap.add_argument('--batch-size',type=int,default=4096);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--seed',type=int,default=2026)
119:     ap.add_argument('--device',default='cpu');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
120:     torch.set_num_threads(args.threads);args.output.mkdir(parents=True,exist_ok=True)
121:     config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
122:     config.update(protocol='Argon4 date split; single epoch baselines; ours 1+1 epochs',weight_decay=1e-6,embedding_decay=0.,lr=.001,
123:                   optimizer='AdamW',physical_effective_batch=args.batch_size,gradient_accumulation=1,scheduler=None,ema=False,
124:                   selection='valid LogLoss at .25/.5/1 epoch; stage2 includes untrained product start',
125:                   comparison='architecture screening; same initial tensors and data, different routing and objectives; ours has two epochs')
126:     config['dataset_manifest_sha256']=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
127:     cp=args.output/'config.json'
128:     if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Use a fresh output directory for a different configuration')
129:     cp.write_text(json.dumps(config,indent=2));manager=request_manager(args.data/'schema',load=True)
130:     train=load_batches(args.data/'train.parquet',manager,args.batch_size,args.smoke)
131:     valid=load_batches(args.data/'valid.parquet',manager,args.batch_size,args.smoke)
132:     test=load_batches(args.data/'test.parquet',manager,args.batch_size,args.smoke)
133:     print('BATCHES',len(train),len(valid),len(test),flush=True)
134:     order=hashlib.sha256(torch.cat([b['request_id'] for b in train]).numpy().tobytes()).hexdigest()
135:     expected=None;summary={}
136:     for mode in args.models:
137:         out=args.output/mode;out.mkdir(parents=True,exist_ok=True)
138:         if (out/'metrics.json').exists():
139:             old=json.loads((out/'manifest.json').read_text())
140:             if expected is not None:assert old['initial_sha256']==expected
141:             expected=old['initial_sha256'];summary[mode]=json.loads((out/'metrics.json').read_text());continue
142:         if list(out.glob('*/progress.jsonl')):raise RuntimeError(f'Incomplete run {out}; use a fresh output or archive it')
143:         torch.manual_seed(args.seed);np.random.seed(args.seed)
144:         model=ExitModelV3(manager,mode);initial=state_hash(model)
145:         if expected is not None:assert expected==initial
146:         expected=initial
147:         (out/'manifest.json').write_text(json.dumps({'initial_sha256':initial,'train_order_sha256':order,
148:             'train_rows':sum(len(b['y_continue']) for b in train),'batch_size':args.batch_size,
149:             'full_train':not args.smoke,'full_test':not args.smoke},indent=2))
150:         def factory(current,stage=None):
151:             name=stage.value if stage else 'single'
152:             phase=out/name
153:             evaluator=EvaluatorManager(log_path=phase/'evaluations.log',title=f'v3 {mode} {name}')
154:             evaluator.register('overall',Evaluator('auc','gauc','logloss'))
155:             for domain in range(5):evaluator.register(f'domain{domain}',Evaluator('auc','gauc','logloss'),DomainFilter('tab',domain))
156:             cfg=SimpleNamespace(device=args.device,max_epochs=1,ckpt_dir=str(phase),experiment_name=f'exit_v3_{mode}_{name}',dataset_name='kuairand_requests_v3',train_log_interval=50)
157:             return ExitTrainer(current,torch.optim.AdamW(split_model_params_by_decay(current,weight_decay=1e-6),lr=.001,weight_decay=0),
158:                 manager,TrainerDataLoaders(train,valid),TrainerComponents(evaluator_manager=evaluator,early_stopper=EarlyStopper(preferred_metrics=('logloss',))),cfg)
159:         if mode=='ours':
160:             result=TwoStageRunner(factory).run(model);model=result.stage_two_model
161:             # Snapshot hashes must cover exactly the stage-one budget tables and net.
162:             for prefix,module in [('budget_embedding.',model.budget_embedding),('budget_net.',model.budget_net)]:
163:                 for name,t in module.state_dict().items():
164:                     assert torch.equal(t.cpu(),result.stage_snapshot.model_state[prefix+name]),f'budget drift {prefix+name}'
165:             ref=on_device(valid[0],args.device)
166:             with torch.no_grad():
167:                 stage1=result.stage_one_trainer.model.budget_components(ref)[0]
168:                 stage2=model.budget_components(ref)[0]
169:                 gamma_drift=(stage1-stage2).abs().max().item()
170:                 assert gamma_drift==0
171:                 stay=model.components(ref)[2].mean().item()
172:             switch={'budget_exactly_frozen':True,'gamma_max_change':gamma_drift,'selected_stage2_content_stay_mean':stay,
173:                     'stage1_steps_executed':result.stage_one_trainer.global_step,'stage2_steps_executed':result.stage_two_trainer.global_step}
174:             (out/'stage_switch.json').write_text(json.dumps(switch,indent=2))
175:             steps=result.stage_one_trainer.global_step+result.stage_two_trainer.global_step
176:         else:
177:             trainer=factory(model);trainer.run();steps=trainer.global_step
178:         pred=predict(model,test,args.device);pred.write_parquet(out/'predictions.parquet')
179:         metric=scores(pred);metric.update(full_train=not args.smoke,full_test=not args.smoke,optimizer_steps_executed=steps)
180:         if mode in ['budget','ours']:
181:             p=pred.sort(['session_id','req_idx_raw']).with_columns(pl.col('log_survival').diff().over('session_id').alias('ds'),pl.col('elapsed_min').diff().over('session_id').alias('dt'))
182:             metric['survival_diagnostics']={'clock_decreases':p.filter(pl.col('dt')<0).height,'S_increase_with_nondecreasing_time':p.filter((pl.col('dt')>=0)&(pl.col('ds')>1e-6)).height}
183:             assert metric['survival_diagnostics']['S_increase_with_nondecreasing_time']==0
184:         if mode=='ours':metric['product_max_abs_error']=(pred['prediction']-pred['gamma']*pred['content_stay']).abs().max()
185:         (out/'metrics.json').write_text(json.dumps(metric,indent=2));summary[mode]=metric
186:         (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
187:         print('RESULT',mode,json.dumps(metric),flush=True)
188:         del model
189:     print('COMPLETE',json.dumps({k:{c:v[c] for c in ['overall.auc','user_gauc','logloss']} for k,v in summary.items()}),flush=True)
190: 
191: if __name__=='__main__':main()
````

## File: workspace/kr_time_budget/pipeline/run_exit_v7_joint.py
````python
  1: """Full KR v3 joint-product training and a matched Fusion baseline.
  2: 
  3: One loss only: BCE(gamma_time * gamma_step * stay, y_continue).
  4: No warmup, detach, frozen tower, auxiliary content loss or dataset recoding.
  5: """
  6: import argparse
  7: import hashlib
  8: import json
  9: from pathlib import Path
 10: import time
 11: from types import SimpleNamespace
 12: import numpy as np
 13: import polars as pl
 14: import torch
 15: from torch.utils.data import DataLoader
 16: from rosen.core.train import EarlyStopper
 17: from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
 18: from rosen.data.dataset import ParquetStreamDataset
 19: from rosen.evaluate.evaluator import Evaluator
 20: from rosen.evaluate.manager import EvaluatorManager, DomainFilter
 21: from rosen.utils.optimize import split_model_params_by_decay
 22: from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, EXTRA_COLUMNS
 23: from workspace.kr_time_budget.pipeline.run_exit import state_hash
 24: from workspace.kr_time_budget.pipeline.run_exit_v3 import ExitTrainer, scores, on_device
 25: from workspace.kr_time_budget.pipeline.time_budget_v7 import TimeBudgetModel
 26: 
 27: ROOT = Path('workspace/kr_time_budget')
 28: 
 29: 
 30: class JointTrainer(ExitTrainer):
 31:     """Native training with an audit of the first real optimizer update."""
 32:     def train_step(self, batch, ctx):
 33:         first_step = self.global_step == 0
 34:         if first_step:
 35:             model = self.model
 36:             if model.mode == 'ours':
 37:                 watched = {
 38:                     'budget_head': model.budget_net[-1].weight,
 39:                     'content_head': model.content_net[-1].weight,
 40:                     'budget_user_embedding': model.budget_embedding.emb_modules['user_id'].embedding.weight,
 41:                     'content_user_embedding': model.omni_embedding.emb_modules['user_id'].embedding.weight,
 42:                 }
 43:             else:
 44:                 watched = {'fusion_head': model.fusion_net[-1].weight,
 45:                            'user_embedding': model.omni_embedding.emb_modules['user_id'].embedding.weight}
 46:             before = {name: p.detach().clone() for name, p in watched.items()}
 47:         loss = super().train_step(batch, ctx)
 48:         if first_step:
 49:             audit = {}
 50:             for name, parameter in watched.items():
 51:                 assert parameter.requires_grad and parameter.grad is not None
 52:                 grad = parameter.grad.norm().item()
 53:                 change = (parameter.detach()-before[name]).abs().max().item()
 54:                 assert np.isfinite(grad) and grad > 0 and change > 0, name
 55:                 audit[name] = {'gradient_norm_after_clipping': grad, 'parameter_max_abs_update': change}
 56:             (self.output/'first_update.json').write_text(json.dumps(audit, indent=2))
 57:             print('FIRST_UPDATE', self.model.mode, json.dumps(audit), flush=True)
 58:         return loss
 59: 
 60: 
 61: def load_data(data, manager, smoke=False):
 62:     batches = {}
 63:     for split in ['train', 'valid', 'test']:
 64:         # Only v3 is accepted below: elapsed_min is physical minutes in this schema.
 65:         frame = pl.scan_parquet(data/f'{split}.parquet').with_columns(pl.col('elapsed_min').alias('elapsed_min_raw'))
 66:         if smoke:
 67:             frame = frame.head(8192 if split == 'train' else 4096)
 68:         dataset = ParquetStreamDataset(frame, manager, batch_size=4096, shuffle=False,
 69:                                        drop_last=False, extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw'])
 70:         batches[split] = list(DataLoader(dataset, batch_size=None, num_workers=0))
 71:     return batches
 72: 
 73: 
 74: @torch.no_grad()
 75: def predict_joint(model, batches):
 76:     model.eval()
 77:     parts = []
 78:     for batch in batches:
 79:         b = on_device(batch, 'cpu')
 80:         data = {c: b[c].numpy() for c in ['request_id', 'session_id', 'user_id_raw', 'tab_raw',
 81:                     'req_idx_raw', 'y_continue', 'elapsed_min', 'elapsed_min_raw']}
 82:         prediction = model.predict(b)
 83:         assert torch.isfinite(prediction).all()
 84:         data['prediction'] = prediction.numpy()
 85:         if model.mode == 'ours':
 86:             product, gamma, stay, survival, log_s = model.components(b)
 87:             assert torch.equal(product, prediction) and gamma.requires_grad is False  # no_grad eval
 88:             clocks = model.clock_components(b)
 89:             data.update(gamma=gamma.numpy(), content_stay=stay.numpy(), survival=survival.numpy(),
 90:                         log_survival=log_s.numpy(), gamma_time=clocks['time'][0].numpy(), gamma_step=clocks['step'][0].numpy())
 91:             assert torch.equal(gamma, clocks['time'][0]*clocks['step'][0])
 92:         parts.append(pl.DataFrame(data))
 93:     return pl.concat(parts)
 94: 
 95: 
 96: def main():
 97:     parser = argparse.ArgumentParser(description=__doc__)
 98:     parser.add_argument('--output', type=Path, default=ROOT/'runs/exit_v7_joint')
 99:     parser.add_argument('--models', nargs='+', choices=['ours', 'fusion'], default=['ours', 'fusion'])
100:     parser.add_argument('--smoke', action='store_true')
101:     args = parser.parse_args()
102:     torch.set_num_threads(4)
103:     args.output.mkdir(parents=True, exist_ok=True)
104:     data = ROOT/'data/v3'
105:     dataset_manifest = json.loads((data/'manifest.json').read_text())
106:     assert dataset_manifest['version'] == 3 and dataset_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
107:     sources = [Path(__file__), Path(__file__).with_name('time_budget_v7.py'),
108:                Path(__file__).with_name('kr_request_schema.py'), Path(__file__).with_name('time_budget_v4.py'),
109:                Path(__file__).with_name('run_exit_v3.py')]
110:     config = {
111:         'data': str(data), 'output': str(args.output), 'models': args.models, 'smoke': args.smoke,
112:         'training_strategy': 'joint', 'clock': 'dual', 'loss': 'BCE(gamma_time*gamma_step*stay,y_continue)',
113:         'content_auxiliary_loss': None, 'warmup': None, 'frozen_budget': False, 'detach_gamma': False,
114:         'seed': 2026, 'epochs': 1, 'batch_size': 4096, 'physical_effective_batch': 4096,
115:         'gradient_accumulation': 1, 'optimizer': 'AdamW', 'lr': .001, 'weight_decay': 1e-6,
116:         'zero_decay': 'Embedding,bias,one-dimensional parameters', 'scheduler': None, 'ema': False,
117:         'device': 'cpu', 'threads': 4, 'initial_content_bias': 3.5,
118:         'selection': 'full valid LogLoss at initial/25/50/100 percent; selected checkpoint full test',
119:         'preprocessing': 'v3 unchanged; lazy elapsed_min_raw alias of physical elapsed_min; no new normalization',
120:         'dataset_manifest_sha256': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
121:         'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
122:         'comparison': 'same initial tensors/data/order/updates; different routing, functional form and active parameters; single-seed screening',
123:     }
124:     cfg_path = args.output/'config.json'
125:     if cfg_path.exists():
126:         assert json.loads(cfg_path.read_text()) == config, 'Use a fresh output for changed configuration/source'
127:     cfg_path.write_text(json.dumps(config, indent=2))
128:     manager = request_manager(data/'schema', load=True)
129:     assert manager.get_setting('elapsed_min').min_val == 0 and manager.get_setting('elapsed_min').max_val == 1
130:     batches = load_data(data, manager, args.smoke)
131:     coverage = {}
132:     for split, loader in batches.items():
133:         ids = torch.cat([b['request_id'] for b in loader]).numpy()
134:         # Dataset manifest hashes the parquet's UInt32 representation; tensor extras are Int64.
135:         raw_ids = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()
136:         if args.smoke:
137:             raw_ids = raw_ids[:len(ids)]
138:         assert np.array_equal(ids, raw_ids), f'{split}: loader changed request order'
139:         order_hash = hashlib.sha256(ids.astype(raw_ids.dtype).tobytes()).hexdigest()
140:         coverage[split] = {'rows': len(ids), 'batches': len(loader), 'request_order_sha256': order_hash}
141:         if not args.smoke:
142:             assert len(ids) == dataset_manifest['splits'][split]['rows']
143:             assert order_hash == dataset_manifest['splits'][split]['request_order_sha256']
144:     (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
145:     print('DATA', json.dumps(coverage), flush=True)
146:     initial_state = None
147:     summary = {}
148:     for mode in args.models:
149:         started = time.monotonic()
150:         out = args.output/mode
151:         out.mkdir(exist_ok=True)
152:         if (out/'metrics.json').exists():
153:             summary[mode] = json.loads((out/'metrics.json').read_text())
154:             continue
155:         if (out/'joint/progress.jsonl').exists():
156:             raise RuntimeError(f'Incomplete run {out}; use a fresh output')
157:         torch.manual_seed(2026)
158:         np.random.seed(2026)
159:         model = TimeBudgetModel(manager, mode=mode, clock='dual', training_strategy='joint')
160:         if initial_state is None:
161:             initial_state = {name: value.clone() for name, value in model.state_dict().items()}
162:         else:
163:             assert all(torch.equal(initial_state[name], value) for name, value in model.state_dict().items())
164:         initial_hash = state_hash(model)
165:         manifest = {'initial_sha256': initial_hash, 'common_initial_tensors_exact': True,
166:                     'train_order_sha256': coverage['train']['request_order_sha256'],
167:                     'full_train': not args.smoke, 'full_valid': not args.smoke, 'full_test': not args.smoke,
168:                     'train_rows': coverage['train']['rows'], 'batch_size': 4096,
169:                     'budget_fields': list(model.budget_view.field_slices),
170:                     'content_fields': list(model.content_view.field_slices),
171:                     'fusion_fields': list(model.fusion_view.field_slices)}
172:         (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
173:         phase = out/'joint'
174:         evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'v7 joint {mode}')
175:         evaluator.register('overall', Evaluator('auc', 'gauc', 'logloss'))
176:         for domain in range(5):
177:             evaluator.register(f'domain{domain}', Evaluator('auc', 'gauc', 'logloss'), DomainFilter('tab', domain))
178:         cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase), experiment_name=f'exit_v7_joint_{mode}',
179:                               dataset_name='kuairand_requests_v3', train_log_interval=50)
180:         optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=0)
181:         trainer = JointTrainer(model, optimizer, manager, TrainerDataLoaders(batches['train'], batches['valid']),
182:                                TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
183:         print('START', mode, initial_hash, flush=True)
184:         if mode != 'ours':
185:             trainer.evaluate_epoch()  # Match the initial-checkpoint selection afforded Ours.
186:         trainer.run()
187:         assert trainer.global_step == coverage['train']['batches']
188:         if mode == 'ours':
189:             assert all(p.requires_grad for module in model.budget_modules for p in module.parameters())
190:         pred = predict_joint(model, batches['test'])
191:         pred.write_parquet(out/'predictions.parquet')
192:         metric = scores(pred)
193:         metric.update(full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
194:                       optimizer_steps_executed=trainer.global_step, selected=trainer.selected,
195:                       selected_state_sha256=state_hash(model), wall_seconds=time.monotonic()-started)
196:         # Independently reload the selected checkpoint and verify every test prediction.
197:         reloaded = TimeBudgetModel(manager, mode=mode, clock='dual', training_strategy='joint')
198:         checkpoint = torch.load(phase/'best.pt', map_location='cpu', weights_only=False)
199:         reloaded.load_state_dict(checkpoint['state_dict'])
200:         reloaded.set_stage(2)
201:         reload_pred = predict_joint(reloaded, batches['test'])
202:         assert pred['request_id'].equals(reload_pred['request_id'])
203:         assert pred['prediction'].equals(reload_pred['prediction'])
204:         assert state_hash(model) == state_hash(reloaded)
205:         metric['checkpoint_reload_predictions_exact'] = True
206:         if mode == 'ours':
207:             assert pred['gamma'].equals(reload_pred['gamma'])
208:             assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
209:             metric['product_max_abs_error'] = 0.
210:             metric['components'] = {name: {'mean': pred[name].mean(), 'std': pred[name].std(),
211:                                           'min': pred[name].min(), 'max': pred[name].max()}
212:                                     for name in ['gamma_time', 'gamma_step', 'gamma', 'content_stay']}
213:             pairs = pred.sort(['session_id', 'req_idx_raw']).with_columns(
214:                 pl.col('elapsed_min_raw').diff().over('session_id').alias('dt'),
215:                 pl.col('gamma').diff().over('session_id').alias('dg'),
216:                 pl.col('log_survival').diff().over('session_id').alias('ds'),
217:                 pl.col('prediction').diff().over('session_id').alias('dp'))
218:             eligible = pairs.filter(pl.col('dt') >= 0)
219:             metric['monotonicity'] = {'nonrewinding_pairs': eligible.height,
220:                 'clock_rewinds': pairs.filter(pl.col('dt') < 0).height,
221:                 'gamma_increases_nonrewinding': eligible.filter(pl.col('dg') > 1e-7).height,
222:                 'logS_increases_nonrewinding': eligible.filter(pl.col('ds') > 1e-6).height,
223:                 'prediction_increases_nonrewinding': eligible.filter(pl.col('dp') > 1e-7).height}
224:             assert metric['monotonicity']['gamma_increases_nonrewinding'] == 0
225:             assert metric['monotonicity']['logS_increases_nonrewinding'] == 0
226:         (out/'metrics.json').write_text(json.dumps(metric, indent=2))
227:         summary[mode] = metric
228:         (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
229:         print('RESULT', mode, json.dumps({k: metric[k] for k in ['overall.auc', 'user_gauc', 'logloss', 'wall_seconds']}), flush=True)
230:     print('COMPLETE', args.output/'summary.json', flush=True)
231: 
232: 
233: if __name__ == '__main__':
234:     main()
````

## File: workspace/kr_time_budget/pipeline/run_exit_v8_branches.py
````python
  1: """Argon 9: paired, full-data ablations and two joint-training branches."""
  2: import argparse
  3: from dataclasses import asdict, dataclass
  4: import hashlib
  5: import json
  6: from pathlib import Path
  7: import time
  8: from types import SimpleNamespace
  9: import numpy as np
 10: import polars as pl
 11: import torch
 12: from rosen.core.train import EarlyStopper
 13: from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
 14: from rosen.evaluate.evaluator import Evaluator
 15: from rosen.evaluate.manager import EvaluatorManager, DomainFilter
 16: from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
 17: from workspace.kr_time_budget.pipeline.run_exit import state_hash
 18: from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
 19: from workspace.kr_time_budget.pipeline.run_exit_v7_joint import JointTrainer, load_data, predict_joint
 20: from workspace.kr_time_budget.pipeline.time_budget_v8 import (
 21:     DualBranchModel, weibull_parameters, single_weibull_survival, joint_parameter_groups,
 22: )
 23: 
 24: ROOT = Path('workspace/kr_time_budget')
 25: 
 26: 
 27: @dataclass(frozen=True)
 28: class Experiment:
 29:     mode: str = 'ours'
 30:     scale_mapping: str = 'log'
 31:     budget_lr: float = .005
 32:     shape_floor: float = 1.05
 33:     content_bias: float = 3.0
 34:     bounded_content: bool = False
 35: 
 36:     def build(self, manager):
 37:         options = asdict(self)
 38:         options.pop('budget_lr')
 39:         return DualBranchModel(manager, **options)
 40: 
 41: 
 42: # Declared before evaluating test; adjacent stages isolate one explicitly named change.
 43: EXPERIMENTS = {
 44:     'reference': Experiment(scale_mapping='legacy', budget_lr=.001, content_bias=3.5),
 45:     'log_scale': Experiment(budget_lr=.001, content_bias=3.5),
 46:     'faster_budget': Experiment(content_bias=3.5),
 47:     'monotone': Experiment(),
 48:     'flexible_shape': Experiment(shape_floor=.3),
 49:     'bounded_content': Experiment(bounded_content=True),
 50:     'fusion': Experiment(mode='fusion', scale_mapping='legacy', budget_lr=.001, content_bias=3.5),
 51: }
 52: 
 53: 
 54: def describe(values):
 55:     a = np.asarray(values, dtype=float)
 56:     assert np.isfinite(a).all()
 57:     return {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()),
 58:             'p05': float(np.quantile(a, .05)), 'p50': float(np.quantile(a, .5)),
 59:             'p95': float(np.quantile(a, .95)), 'max': float(a.max())}
 60: 
 61: 
 62: def diagnostics(model, batches, prediction):
 63:     """Measurements, not an additional training objective or optimizer update."""
 64:     if model.mode == 'fusion':
 65:         return {}
 66:     with torch.no_grad():
 67:         raw = torch.cat([model.budget_raw(b) for b in batches])
 68:         logit = torch.cat([model.content_net(model.content_view(b)).squeeze(-1) for b in batches])
 69:     result = {'components': {c: describe(prediction[c].to_numpy()) for c in
 70:                              ['gamma_time', 'gamma_step', 'gamma', 'content_stay', 'prediction']},
 71:               'content_logit': describe(logit.numpy()), 'clocks': {}}
 72:     for name, r in [('time', raw[:, :2]), ('step', raw[:, 2:])]:
 73:         scale, shape = weibull_parameters(r, model.scale_mapping, model.shape_floor)
 74:         result['clocks'][name] = {'scale': describe(scale.numpy()), 'shape': describe(shape.numpy()),
 75:             'shape_below_one_fraction': float((shape < 1).float().mean()),
 76:             'shape_upper_clamp_fraction': float((shape >= 5).float().mean()),
 77:             'log_scale_clamped_fraction': float(((r[:, 0] <= -2) | (r[:, 0] >= 4.5)).float().mean())
 78:                 if model.scale_mapping == 'log' else None}
 79:     # Per-row derivatives on the same first batch: compare actual raw/logit paths,
 80:     # without inferring AdamW speed from differently-sized parameter gradient norms.
 81:     b = batches[0]
 82:     r = raw[:len(b['y_continue'])].detach().requires_grad_(True)
 83:     z = logit[:len(b['y_continue'])].detach().requires_grad_(True)
 84:     gt = single_weibull_survival(r[:, :2], b['elapsed_min_raw'], torch.ones_like(z), model.scale_mapping, model.shape_floor)[0]
 85:     gs = single_weibull_survival(r[:, 2:], b['req_idx_raw'].float()-1, torch.ones_like(z), model.scale_mapping, model.shape_floor)[0]
 86:     gamma, stay = gt*gs, model.content_stay(z)
 87:     dg = torch.autograd.grad(gamma.sum(), r, retain_graph=True)[0]
 88:     ds = torch.autograd.grad(stay.sum(), z, retain_graph=True)[0]
 89:     loss = torch.nn.functional.binary_cross_entropy((gamma*stay).clamp(1e-6, 1-1e-6), b['y_continue'].float(), reduction='sum')
 90:     grad_r, grad_z = torch.autograd.grad(loss, (r, z))
 91:     result['first_batch_derivatives'] = {
 92:         'abs_mean_dgamma_draw': dg.abs().mean(0).tolist(),
 93:         'abs_mean_dstay_dlogit': float(ds.abs().mean()),
 94:         'abs_mean_drow_bce_draw': grad_r.abs().mean(0).tolist(),
 95:         'abs_mean_drow_bce_dlogit': float(grad_z.abs().mean()),
 96:         'rows': len(z), 'source': 'first batch of full test; diagnostic only',
 97:     }
 98:     pairs = prediction.sort(['session_id', 'req_idx_raw']).with_columns(
 99:         pl.col('elapsed_min_raw').diff().over('session_id').alias('dt'),
100:         pl.col('gamma').diff().over('session_id').alias('dg'),
101:         pl.col('log_survival').diff().over('session_id').alias('ds'),
102:         pl.col('prediction').diff().over('session_id').alias('dp'))
103:     eligible = pairs.filter(pl.col('dt') >= 0)
104:     result['monotonicity'] = {'nonrewinding_pairs': eligible.height, 'clock_rewinds': pairs.filter(pl.col('dt') < 0).height,
105:         'gamma_increases': eligible.filter(pl.col('dg') > 1e-7).height,
106:         'logS_increases': eligible.filter(pl.col('ds') > 1e-6).height,
107:         'prediction_increases': eligible.filter(pl.col('dp') > 1e-7).height}
108:     assert result['monotonicity']['logS_increases'] == 0
109:     if model.shape_floor >= 1.05:
110:         assert result['monotonicity']['gamma_increases'] == 0
111:     # Fixed-checkpoint component readouts: no retraining, no causal attribution.
112:     result['component_readouts'] = {}
113:     for component in ['gamma', 'content_stay']:
114:         metric = scores(prediction.with_columns(pl.col(component).alias('prediction')))
115:         result['component_readouts'][component] = {'auc': metric['overall.auc'], 'user_gauc': metric['user_gauc']['auc']}
116:     return result
117: 
118: 
119: def main():
120:     parser = argparse.ArgumentParser(description=__doc__)
121:     parser.add_argument('--output', type=Path, default=ROOT/'runs/exit_v8_branches')
122:     parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
123:     parser.add_argument('--seeds', nargs='+', type=int, default=[2026])
124:     parser.add_argument('--smoke', action='store_true')
125:     args = parser.parse_args()
126:     torch.set_num_threads(4)
127:     args.output.mkdir(parents=True, exist_ok=True)
128:     data = ROOT/'data/v3'
129:     dataset_manifest = json.loads((data/'manifest.json').read_text())
130:     assert dataset_manifest['version'] == 3 and dataset_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
131:     sources = [Path(__file__), Path(__file__).with_name('time_budget_v8.py'), Path(__file__).with_name('kr_request_schema.py'),
132:                Path(__file__).with_name('run_exit_v7_joint.py'), Path(__file__).with_name('run_exit_v3.py')]
133:     config = {'data': str(data), 'output': str(args.output), 'smoke': args.smoke, 'seeds': args.seeds,
134:         'experiments': {name: asdict(EXPERIMENTS[name]) for name in args.experiments},
135:         'training_strategy': 'joint', 'loss': 'BCE(gamma_time*gamma_step*stay,y_continue)',
136:         'content_auxiliary_loss': None, 'warmup': None, 'detach_gamma': False, 'frozen_budget': False,
137:         'epochs': 1, 'batch_size': 4096, 'gradient_accumulation': 1, 'optimizer': 'AdamW',
138:         'content_lr': .001, 'weight_decay': 1e-6, 'zero_decay': 'Embedding,bias,one-dimensional parameters',
139:         'scheduler': None, 'ema': False, 'device': 'cpu', 'threads': 4,
140:         'selection': 'full valid LogLoss at initial/25/50/100 percent; full test on selected checkpoint',
141:         'preprocessing': 'v3 unchanged; elapsed_min_raw aliases physical elapsed_min; raw step uses req_idx_raw',
142:         'dataset_manifest_sha256': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
143:         'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
144:         'interpretation': 'paired ablation chain; two branch packages vs Fusion are exploratory architecture comparisons'}
145:     cfg_path = args.output/'config.json'
146:     if cfg_path.exists():
147:         assert json.loads(cfg_path.read_text()) == config, 'Changed source/configuration needs fresh output'
148:     cfg_path.write_text(json.dumps(config, indent=2))
149:     manager = request_manager(data/'schema', load=True)
150:     assert manager.get_setting('elapsed_min').min_val == 0 and manager.get_setting('elapsed_min').max_val == 1
151:     batches = load_data(data, manager, args.smoke)
152:     coverage = {}
153:     for split, loader in batches.items():
154:         ids = torch.cat([b['request_id'] for b in loader]).numpy()
155:         expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()[:len(ids)]
156:         assert np.array_equal(ids, expected)
157:         h = hashlib.sha256(ids.astype(expected.dtype).tobytes()).hexdigest()
158:         coverage[split] = dict(rows=len(ids), batches=len(loader), request_order_sha256=h)
159:         if not args.smoke:
160:             assert len(ids) == dataset_manifest['splits'][split]['rows']
161:             assert h == dataset_manifest['splits'][split]['request_order_sha256']
162:     (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
163:     print('DATA', json.dumps(coverage), flush=True)
164:     summary = {}
165:     for seed in args.seeds:
166:         torch.manual_seed(seed)
167:         reference = EXPERIMENTS['reference'].build(manager)
168:         reference_state = {k: v.clone() for k, v in reference.state_dict().items()}
169:         for name in args.experiments:
170:             started = time.monotonic()
171:             experiment = EXPERIMENTS[name]
172:             key = f'{seed}/{name}'
173:             out = args.output/key
174:             out.mkdir(parents=True, exist_ok=True)
175:             if (out/'metrics.json').exists():
176:                 summary[key] = json.loads((out/'metrics.json').read_text())
177:                 continue
178:             if (out/'joint/progress.jsonl').exists():
179:                 raise RuntimeError(f'Incomplete training at {out}; use fresh output')
180:             torch.manual_seed(seed)
181:             np.random.seed(seed)
182:             model = experiment.build(manager)
183:             differing = [k for k, v in model.state_dict().items() if not torch.equal(v, reference_state[k])]
184:             assert set(differing) <= {'prior_stay', 'content_net.4.bias'}
185:             groups = joint_parameter_groups(model, experiment.budget_lr)
186:             group_meta = [dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
187:                                tensors=len(g['params']), elements=sum(p.numel() for p in g['params'])) for g in groups]
188:             manifest = dict(initial_sha256=state_hash(model), initial_differences_from_reference=differing,
189:                 initial_all_other_tensors_exact=True, train_order_sha256=coverage['train']['request_order_sha256'],
190:                 full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
191:                 optimizer_groups=group_meta, experiment=asdict(experiment), seed=seed,
192:                 budget_fields=list(model.budget_view.field_slices), content_fields=list(model.content_view.field_slices))
193:             (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
194:             initial_prediction = predict_joint(model, batches['test'])
195:             initial_prediction.write_parquet(out/'initial_predictions.parquet')
196:             initial_diagnostics = diagnostics(model, batches['test'], initial_prediction)
197:             (out/'initial_diagnostics.json').write_text(json.dumps(initial_diagnostics, indent=2))
198:             phase = out/'joint'
199:             evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'v8 {key}')
200:             evaluator.register('overall', Evaluator('auc', 'gauc', 'logloss'))
201:             for domain in range(5):
202:                 evaluator.register(f'domain{domain}', Evaluator('auc', 'gauc', 'logloss'), DomainFilter('tab', domain))
203:             cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase), experiment_name=f'exit_v8_{name}_{seed}',
204:                 dataset_name='kuairand_requests_v3', train_log_interval=100)
205:             trainer = JointTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
206:                 TrainerDataLoaders(batches['train'], batches['valid']),
207:                 TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
208:             print('START', key, json.dumps(asdict(experiment)), flush=True)
209:             if model.mode == 'fusion':
210:                 trainer.evaluate_epoch()
211:             trainer.run()
212:             assert trainer.global_step == coverage['train']['batches']
213:             assert all(p.requires_grad for p in model.parameters())
214:             prediction = predict_joint(model, batches['test'])
215:             prediction.write_parquet(out/'predictions.parquet')
216:             final_diagnostics = diagnostics(model, batches['test'], prediction)
217:             (out/'final_diagnostics.json').write_text(json.dumps(final_diagnostics, indent=2))
218:             reloaded = experiment.build(manager)
219:             reloaded.load_state_dict(torch.load(phase/'best.pt', map_location='cpu', weights_only=False)['state_dict'])
220:             assert state_hash(model) == state_hash(reloaded)
221:             reload_pred = predict_joint(reloaded, batches['test'])
222:             assert prediction.equals(reload_pred)
223:             metric = scores(prediction)
224:             metric.update(full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
225:                 selected=trainer.selected, optimizer_steps_executed=trainer.global_step,
226:                 selected_state_sha256=state_hash(model), checkpoint_reload_predictions_exact=True,
227:                 wall_seconds=time.monotonic()-started)
228:             if model.mode == 'ours':
229:                 assert (prediction['prediction']-prediction['gamma']*prediction['content_stay']).abs().max() == 0
230:                 metric['product_max_abs_error'] = 0.
231:                 metric['gamma_std'] = final_diagnostics['components']['gamma']['std']
232:             (out/'metrics.json').write_text(json.dumps(metric, indent=2))
233:             summary[key] = metric
234:             (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
235:             print('RESULT', key, json.dumps({k: metric[k] for k in ['overall.auc', 'user_gauc', 'logloss', 'wall_seconds']}), flush=True)
236:     (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
237:     print('COMPLETE', args.output/'summary.json', flush=True)
238: 
239: 
240: if __name__ == '__main__':
241:     main()
````

## File: workspace/kr_time_budget/pipeline/run_exit_v9_pooled.py
````python
  1: """Argon 10: full-data, three-seed Pooled AUC/LogLoss scale-cap ablations."""
  2: import argparse
  3: import hashlib
  4: import json
  5: from pathlib import Path
  6: import time
  7: from types import SimpleNamespace
  8: import numpy as np
  9: import polars as pl
 10: import torch
 11: from rosen.core.train import EarlyStopper
 12: from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
 13: from rosen.evaluate.evaluator import Evaluator
 14: from rosen.evaluate.manager import EvaluatorManager, DomainFilter
 15: from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
 16: from workspace.kr_time_budget.pipeline.run_exit import state_hash
 17: from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
 18: from workspace.kr_time_budget.pipeline.run_exit_v7_joint import JointTrainer, load_data, predict_joint
 19: from workspace.kr_time_budget.pipeline.run_exit_v8_branches import describe
 20: from workspace.kr_time_budget.pipeline.time_budget_v8 import joint_parameter_groups
 21: from workspace.kr_time_budget.pipeline.time_budget_v9 import (
 22:     PooledTimeBudgetModel, weibull_parameters, single_weibull_survival,
 23: )
 24: 
 25: ROOT = Path('workspace/kr_time_budget')
 26: EXPERIMENTS = {
 27:     'capped': dict(mode='ours', scale_caps='both'),
 28:     'minute_uncapped': dict(mode='ours', scale_caps='step'),
 29:     'both_uncapped': dict(mode='ours', scale_caps='none'),
 30:     'fusion': dict(mode='fusion', scale_caps='both'),
 31: }
 32: 
 33: 
 34: def calibration(pred):
 35:     y, p = pred['y_continue'].to_numpy().astype(float), pred['prediction'].to_numpy().astype(float)
 36:     bins = np.minimum((p*20).astype(int), 19)
 37:     ece = sum(np.mean(bins == i)*abs(p[bins == i].mean()-y[bins == i].mean()) for i in np.unique(bins))
 38:     labels = ['1', '2', '3', '4–5', '6–10', '11–20', '21–50', '51+']
 39:     step_bins = np.digitize(pred['req_idx_raw'].to_numpy(), [1, 2, 3, 5, 10, 20, 50], right=True)
 40:     return {'brier': float(np.square(p-y).mean()), 'ece20_fixed_width': float(ece),
 41:         'mean_prediction': float(p.mean()), 'observed_continue_rate': float(y.mean()),
 42:         'mean_prediction_minus_observed': float((p-y).mean()),
 43:         'by_step': [dict(step=label, rows=int((step_bins == i).sum()),
 44:                         observed_rate=float(y[step_bins == i].mean()), prediction_mean=float(p[step_bins == i].mean()))
 45:                     for i, label in enumerate(labels) if (step_bins == i).any()]}
 46: 
 47: 
 48: def diagnostics(model, batches, pred):
 49:     if model.mode == 'fusion':
 50:         return {}
 51:     with torch.no_grad():
 52:         raw = torch.cat([model.budget_raw(b) for b in batches])
 53:     result = {'components': {c: describe(pred[c].to_numpy()) for c in
 54:         ['gamma_time', 'gamma_step', 'gamma', 'content_stay']}, 'clocks': {}}
 55:     for name, r, elapsed in [('time', raw[:, :2], pred['elapsed_min_raw'].to_numpy()),
 56:                               ('step', raw[:, 2:], pred['req_idx_raw'].to_numpy()-1)]:
 57:         capped = model.is_capped(name)
 58:         scale, shape = weibull_parameters(r, capped)
 59:         # Diagnose whether the scale path above the old ceiling actually has a derivative.
 60:         derivative_raw = r.detach().requires_grad_(True)
 61:         gamma = single_weibull_survival(derivative_raw, torch.as_tensor(elapsed.copy()),
 62:                                        torch.ones(len(r)), capped)[0]
 63:         derivative = torch.autograd.grad(gamma.sum(), derivative_raw)[0][:, 0]
 64:         assert torch.isfinite(derivative).all()
 65:         above = r[:, 0] > 4.5
 66:         result['clocks'][name] = dict(capped=capped, scale=describe(scale.numpy()), shape=describe(shape.numpy()),
 67:             raw_scale=describe(r[:, 0].numpy()), above_old_upper_fraction=float(above.float().mean()),
 68:             hard_clamped_fraction=float(((r[:, 0] <= -2) | (r[:, 0] >= 4.5)).float().mean()) if capped else 0.,
 69:             above_old_upper_with_nonzero_derivative=int((above & (derivative.abs() > 0)).sum()),
 70:             abs_mean_dgamma_draw_scale=float(derivative.abs().mean()))
 71:     pairs = pred.sort(['session_id', 'req_idx_raw']).with_columns(
 72:         pl.col('elapsed_min_raw').diff().over('session_id').alias('dt'),
 73:         pl.col('log_survival').diff().over('session_id').alias('ds'),
 74:         pl.col('gamma').diff().over('session_id').alias('dg'))
 75:     eligible = pairs.filter(pl.col('dt') >= 0)
 76:     result['clock_diagnostics'] = dict(nonrewinding_pairs=eligible.height, clock_rewinds=pairs.filter(pl.col('dt') < 0).height,
 77:         logS_increases=eligible.filter(pl.col('ds') > 1e-6).height, gamma_increases=eligible.filter(pl.col('dg') > 1e-7).height)
 78:     assert result['clock_diagnostics']['logS_increases'] == 0
 79:     return result
 80: 
 81: 
 82: def main():
 83:     parser = argparse.ArgumentParser(description=__doc__)
 84:     parser.add_argument('--output', type=Path, default=ROOT/'runs/exit_v9_pooled')
 85:     parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
 86:     parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
 87:     parser.add_argument('--smoke', action='store_true')
 88:     args = parser.parse_args()
 89:     torch.set_num_threads(4)
 90:     args.output.mkdir(parents=True, exist_ok=True)
 91:     data = ROOT/'data/v3'
 92:     data_manifest = json.loads((data/'manifest.json').read_text())
 93:     assert data_manifest['version'] == 3 and data_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
 94:     sources = [Path(__file__), *[Path(__file__).with_name(n) for n in ['time_budget_v9.py', 'time_budget_v8.py',
 95:         'kr_request_schema.py', 'run_exit_v7_joint.py', 'run_exit_v8_branches.py', 'run_exit_v3.py', 'run_exit.py']]]
 96:     config = dict(data=str(data), output=str(args.output), smoke=args.smoke, seeds=args.seeds,
 97:         experiments={n: EXPERIMENTS[n] for n in args.experiments},
 98:         primary_metrics=['overall.auc', 'logloss'], auxiliary_metric='intra_step_gauc',
 99:         user_gauc_role='diagnostic only; never used for loss or checkpoint selection',
100:         training_strategy='joint', loss='BCE(gamma_time*gamma_step*stay,y_continue)',
101:         auxiliary_loss=None, warmup=None, detach_gamma=False, frozen_budget=False,
102:         shape_floor=.3, shape_ceiling=5., scale_log_offset=1.6, initial_content_bias=3.,
103:         epochs=1, batch_size=4096, gradient_accumulation=1, optimizer='AdamW',
104:         budget_lr=.005, content_lr=.001, weight_decay=1e-6, zero_decay='Embedding,bias,one-dimensional parameters',
105:         scheduler=None, ema=False, device='cpu', threads=4,
106:         selection='full valid LogLoss at initial/25/50/100 percent, then full test',
107:         preprocessing='v3 unchanged; physical elapsed_min aliases elapsed_min_raw; req_idx_raw preserved',
108:         source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
109:         dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
110:     config_path = args.output/'config.json'
111:     if config_path.exists():
112:         assert json.loads(config_path.read_text()) == config, 'Changed source/config needs fresh output'
113:     config_path.write_text(json.dumps(config, indent=2))
114:     manager = request_manager(data/'schema', load=True)
115:     assert manager.get_setting('elapsed_min').min_val == 0 and manager.get_setting('elapsed_min').max_val == 1
116:     batches = load_data(data, manager, args.smoke)
117:     coverage = {}
118:     for split, loader in batches.items():
119:         ids = torch.cat([b['request_id'] for b in loader]).numpy()
120:         expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()[:len(ids)]
121:         assert np.array_equal(ids, expected)
122:         order = hashlib.sha256(ids.astype(expected.dtype).tobytes()).hexdigest()
123:         coverage[split] = dict(rows=len(ids), batches=len(loader), request_order_sha256=order)
124:         if not args.smoke:
125:             assert len(ids) == data_manifest['splits'][split]['rows']
126:             assert order == data_manifest['splits'][split]['request_order_sha256']
127:     (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
128:     print('DATA', json.dumps(coverage), flush=True)
129:     summary = {}
130:     for seed in args.seeds:
131:         torch.manual_seed(seed)
132:         reference_state = {k: v.clone() for k, v in PooledTimeBudgetModel(manager, scale_caps='both').state_dict().items()}
133:         for name in args.experiments:
134:             started = time.monotonic()
135:             key, options = f'{seed}/{name}', EXPERIMENTS[name]
136:             out = args.output/key
137:             out.mkdir(parents=True, exist_ok=True)
138:             if (out/'metrics.json').exists():
139:                 summary[key] = json.loads((out/'metrics.json').read_text())
140:                 continue
141:             if (out/'joint/progress.jsonl').exists():
142:                 raise RuntimeError(f'Incomplete run {out}; use fresh output')
143:             torch.manual_seed(seed)
144:             np.random.seed(seed)
145:             model = PooledTimeBudgetModel(manager, **options)
146:             assert all(torch.equal(v, reference_state[k]) for k, v in model.state_dict().items())
147:             groups = joint_parameter_groups(model, .005)
148:             manifest = dict(initial_sha256=state_hash(model), initial_all_tensors_exact=True,
149:                 train_order_sha256=coverage['train']['request_order_sha256'], full_train=not args.smoke,
150:                 full_valid=not args.smoke, full_test=not args.smoke, seed=seed, experiment=options,
151:                 budget_fields=list(model.budget_view.field_slices), content_fields=list(model.content_view.field_slices),
152:                 optimizer_groups=[dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
153:                     tensors=len(g['params']), elements=sum(p.numel() for p in g['params'])) for g in groups])
154:             (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
155:             initial_pred = predict_joint(model, batches['test'])
156:             initial_pred.write_parquet(out/'initial_predictions.parquet')
157:             (out/'initial_diagnostics.json').write_text(json.dumps(diagnostics(model, batches['test'], initial_pred), indent=2))
158:             phase = out/'joint'
159:             evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'v9 pooled {key}')
160:             evaluator.register('overall', Evaluator('auc', 'logloss'))
161:             for domain in range(5):
162:                 evaluator.register(f'domain{domain}', Evaluator('auc', 'logloss'), DomainFilter('tab', domain))
163:             cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase), experiment_name=f'exit_v9_{name}_{seed}',
164:                 dataset_name='kuairand_requests_v3', train_log_interval=100)
165:             trainer = JointTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
166:                 TrainerDataLoaders(batches['train'], batches['valid']),
167:                 TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
168:             print('START', key, json.dumps(options), flush=True)
169:             if model.mode == 'fusion':
170:                 trainer.evaluate_epoch()
171:             trainer.run()
172:             assert trainer.global_step == coverage['train']['batches']
173:             assert all(p.requires_grad for p in model.parameters())
174:             pred = predict_joint(model, batches['test'])
175:             pred.write_parquet(out/'predictions.parquet')
176:             (out/'final_diagnostics.json').write_text(json.dumps(diagnostics(model, batches['test'], pred), indent=2))
177:             cal = calibration(pred)
178:             (out/'calibration.json').write_text(json.dumps(cal, indent=2))
179:             reloaded = PooledTimeBudgetModel(manager, **options)
180:             reloaded.load_state_dict(torch.load(phase/'best.pt', map_location='cpu', weights_only=False)['state_dict'])
181:             assert state_hash(reloaded) == state_hash(model)
182:             assert predict_joint(reloaded, batches['test']).equals(pred)
183:             metric = scores(pred)
184:             metric.update(full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
185:                 optimizer_steps_executed=trainer.global_step, selected=trainer.selected,
186:                 checkpoint_reload_predictions_exact=True, selected_state_sha256=state_hash(model),
187:                 calibration={k: v for k, v in cal.items() if k != 'by_step'}, wall_seconds=time.monotonic()-started)
188:             if model.mode == 'ours':
189:                 assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
190:                 metric['product_max_abs_error'] = 0.
191:             (out/'metrics.json').write_text(json.dumps(metric, indent=2))
192:             summary[key] = metric
193:             (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
194:             print('RESULT', key, json.dumps({k: metric[k] for k in ['overall.auc', 'logloss', 'intra_step_gauc', 'wall_seconds']}), flush=True)
195:     (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
196:     print('COMPLETE', args.output/'summary.json', flush=True)
197: 
198: 
199: if __name__ == '__main__':
200:     main()
````

## File: workspace/kr_time_budget/pipeline/run_exit.py
````python
  1: """Full user-grouped 5-fold exit screening with auditable OOF predictions."""
  2: import argparse
  3: import hashlib
  4: import json
  5: import time
  6: from pathlib import Path
  7: import numpy as np
  8: import polars as pl
  9: import torch
 10: from sklearn.metrics import roc_auc_score, log_loss
 11: from torch.utils.data import DataLoader
 12: from rosen.data.dataset import ParquetStreamDataset
 13: from rosen.utils.optimize import split_model_params_by_decay
 14: from workspace.kr_time_budget.pipeline.time_budget import ExitModel
 15: from workspace.kr_time_budget.pipeline.features import prepare_fold, EXTRAS
 16: 
 17: 
 18: def grouped_auc(y, p, groups):
 19:     # Mann-Whitney rank formula, with average ranks for ties. No O(groups)
 20:     # Python/sklearn loop, which is expensive for >100k sessions.
 21:     ranked = pl.DataFrame({'g': groups, 'y': y, 'p': p}).with_columns(pl.col('p').rank('average').over('g').alias('rank'))
 22:     stats = ranked.group_by('g').agg(pl.len().alias('n'), pl.col('y').sum().alias('pos'), (pl.col('rank') * pl.col('y')).sum().alias('rank_sum'))
 23:     valid = stats.filter((pl.col('pos') > 0) & (pl.col('pos') < pl.col('n'))).with_columns(((pl.col('rank_sum') - pl.col('pos') * (pl.col('pos') + 1) / 2) / (pl.col('pos') * (pl.col('n') - pl.col('pos')))).alias('auc'))
 24:     weight = valid['n'].sum()
 25:     return {'auc': (valid['auc'] * valid['n']).sum() / weight if weight else None, 'eligible_groups': valid.height, 'eligible_rows': weight, 'total_groups': stats.height}
 26: 
 27: 
 28: def metrics(df):
 29:     y, p = df['y_continue'].to_numpy(), df['prediction'].to_numpy()
 30:     return {'overall.auc': roc_auc_score(y, p), 'logloss': log_loss(y, p, labels=[0, 1]), 'user_gauc': grouped_auc(y, p, df['user_id'].to_numpy()), 'session_gauc': grouped_auc(y, p, df['session_id'].to_numpy()), 'rows': len(y)}
 31: 
 32: 
 33: def state_hash(model):
 34:     h = hashlib.sha256()
 35:     for name, tensor in sorted(model.state_dict().items()):
 36:         h.update(name.encode())
 37:         h.update(tensor.detach().cpu().numpy().tobytes())
 38:     return h.hexdigest()
 39: 
 40: 
 41: def batches(path, manager, size):
 42:     ds = ParquetStreamDataset(path, manager, batch_size=size, shuffle=False, drop_last=False, extra_col_names=EXTRAS)
 43:     return DataLoader(ds, batch_size=None, num_workers=0)
 44: 
 45: 
 46: def on_device(batch, device):
 47:     return {k: v.to(device) for k, v in batch.interaction.items()}
 48: 
 49: 
 50: def main():
 51:     ap = argparse.ArgumentParser()
 52:     ap.add_argument('--root', type=Path, default=Path('workspace/kr_time_budget/data/v1'))
 53:     ap.add_argument('--output', type=Path, default=Path('workspace/kr_time_budget/runs/exit_v1'))
 54:     ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
 55:     ap.add_argument('--models', nargs='+', default=['content', 'budget', 'fusion', 'ours'])
 56:     ap.add_argument('--epochs', type=int, default=8)
 57:     ap.add_argument('--batch-size', type=int, default=4096)
 58:     ap.add_argument('--device', default='cpu')
 59:     ap.add_argument('--threads', type=int, default=4)
 60:     ap.add_argument('--seed', type=int, default=2026)
 61:     ap.add_argument('--smoke', action='store_true')
 62:     args = ap.parse_args()
 63:     torch.set_num_threads(args.threads)
 64:     args.output.mkdir(parents=True, exist_ok=True)
 65:     config = vars(args).copy()
 66:     config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
 67:     config.update(weight_decay=1e-6, decay_exclusions='embedding, bias, normalization, scalar/vector parameters', optimizer='AdamW', lr=.001, ours_budget_stage2_lr=.0001, scheduler=None, checkpoint='last fixed epoch; no test-driven selection', comparison='non-causal architecture screening; identical initial tensors/data/batches, different masks and training objectives', ema=False)
 68:     config_path = args.output / 'config.json'
 69:     if config_path.exists() and json.loads(config_path.read_text()) != config:
 70:         raise RuntimeError('Output already contains a different config; use a new output directory')
 71:     config_path.write_text(json.dumps(config, indent=2))
 72:     for fold in args.folds:
 73:         mgr, data = prepare_fold(args.root, fold, args.seed)
 74:         start = time.monotonic()
 75:         train_batches = []
 76:         for batch in batches(data / 'train.parquet', mgr, args.batch_size):
 77:             train_batches.append(batch)
 78:             if args.smoke:
 79:                 break
 80:         print('Cached rosen train batches', fold, len(train_batches), 'seconds', time.monotonic() - start, flush=True)
 81:         hashes = {}
 82:         for mode in args.models:
 83:             out = args.output / f'fold_{fold}' / mode
 84:             out.mkdir(parents=True, exist_ok=True)
 85:             if (out / 'metrics.json').exists():
 86:                 print('RESUME skip completed', fold, mode, flush=True)
 87:                 continue
 88:             torch.manual_seed(args.seed)
 89:             np.random.seed(args.seed)
 90:             model = ExitModel(mgr, mode=mode)
 91:             init_hash = state_hash(model)
 92:             hashes[mode] = init_hash
 93:             assert len(set(hashes.values())) == 1, hashes
 94:             model.to(args.device)
 95:             optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=0)
 96:             manifest = {'initial_state_sha256': init_hash, 'fold': fold, 'mode': mode, 'train_rows': json.loads((data / 'manifest.json').read_text())['train_rows'], 'physical_batch': args.batch_size, 'effective_batch': args.batch_size, 'accumulation': 1, 'epochs': args.epochs, 'seed': args.seed}
 97:             (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
 98:             steps = 0
 99:             for epoch in range(args.epochs):
100:                 model.train()
101:                 model.stage = 1 if mode == 'ours' and epoch < 3 else 2
102:                 if mode == 'ours' and epoch == 3:
103:                     # Preserve optimizer moments while assigning the predeclared lower tower LR.
104:                     budget_ids = {id(p) for p in model.budget_net.parameters()}
105:                     for group in list(optimizer.param_groups):
106:                         budget = [p for p in group['params'] if id(p) in budget_ids]
107:                         group['params'] = [p for p in group['params'] if id(p) not in budget_ids]
108:                         if budget:
109:                             optimizer.add_param_group({'params': budget, 'weight_decay': group['weight_decay'], 'lr': .0001})
110:                 start = time.monotonic()
111:                 rows, loss_sum = 0, 0.
112:                 for batch in train_batches:
113:                     b = on_device(batch, args.device)
114:                     optimizer.zero_grad(set_to_none=True)
115:                     loss = model.calculate_loss(b)
116:                     if not torch.isfinite(loss):
117:                         raise FloatingPointError(f'nonfinite loss fold={fold} mode={mode} step={steps}')
118:                     loss.backward()
119:                     torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
120:                     optimizer.step()
121:                     n = len(b['y_continue'])
122:                     rows += n
123:                     loss_sum += loss.item() * n
124:                     steps += 1
125:                     if args.smoke:
126:                         break
127:                 record = {'fold': fold, 'model': mode, 'epoch': epoch + 1, 'stage': model.stage, 'train_rows': rows, 'optimizer_steps': steps, 'loss': loss_sum / rows, 'seconds': time.monotonic() - start}
128:                 print(json.dumps(record), flush=True)
129:                 with (out / 'epochs.jsonl').open('a') as f:
130:                     f.write(json.dumps(record) + '\n')
131:                 torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch + 1, 'stage': model.stage, 'steps': steps, 'config': config}, out / 'last.pt')
132:                 model.to(args.device)
133:             model.eval()
134:             model.stage = 2
135:             pieces = []
136:             with torch.no_grad():
137:                 for batch in batches(data / 'test.parquet', mgr, args.batch_size):
138:                     b = on_device(batch, args.device)
139:                     pred = model.predict(b)
140:                     vals = {k: b[k].cpu().numpy() for k in ['request_id', 'user_id', 'session_id', 'y_continue']}
141:                     vals['prediction'] = pred.cpu().numpy()
142:                     if mode == 'ours':
143:                         _, gamma, stay, survival, log_survival = model.components(b)
144:                         vals.update(gamma=gamma.cpu().numpy(), content_stay=stay.cpu().numpy(), survival=survival.cpu().numpy(), log_survival=log_survival.cpu().numpy())
145:                     pieces.append(pl.DataFrame(vals))
146:                     if args.smoke:
147:                         break
148:             predictions = pl.concat(pieces).with_columns(pl.lit(fold).alias('fold'))
149:             predictions.write_parquet(out / 'predictions.parquet')
150:             result = metrics(predictions)
151:             result.update(full_train=not args.smoke, full_test=not args.smoke, optimizer_steps=steps)
152:             (out / 'metrics.json').write_text(json.dumps(result, indent=2))
153:             print('RESULT', fold, mode, json.dumps(result), flush=True)
154:             del model, optimizer, pieces
155:         del train_batches
156:     summaries = {}
157:     for mode in args.models:
158:         paths = sorted(args.output.glob(f'fold_*/{mode}/predictions.parquet'))
159:         if paths:
160:             df = pl.read_parquet(paths)
161:             result = metrics(df)
162:             fold_results = [json.loads(p.with_name('metrics.json').read_text()) for p in paths]
163:             result['fold_results'] = fold_results
164:             result['completed_folds'] = len(paths)
165:             summaries[mode] = result
166:     (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
167:     print('SUMMARY', json.dumps(summaries), flush=True)
168: 
169: if __name__ == '__main__':
170:     main()
````

## File: workspace/kr_time_budget/pipeline/time_budget_v8.py
````python
  1: """Argon 9双分支：保持54字段路由，检验预算参数化及条件存活率形状。
  2: 
  3: 30共享背景进入两塔；4预算先验仅进入34字段预算参数网；分钟/屏数raw时钟
  4: 只进入各自威布尔公式；18内容字段与30共享背景进入48字段内容视图。
  5: 两套Omni参数独立。预处理、sum聚合和log1p沿用v3，不在本轮修改。
  6: 
  7: 只做联合训练：prediction=gamma_time*gamma_step*stay，一个BCE更新两塔。
  8: 不增加辅助Loss，不预热、不冻结、不detach，不给预算塔加入动态内容反馈。
  9: shape_floor=1.05保留预算gamma递减约束；0.3允许shape<1，放弃该约束，
 10: 但两支的累计存活S依然不增。此约束不保证乘上变化的stay后预测也递减。
 11: 
 12: legacy参数化用于逐值复现v7；log参数化严格按Argon 9的exp(clamp(raw)+1.6)。
 13: bounded_content限定stay=sigmoid(3+0.5*tanh(logit-3))；减3只撤掉末层
 14: 初始化bias，使有界/无界分支保持相同初始tensor。它会限制概率表达范围。
 15: """
 16: import torch
 17: from torch import nn
 18: from torch.nn import functional as F
 19: from rosen.models.msr.base import MSRModel
 20: from rosen.emb.emblayer import OmniEmbLayer, EmbView
 21: from workspace.kr_time_budget.pipeline.kr_request_schema import (
 22:     FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
 23: )
 24: from rosen.utils.optimize import split_model_params_by_decay
 25: 
 26: 
 27: def weibull_parameters(raw, scale_mapping='log', shape_floor=1.05):
 28:     if scale_mapping == 'legacy':
 29:         scale = F.softplus(raw[:, 0])*30.0+0.5
 30:     elif scale_mapping == 'log':
 31:         scale = torch.exp(raw[:, 0].clamp(-2.0, 4.5)+1.6)
 32:     else:
 33:         raise ValueError(scale_mapping)
 34:     return scale, (shape_floor+F.softplus(raw[:, 1])).clamp(max=5.0)
 35: 
 36: 
 37: def single_weibull_survival(raw, elapsed, delta, scale_mapping='log', shape_floor=1.05):
 38:     """Stable float64 hazard increments, also valid for 0<shape<1 at t=0."""
 39:     scale, shape = (x.double() for x in weibull_parameters(raw, scale_mapping, shape_floor))
 40:     t, d = elapsed.double().clamp(min=0), delta.double().clamp(min=0)
 41:     safe_t = t.clamp(min=1e-12)
 42:     # When shape<1, differentiating (0/scale)**shape gives 0*inf in PyTorch.
 43:     # Evaluate a positive base, then impose the exact zero boundary value.
 44:     hazard = torch.where(t > 0, (safe_t/scale).pow(shape), torch.zeros_like(t))
 45:     increment_positive = (safe_t/scale).pow(shape)*torch.expm1(shape*torch.log1p(d/safe_t))
 46:     origin_increment = torch.where(d > 0, (d.clamp(min=1e-12)/scale).pow(shape), torch.zeros_like(d))
 47:     increment = torch.where(t > 0, increment_positive, origin_increment)
 48:     log_s = -hazard
 49:     return (-increment).exp().to(raw.dtype), log_s.exp().to(raw.dtype), log_s
 50: 
 51: 
 52: class DualBranchModel(MSRModel):
 53:     def __init__(self, manager, mode='ours', scale_mapping='log', shape_floor=1.05,
 54:                  content_bias=3.0, bounded_content=False):
 55:         if mode not in {'ours', 'fusion'} or scale_mapping not in {'legacy', 'log'}:
 56:             raise ValueError((mode, scale_mapping))
 57:         if shape_floor not in {0.3, 1.05} or (bounded_content and content_bias != 3.0):
 58:             raise ValueError((shape_floor, content_bias, bounded_content))
 59:         if {s.field_name for s in manager.settings} != set(FEATURES):
 60:             raise ValueError('DualBranchModel requires the KR 54-field schema')
 61:         super().__init__(manager, num_domains=5)
 62:         self.mode, self.scale_mapping, self.shape_floor = mode, scale_mapping, shape_floor
 63:         self.content_bias, self.bounded_content = content_bias, bounded_content
 64:         self.training_strategy, self.stage = 'joint', 2
 65:         budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
 66:         self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
 67:         for name, table in self.budget_embedding.emb_modules.items():
 68:             table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
 69:         self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
 70:         self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
 71:         self.fusion_view = self.omni_embedding.whole
 72:         # 与v7相同模块构造顺序：各方案只有明确的bias变化，其他初始tensor相同。
 73:         self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim, 64), nn.SiLU(), nn.Linear(64, 4))
 74:         self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim, 64), nn.SiLU(),
 75:                                          nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
 76:         self.fusion_net = nn.Sequential(nn.Linear(self.fusion_view.embedding_dim, 64), nn.SiLU(),
 77:                                         nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
 78:         nn.init.constant_(self.content_net[-1].bias, content_bias)
 79:         self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(content_bias)))
 80: 
 81:     @property
 82:     def budget_modules(self):
 83:         return self.budget_embedding, self.budget_net
 84: 
 85:     def budget_raw(self, batch):
 86:         return self.budget_net(self.budget_view(batch))
 87: 
 88:     def clock_components(self, batch):
 89:         raw = self.budget_raw(batch)
 90:         time = batch['elapsed_min_raw'].float().clamp(min=0)
 91:         step = (batch['req_idx_raw'].float()-1).clamp(min=0)
 92:         return {name: single_weibull_survival(head, t, torch.ones_like(t), self.scale_mapping, self.shape_floor)
 93:                 for name, head, t in [('time', raw[:, :2], time), ('step', raw[:, 2:], step)]}
 94: 
 95:     def budget_components(self, batch):
 96:         clocks = self.clock_components(batch)
 97:         gamma = clocks['time'][0]*clocks['step'][0]
 98:         log_s = clocks['time'][2]+clocks['step'][2]
 99:         return gamma, log_s.exp().to(gamma.dtype), log_s
100: 
101:     def content_stay(self, logit):
102:         if self.bounded_content:
103:             logit = 3.0+0.5*torch.tanh(logit-3.0)
104:         return torch.sigmoid(logit)
105: 
106:     def components(self, batch):
107:         gamma, survival, log_s = self.budget_components(batch)
108:         stay = self.content_stay(self.content_net(self.content_view(batch)).squeeze(-1))
109:         return gamma*stay, gamma, stay, survival, log_s
110: 
111:     def predict(self, batch):
112:         if self.mode == 'fusion':
113:             return torch.sigmoid(self.fusion_net(self.fusion_view(batch)).squeeze(-1))
114:         return self.components(batch)[0]
115: 
116:     def calculate_loss(self, batch):
117:         return F.binary_cross_entropy(self.predict(batch).clamp(1e-6, 1-1e-6), batch['y_continue'].float())
118: 
119: 
120: def joint_parameter_groups(model, budget_lr):
121:     """Native decay grouping, then disjoint budget/content LR groups; one optimizer."""
122:     budget_ids = {id(p) for module in model.budget_modules for p in module.parameters()}
123:     groups = []
124:     for decay_group in split_model_params_by_decay(model, weight_decay=1e-6):
125:         for scope in ['budget', 'other']:
126:             params = [p for p in decay_group['params'] if (id(p) in budget_ids) == (scope == 'budget')]
127:             if params:
128:                 groups.append(dict(params=params, weight_decay=decay_group['weight_decay'],
129:                                    lr=budget_lr if scope == 'budget' else .001, scope=scope))
130:     ids = [id(p) for group in groups for p in group['params']]
131:     assert len(ids) == len(set(ids)) == len(list(model.parameters()))
132:     return groups
````

## File: workspace/kr_time_budget/pipeline/time_budget_v9.py
````python
  1: """Argon 10：以Pooled AUC/LogLoss为主线的双时钟联合模型。
  2: 
  3: 沿用v8 flexible_shape：34字段预算参数网、48字段内容塔、两套独立Omni；
  4: 30共享背景进入两塔，4预算先验仅进预算网，两个raw时钟只进各自公式，
  5: 18内容字段仅进内容塔。数据处理与schema不变。只优化gamma_time*gamma_step
  6: *stay的一个BCE，不使用User-GAUC训练目标、辅助Loss、预热或冻结。
  7: 
  8: scale_caps='both'逐值复现v8 flexible_shape；'step'只保留屏数尺度截断；
  9: 'none'去掉两个尺度的上下硬截断。未截断的lambda=exp(raw+1.6)>0，
 10: 不再偷偷加另一个尺度上界；保留shape=clamp(0.3+softplus(raw),max=5)。
 11: 所有模式构造完全相同的模块，保证全state初始化逐值相同。
 12: """
 13: import torch
 14: from torch import nn
 15: from torch.nn import functional as F
 16: from rosen.models.msr.base import MSRModel
 17: from rosen.emb.emblayer import OmniEmbLayer, EmbView
 18: from workspace.kr_time_budget.pipeline.kr_request_schema import (
 19:     FEATURES, BUDGET_PARAMETER_FIELDS, CONTENT_TOWER_FIELDS,
 20: )
 21: 
 22: 
 23: def weibull_parameters(raw, capped):
 24:     log_scale = raw[:, 0].clamp(-2.0, 4.5) if capped else raw[:, 0]
 25:     scale = torch.exp(log_scale+1.6)
 26:     if not torch.isfinite(scale).all() or not (scale > 0).all():
 27:         raise FloatingPointError('Nonfinite/zero uncapped scale; stop rather than silently cap it')
 28:     return scale, (.3+F.softplus(raw[:, 1])).clamp(max=5.)
 29: 
 30: 
 31: def single_weibull_survival(raw, elapsed, delta, capped):
 32:     scale, shape = (x.double() for x in weibull_parameters(raw, capped))
 33:     t, d = elapsed.double().clamp(min=0), delta.double().clamp(min=0)
 34:     safe_t = t.clamp(min=1e-12)
 35:     hazard = torch.where(t > 0, (safe_t/scale).pow(shape), torch.zeros_like(t))
 36:     increment_positive = (safe_t/scale).pow(shape)*torch.expm1(shape*torch.log1p(d/safe_t))
 37:     origin_increment = torch.where(d > 0, (d.clamp(min=1e-12)/scale).pow(shape), torch.zeros_like(d))
 38:     increment = torch.where(t > 0, increment_positive, origin_increment)
 39:     log_s = -hazard
 40:     return (-increment).exp().to(raw.dtype), log_s.exp().to(raw.dtype), log_s
 41: 
 42: 
 43: class PooledTimeBudgetModel(MSRModel):
 44:     def __init__(self, manager, mode='ours', scale_caps='none'):
 45:         if mode not in {'ours', 'fusion'} or scale_caps not in {'both', 'step', 'none'}:
 46:             raise ValueError((mode, scale_caps))
 47:         if {s.field_name for s in manager.settings} != set(FEATURES):
 48:             raise ValueError('PooledTimeBudgetModel requires the KR 54-field schema')
 49:         super().__init__(manager, num_domains=5)
 50:         self.mode, self.scale_caps = mode, scale_caps
 51:         self.training_strategy, self.stage = 'joint', 2
 52:         budget_settings = self.omni_embedding.get_settings(include_fields=BUDGET_PARAMETER_FIELDS)
 53:         self.budget_embedding = OmniEmbLayer(emb_settings=budget_settings)
 54:         for name, table in self.budget_embedding.emb_modules.items():
 55:             table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
 56:         self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_PARAMETER_FIELDS)
 57:         self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_TOWER_FIELDS)
 58:         self.fusion_view = self.omni_embedding.whole
 59:         self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim, 64), nn.SiLU(), nn.Linear(64, 4))
 60:         self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim, 64), nn.SiLU(),
 61:                                          nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
 62:         self.fusion_net = nn.Sequential(nn.Linear(self.fusion_view.embedding_dim, 64), nn.SiLU(),
 63:                                         nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
 64:         nn.init.constant_(self.content_net[-1].bias, 3.)
 65:         self.register_buffer('prior_stay', torch.sigmoid(torch.tensor(3.)))
 66: 
 67:     @property
 68:     def budget_modules(self):
 69:         return self.budget_embedding, self.budget_net
 70: 
 71:     def budget_raw(self, batch):
 72:         return self.budget_net(self.budget_view(batch))
 73: 
 74:     def is_capped(self, clock):
 75:         return self.scale_caps == 'both' or (self.scale_caps == 'step' and clock == 'step')
 76: 
 77:     def clock_components(self, batch):
 78:         raw = self.budget_raw(batch)
 79:         time = batch['elapsed_min_raw'].float().clamp(min=0)
 80:         step = (batch['req_idx_raw'].float()-1).clamp(min=0)
 81:         return {name: single_weibull_survival(head, t, torch.ones_like(t), self.is_capped(name))
 82:                 for name, head, t in [('time', raw[:, :2], time), ('step', raw[:, 2:], step)]}
 83: 
 84:     def budget_components(self, batch):
 85:         clocks = self.clock_components(batch)
 86:         gamma = clocks['time'][0]*clocks['step'][0]
 87:         log_s = clocks['time'][2]+clocks['step'][2]
 88:         return gamma, log_s.exp().to(gamma.dtype), log_s
 89: 
 90:     def components(self, batch):
 91:         gamma, survival, log_s = self.budget_components(batch)
 92:         stay = torch.sigmoid(self.content_net(self.content_view(batch)).squeeze(-1))
 93:         return gamma*stay, gamma, stay, survival, log_s
 94: 
 95:     def predict(self, batch):
 96:         if self.mode == 'fusion':
 97:             return torch.sigmoid(self.fusion_net(self.fusion_view(batch)).squeeze(-1))
 98:         return self.components(batch)[0]
 99: 
100:     def calculate_loss(self, batch):
101:         return F.binary_cross_entropy(self.predict(batch).clamp(1e-6, 1-1e-6), batch['y_continue'].float())
````

## File: workspace/kr_time_budget/tests/test_feature_categories.py
````python
 1: from workspace.kr_time_budget.pipeline.kr_request_schema import (
 2:     FEATURES, FEATURE_CATEGORIES, SHARED_FIELDS, BUDGET_EXCLUSIVE_FIELDS,
 3:     CONTENT_EXCLUSIVE_FIELDS, BUDGET_PARAMETER_FIELDS, BUDGET_PRIOR_FIELDS,
 4:     CONTENT_TOWER_FIELDS, CONTENT_MASK_FIELDS,
 5: )
 6: 
 7: 
 8: def test_feature_partition_and_legacy_semantics():
 9:     groups = [set(fields) for fields in FEATURE_CATEGORIES.values()]
10:     assert [len(g) for g in groups] == [30, 6, 18]
11:     assert all(not left & right for i, left in enumerate(groups) for right in groups[i+1:])
12:     assert set.union(*groups) == set(FEATURES)
13:     assert set(BUDGET_PARAMETER_FIELDS) == set(SHARED_FIELDS + BUDGET_PRIOR_FIELDS)
14:     assert len(BUDGET_PARAMETER_FIELDS) == 34 and len(CONTENT_TOWER_FIELDS) == 48
15:     assert set(CONTENT_TOWER_FIELDS) == set(SHARED_FIELDS + CONTENT_EXCLUSIVE_FIELDS)
16:     assert not set(CONTENT_TOWER_FIELDS) & set(BUDGET_EXCLUSIVE_FIELDS)
17:     assert CONTENT_MASK_FIELDS == frozenset(BUDGET_EXCLUSIVE_FIELDS)
18:     assert len(CONTENT_EXCLUSIVE_FIELDS) == 18  # 历史Content-only不能悄然变成48字段
````

## File: workspace/kr_time_budget/tests/test_v3.py
````python
 1: import polars as pl
 2: import pytest
 3: import torch
 4: from rosen.data.dataset import arrow_table_to_batch_dict
 5: from rosen.data.padding import FeatureContext, TensorFormatter
 6: from rosen.core.train.two_stage import StageSnapshot, StageReport
 7: from workspace.kr_time_budget.pipeline.build_requests_v3 import aggregate, user_priors, item_priors
 8: from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, FEATURES, USER_CATEGORY_FIELDS, BUDGET_PARAMETER_FIELDS
 9: from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3
10: from workspace.kr_time_budget.pipeline.run_exit import state_hash
11: 
12: 
13: def test_past_priors_exclude_current_and_future_and_snapshot():
14:     q=pl.DataFrame({'request_id':[0,1],'user_id':[1,1],'date':[20220505,20220507]})
15:     sess=pl.DataFrame({'user_id':[1,1,1],'end_date':[20220504,20220506,20220508],
16:                        'session_min':[10.,20.,10000.],'session_req_cnt':[2.,4.,2000.]})
17:     r=user_priors(q,sess).sort('request_id')
18:     assert r['user_hist_session_min'].to_list()==[10.,15.]
19:     obs=pl.DataFrame({'video_id':[1,1,1],'date':[20220504,20220506,20220508],'is_click':[0,1,1],'long_view':[1,0,1]})
20:     query=q.select('request_id','date').with_columns(pl.lit([1]*6).alias('slate_items'))
21:     p=item_priors(query,obs).sort('request_id')
22:     assert p['slate_prior_ctr_mean'].to_list()==[0.,.5]
23:     assert p['slate_prior_longview_mean'].to_list()==[1.,.5]
24: 
25: 
26: def test_history_only_previous_chunks_tags_are_tokens():
27:     n=18
28:     e=pl.DataFrame({'user_id':[1]*n,'session_id':[1]*n,'req_idx':[1]*6+[2]*6+[3]*6,
29:         'video_id':list(range(n)),'tag':['20,43']*n,'video_type':['NORMAL']*n,'music_type':['1']*n,
30:         'duration_ms':[60000]*n,'author_id':[1]*n,'is_rand':[0]*n,'is_click':[1]*6+[0]*12,
31:         'long_view':[0]*n,'positive':[1]*6+[0]*12,'is_hate':[0]*n,'time_ms':[i*1000 for i in range(n)],
32:         'session_start_ms':[0]*n,'play_clip':[1000]*n})
33:     r=aggregate(e)
34:     assert r['y_continue'].to_list()==[1,1,0]
35:     assert r['prev1_click_cnt'].to_list()==[0,6,0]
36:     assert r['sess_cum_click_rate'].to_list()==[0.,1.,.5]
37:     assert r['slate_unique_tag_cnt'].to_list()==[2]*3
38:     assert r['is_first_req'].to_list()==[1,0,0]
39:     changed=e.with_columns(pl.when(pl.col('req_idx')>=2).then(1).otherwise(pl.col('is_click')).alias('is_click'))
40:     a=aggregate(changed)
41:     assert a['hist_tags'][:2].to_list()==r['hist_tags'][:2].to_list()
42:     assert a['sess_cum_click_rate'][:2].to_list()==r['sess_cum_click_rate'][:2].to_list()
43: 
44: 
45: @pytest.fixture
46: def model_batch(tmp_path):
47:     mgr=request_manager(tmp_path)
48:     data={s.field_name: ([0.,1.] if not s.requires_embedding_module else ['1','2']) for s in mgr.settings}
49:     for c in ['hist_tags','slate_tags']:data[c]=['20,43','20,67']
50:     for c in ['slate_video_types','slate_music_types']:data[c]=['1,1','1,2']
51:     data['y_continue']=[0,1]
52:     raw=pl.DataFrame(data)
53:     # min_freq=10 is deliberate in production; repeat here to exercise real user vocab.
54:     raw=pl.concat([raw]*6)
55:     mgr.fit(raw.lazy())
56:     frame=mgr.transform(raw.lazy()).collect()
57:     b=TensorFormatter(FeatureContext.from_manager(mgr)).format(arrow_table_to_batch_dict(frame.to_arrow())).interaction
58:     torch.manual_seed(2026)
59:     return ExitModelV3(mgr),b,mgr
60: 
61: 
62: def test_schema_native_multitag_and_parameter_ownership(model_batch):
63:     m,b,mgr=model_batch
64:     assert len(FEATURES)==54
65:     assert mgr.get_setting('slate_tags').vocab['20']!=mgr.get_setting('slate_tags').vocab['43']
66:     assert b['slate_tags'].shape[-1]==12
67:     setting=mgr.get_setting('slate_tags')
68:     vector=m.omni_embedding(b,split_by='name')['slate_tags']
69:     native=setting.compute_tensor(b,m.omni_embedding.emb_modules)
70:     assert torch.equal(vector,native)
71:     names=[name for name,_ in m.named_parameters(remove_duplicate=False)]
72:     assert len(names)==len({id(p) for _,p in m.named_parameters(remove_duplicate=False)})
73:     hashes=[]
74:     for mode in ['content','budget','fusion','ours']:
75:         torch.manual_seed(2026);hashes.append(state_hash(ExitModelV3(mgr,mode)))
76:     assert len(set(hashes))==1
77: 
78: 
79: def test_budget_static_and_stage_transition_freezes_every_budget_parameter(model_batch):
80:     m,b,mgr=model_batch
81:     raw=m.budget_raw(b)
82:     changed={c:(v if c in BUDGET_PARAMETER_FIELDS else v*0) for c,v in b.items()}
83:     assert torch.equal(raw,m.budget_raw(changed))
84:     m.calculate_loss(b).backward()
85:     assert all(p.grad is None for p in m.content_net.parameters())
86:     snapshot=StageSnapshot.capture(m,StageReport(1,1,{},'student',{}))
87:     stage2=m.initialize_stage_two(snapshot)
88:     assert stage2 is not m
89:     before=stage2.budget_components(b)[0].detach().clone()
90:     opt=torch.optim.AdamW([p for p in stage2.parameters() if p.requires_grad],lr=.001,weight_decay=0)
91:     pred,gamma,stay,_,_=stage2.components(b)
92:     assert torch.equal(pred,gamma*stay) and not gamma.requires_grad
93:     stage2.calculate_loss(b).backward();opt.step()
94:     assert torch.equal(before,stage2.budget_components(b)[0])
95:     assert all(not p.requires_grad for module in stage2.budget_modules for p in module.parameters())
````

## File: workspace/kr_time_budget/tests/test_v8.py
````python
 1: import pytest
 2: import torch
 3: from workspace.kr_time_budget.tests.test_v3 import model_batch
 4: from workspace.kr_time_budget.pipeline.time_budget_v7 import TimeBudgetModel
 5: from workspace.kr_time_budget.pipeline.kr_request_schema import SHARED_FIELDS
 6: from workspace.kr_time_budget.pipeline.time_budget_v8 import (
 7:     DualBranchModel, single_weibull_survival, weibull_parameters, joint_parameter_groups,
 8: )
 9: 
10: 
11: @pytest.fixture
12: def inputs(model_batch):
13:     _, b, manager = model_batch
14:     return dict(b, elapsed_min_raw=b['elapsed_min'].clone(), req_idx_raw=(b['req_idx']+1).long()), manager
15: 
16: 
17: def test_legacy_reproduces_v7_state_predictions_loss_and_gradients(inputs):
18:     batch, manager = inputs
19:     torch.manual_seed(2026)
20:     old = TimeBudgetModel(manager)
21:     torch.manual_seed(2026)
22:     new = DualBranchModel(manager, scale_mapping='legacy', content_bias=3.5)
23:     assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
24:     assert torch.equal(old.predict(batch), new.predict(batch))
25:     assert torch.equal(old.calculate_loss(batch), new.calculate_loss(batch))
26:     old.calculate_loss(batch).backward()
27:     new.calculate_loss(batch).backward()
28:     for name, parameter in old.named_parameters():
29:         other = dict(new.named_parameters())[name]
30:         assert (parameter.grad is None) == (other.grad is None)
31:         if parameter.grad is not None:
32:             assert torch.equal(parameter.grad, other.grad), name
33: 
34: 
35: @pytest.mark.parametrize('floor', [1.05, .3])
36: def test_survival_matches_direct_formula_and_finite_derivatives_at_origin(floor):
37:     raw = torch.tensor([[0., -2.]]*6, dtype=torch.float64, requires_grad=True)
38:     t = torch.tensor([0., .01, .5, 1., 10., 100.], dtype=torch.float64)
39:     gamma, survival, log_s = single_weibull_survival(raw, t, torch.ones_like(t), 'log', floor)
40:     scale, shape = weibull_parameters(raw, 'log', floor)
41:     torch.testing.assert_close(scale, torch.full_like(scale, torch.exp(torch.tensor(1.6, dtype=torch.float64))))
42:     direct_log = (t/scale).pow(shape)-((t+1)/scale).pow(shape)
43:     torch.testing.assert_close(gamma.log(), direct_log, rtol=1e-10, atol=1e-10)
44:     torch.testing.assert_close(survival, log_s.exp())
45:     assert (log_s[1:] < log_s[:-1]).all()
46:     assert ((gamma[1:] < gamma[:-1]) if floor > 1 else (gamma[1:] > gamma[:-1])).all()
47:     (gamma.sum()+survival.sum()).backward()
48:     assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
49:     zero_raw = raw.detach().clone().requires_grad_(True)
50:     zero_gamma, _, _ = single_weibull_survival(zero_raw, t, torch.zeros_like(t), 'log', floor)
51:     assert torch.equal(zero_gamma, torch.ones_like(t))
52:     zero_gamma.sum().backward()
53:     assert torch.isfinite(zero_raw.grad).all()
54: 
55: 
56: @pytest.mark.parametrize('floor,bounded,bias', [(1.05, False, 3.5), (1.05, False, 3.), (.3, False, 3.), (1.05, True, 3.)])
57: def test_single_joint_loss_updates_both_towers_and_lr_decay_ownership(inputs, floor, bounded, bias):
58:     batch, manager = inputs
59:     model = DualBranchModel(manager, shape_floor=floor, content_bias=bias, bounded_content=bounded)
60:     expected = model.components(batch)[1]*model.components(batch)[2]
61:     assert torch.equal(model.predict(batch), expected)
62:     assert torch.equal(model.calculate_loss(batch), torch.nn.functional.binary_cross_entropy(
63:         expected.clamp(1e-6, 1-1e-6), batch['y_continue'].float()))
64:     groups = joint_parameter_groups(model, .005)
65:     ownership = {id(p): group for group in groups for p in group['params']}
66:     for module in model.budget_modules:
67:         assert all(ownership[id(p)]['lr'] == .005 for p in module.parameters())
68:     assert all(ownership[id(p)]['lr'] == .001 for p in model.content_net.parameters())
69:     assert ownership[id(model.budget_net[0].weight)]['weight_decay'] == 1e-6
70:     assert ownership[id(model.budget_net[0].bias)]['weight_decay'] == 0
71:     for module in model.modules():
72:         if isinstance(module, torch.nn.Embedding):
73:             assert ownership[id(module.weight)]['weight_decay'] == 0
74:     watched = [model.budget_net[-1].weight, model.content_net[-1].weight,
75:                model.budget_embedding.emb_modules['user_id'].embedding.weight,
76:                model.omni_embedding.emb_modules['user_id'].embedding.weight]
77:     before = [p.detach().clone() for p in watched]
78:     model.calculate_loss(batch).backward()
79:     assert all(p.requires_grad and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in watched)
80:     torch.optim.AdamW(groups, weight_decay=0).step()
81:     assert all(not torch.equal(old, p) for old, p in zip(before, watched))
82:     assert set(model.budget_view.field_slices).intersection(model.content_view.field_slices) == set(SHARED_FIELDS)
83: 
84: 
85: def test_bounded_content_has_explicit_range_and_same_initial_tensors(inputs):
86:     _, manager = inputs
87:     torch.manual_seed(2026)
88:     plain = DualBranchModel(manager)
89:     torch.manual_seed(2026)
90:     bounded = DualBranchModel(manager, bounded_content=True)
91:     assert all(torch.equal(value, bounded.state_dict()[name]) for name, value in plain.state_dict().items())
92:     logits = torch.tensor([-1e6, 3., 1e6], requires_grad=True)
93:     stay = bounded.content_stay(logits)
94:     torch.testing.assert_close(stay, torch.sigmoid(torch.tensor([2.5, 3., 3.5])))
95:     stay[1].backward()
96:     assert logits.grad[1] > 0
````

## File: workspace/kr_time_budget/tests/test_v9.py
````python
 1: import pytest
 2: import torch
 3: from workspace.kr_time_budget.tests.test_v3 import model_batch
 4: from workspace.kr_time_budget.pipeline.time_budget_v8 import DualBranchModel, joint_parameter_groups
 5: from workspace.kr_time_budget.pipeline.time_budget_v9 import (
 6:     PooledTimeBudgetModel, weibull_parameters, single_weibull_survival,
 7: )
 8: 
 9: 
10: @pytest.fixture
11: def inputs(model_batch):
12:     _, b, manager = model_batch
13:     return dict(b, elapsed_min_raw=b['elapsed_min'].clone(), req_idx_raw=(b['req_idx']+1).long()), manager
14: 
15: 
16: def test_capped_v8_exact_state_prediction_loss_and_gradients(inputs):
17:     b, manager = inputs
18:     torch.manual_seed(2026)
19:     old = DualBranchModel(manager, shape_floor=.3)
20:     torch.manual_seed(2026)
21:     new = PooledTimeBudgetModel(manager, scale_caps='both')
22:     assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
23:     assert torch.equal(old.predict(b), new.predict(b))
24:     old.calculate_loss(b).backward()
25:     new.calculate_loss(b).backward()
26:     for name, p in old.named_parameters():
27:         other = dict(new.named_parameters())[name]
28:         assert (p.grad is None) == (other.grad is None)
29:         if p.grad is not None:
30:             assert torch.equal(p.grad, other.grad)
31: 
32: 
33: def test_uncapping_removes_both_plateaus_without_changing_inside_values():
34:     raw = torch.tensor([[-3., -1.], [0., -1.], [4., -1.], [6., -1.]], dtype=torch.float64, requires_grad=True)
35:     capped = weibull_parameters(raw, True)[0]
36:     uncapped = weibull_parameters(raw, False)[0]
37:     assert torch.equal(capped[1:3], uncapped[1:3])
38:     assert uncapped[0] < capped[0] and uncapped[-1] > capped[-1]
39:     dc = torch.autograd.grad(capped.sum(), raw, retain_graph=True)[0][:, 0]
40:     du = torch.autograd.grad(uncapped.sum(), raw)[0][:, 0]
41:     assert dc[0] == 0 and dc[-1] == 0 and (du > 0).all()
42: 
43: 
44: @pytest.mark.parametrize('caps', ['both', 'step', 'none'])
45: def test_same_initialization_single_product_and_both_tower_updates(inputs, caps):
46:     b, manager = inputs
47:     torch.manual_seed(2026)
48:     control = PooledTimeBudgetModel(manager, scale_caps='both')
49:     torch.manual_seed(2026)
50:     model = PooledTimeBudgetModel(manager, scale_caps=caps)
51:     assert all(torch.equal(v, model.state_dict()[k]) for k, v in control.state_dict().items())
52:     assert model.is_capped('time') == (caps == 'both')
53:     assert model.is_capped('step') == (caps != 'none')
54:     p, gamma, stay, _, _ = model.components(b)
55:     assert torch.equal(p, gamma*stay)
56:     assert torch.equal(model.calculate_loss(b), torch.nn.functional.binary_cross_entropy(
57:         p.clamp(1e-6, 1-1e-6), b['y_continue'].float()))
58:     watched = [model.budget_net[-1].weight, model.content_net[-1].weight,
59:                model.budget_embedding.emb_modules['user_id'].embedding.weight,
60:                model.omni_embedding.emb_modules['user_id'].embedding.weight]
61:     before = [x.detach().clone() for x in watched]
62:     model.calculate_loss(b).backward()
63:     assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in watched)
64:     torch.optim.AdamW(joint_parameter_groups(model, .005), weight_decay=0).step()
65:     assert all(not torch.equal(old, new) for old, new in zip(before, watched))
66: 
67: 
68: def test_uncapped_survival_origin_extreme_clock_and_gradient():
69:     raw = torch.tensor([[6., -2.]]*5, dtype=torch.float64, requires_grad=True)
70:     t = torch.tensor([0., 1., 100., 879., 10000.], dtype=torch.float64)
71:     gamma, s, log_s = single_weibull_survival(raw, t, torch.ones_like(t), False)
72:     scale, shape = weibull_parameters(raw, False)
73:     direct = torch.exp((t/scale).pow(shape)-((t+1)/scale).pow(shape))
74:     torch.testing.assert_close(gamma, direct, rtol=1e-10, atol=1e-10)
75:     assert (log_s[1:] < log_s[:-1]).all()
76:     assert (gamma[1:] > gamma[:-1]).all()  # DFR allows rising conditional survival.
77:     (gamma.sum()+s.sum()).backward()
78:     assert torch.isfinite(raw.grad).all() and raw.grad[:, 0].abs().sum() > 0
````

## File: workspace/kr_time_budget/debug_dataflow.py
````python
  1: """用一个真实会话逐段调试 KR 数据流，默认只读、不训练。
  2: 
  3: 直接运行（工作目录不限）：
  4:     uv run python workspace/kr_time_budget/debug_dataflow.py
  5: 命令行断点：
  6:     uv run python workspace/kr_time_budget/debug_dataflow.py --stop-at batch
  7:     uv run python workspace/kr_time_budget/debug_dataflow.py --pdb
  8: IDE：在标记 DEBUG 1～7 的下一行设置断点，再 Debug 当前文件。
  9: 
 10: 所有中间表和 tensor 留在 main 的局部变量中；使用 step into 可以进入正式实现。
 11: 默认重算聚合与过去日先验，与落盘结果核对；--skip-priors 可跳过全体曝光的先验扫描。
 12: """
 13: import argparse
 14: from pathlib import Path
 15: import sys
 16: 
 17: # 支持在 IDE 中直接执行本文件，不依赖 IDE 的 working directory/PYTHONPATH。
 18: REPO_ROOT = Path(__file__).resolve().parents[2]
 19: if str(REPO_ROOT) not in sys.path:
 20:     sys.path.insert(0, str(REPO_ROOT))
 21: 
 22: import numpy as np
 23: import polars as pl
 24: import torch
 25: from torch.utils.data import DataLoader
 26: from rosen.data.dataset import ParquetStreamDataset
 27: from rosen.datasets.kuairand import KuaiRandDataset
 28: from workspace.kr_time_budget.pipeline.build_requests import sessionize
 29: from workspace.kr_time_budget.pipeline.build_requests_v3 import aggregate, user_priors, item_priors
 30: from workspace.kr_time_budget.pipeline.kr_request_schema import (
 31:     request_manager, FEATURES, LOG1P_FIELDS, EXTRA_COLUMNS, USER_CATEGORY_FIELDS,
 32:     FEATURE_CATEGORIES, BUDGET_PARAMETER_FIELDS, SESSION_CLOCK_FIELDS, CONTENT_TOWER_FIELDS,
 33: )
 34: from workspace.kr_time_budget.pipeline.time_budget_v4 import weibull_parameters
 35: from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5, CONTENT_CLOCK_FIELDS
 36: 
 37: WORKSPACE = Path(__file__).resolve().parent
 38: STAGES = ['events', 'requests', 'priors', 'encoded', 'batch', 'forward', 'backward']
 39: RAW_COLUMNS = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click', 'is_like',
 40:                'is_follow', 'is_comment', 'is_forward', 'is_hate', 'long_view', 'play_time_ms',
 41:                'duration_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter', 'is_rand', 'tab']
 42: 
 43: 
 44: def compare_frames(actual, expected, columns):
 45:     """比较前必须按 request_id 对齐；出错会明确指出字段。"""
 46:     assert actual['request_id'].to_list() == expected['request_id'].to_list()
 47:     for column in columns:
 48:         left, right = actual[column], expected[column]
 49:         if left.dtype.is_numeric():
 50:             np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), rtol=1e-6, atol=1e-7, err_msg=column)
 51:         else:
 52:             assert left.to_list() == right.to_list(), column
 53: 
 54: 
 55: def main():
 56:     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
 57:     parser.add_argument('--data', type=Path, default=WORKSPACE/'data/v3')
 58:     parser.add_argument('--source', type=Path, default=KuaiRandDataset.DATA_KUAI_1K)
 59:     parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
 60:     parser.add_argument('--request-id', type=int, help='挑选该Request所在的完整Session；优先于--split')
 61:     parser.add_argument('--batch-size', type=int, default=8, help='前向最多几条Request；会包含指定Request')
 62:     parser.add_argument('--checkpoint', type=Path, default=WORKSPACE/'runs/exit_v5/ours/stage_two/best.pt')
 63:     parser.add_argument('--stop-at', nargs='+', choices=STAGES, default=[], help='在这些阶段停入pdb，可写多个')
 64:     parser.add_argument('--pdb', action='store_true', help='等价于--stop-at forward')
 65:     parser.add_argument('--skip-priors', action='store_true', help='使用落盘先验，不重新扫描全部历史曝光')
 66:     parser.add_argument('--backward', action='store_true', help='额外执行一次loss.backward；不执行optimizer.step')
 67:     args = parser.parse_args()
 68:     if args.batch_size < 1:
 69:         parser.error('--batch-size必须>=1')
 70:     if 'backward' in args.stop_at:
 71:         args.backward = True
 72:     if args.pdb:
 73:         args.stop_at.append('forward')
 74:     torch.set_num_threads(4)
 75:     pl.Config.set_tbl_rows(12)
 76:     pl.Config.set_tbl_cols(12)
 77: 
 78:     # 默认挑一个3～12屏的会话，避免只有一屏、看不到历史和继续标签。
 79:     features_scan = pl.scan_parquet(args.data/'features_raw.parquet')
 80:     if args.request_id is not None:
 81:         focus = features_scan.filter(pl.col('request_id') == args.request_id).collect()
 82:     else:
 83:         focus = features_scan.filter((pl.col('split') == args.split) & (pl.col('req_idx') == 1)
 84:                                      & pl.col('session_req_cnt').is_between(3, 12)).head(1).collect()
 85:     if focus.height != 1:
 86:         raise ValueError('没有找到目标Request；检查request_id或split')
 87:     request_id = focus['request_id'].item()
 88:     session_id = focus['session_id'].item()
 89:     user_id = focus['user_id'].item()
 90:     split = focus['split'].item()
 91:     print(f'目标：user_id={user_id}, session_id={session_id}, request_id={request_id}, split={split}', flush=True)
 92: 
 93:     # DEBUG 1：原始19列事件→排序/切场/倒切。event_id为全局事件键。
 94:     # events.parquet已完成全19列去重；本脚本不重新读取全部CSV做全局去重。
 95:     raw_user_events = pl.scan_parquet(args.data/'events.parquet').filter(pl.col('user_id') == user_id).collect()
 96:     sessionized_user = sessionize(raw_user_events.select('event_id', *RAW_COLUMNS, 'source_order', 'source_row'))
 97:     # 只重放一个用户，所以sessionized_user的session_id是局部编号，不可直接比较全局编号。
 98:     raw_events = raw_user_events.filter(pl.col('session_id') == session_id).sort('event_idx')
 99:     replayed_events = sessionized_user.filter(pl.col('event_id').is_in(raw_events['event_id'].implode())).sort('event_idx')
100:     assert replayed_events['req_idx'].equals(raw_events['req_idx'])
101:     assert replayed_events['session_n'].equals(raw_events['session_n'])
102:     source_files = sorted(args.source.glob('log_*.csv'))
103:     provenance = raw_events.select('event_id', 'source_order', 'source_row').with_columns(
104:         pl.col('source_order').replace_strict({i: str(path) for i, path in enumerate(source_files)}).alias('source_file'))
105:     # source_row是CSV数据行的0起始序号（不含表头）；req_idx=0是头部余数。
106:     print('\n[1 events] 当前场的完整事件（含req0）：')
107:     print(raw_events.select('event_id', 'event_idx', 'req_idx', 'video_id', 'time_ms', 'play_time_ms', 'is_click', 'long_view'))
108:     if 'events' in args.stop_at:
109:         breakpoint()
110: 
111:     # DEBUG 2：连接视频侧表→正式aggregate函数→Request及前序历史。
112:     video_features = pl.scan_csv(args.source/'video_features_basic_1k.csv', schema_overrides={'tag': pl.String}).filter(
113:         pl.col('video_id').is_in(raw_events['video_id'].unique().implode())).select(
114:             'video_id', 'author_id', 'tag', 'video_type', 'music_type').collect()
115:     joined_events = raw_events.join(video_features, on='video_id', how='left', validate='m:1', maintain_order='left').with_columns(
116:         pl.col('tag').fill_null('__MISSING__'), pl.col('video_type').fill_null('__MISSING__'),
117:         pl.col('music_type').cast(pl.Int64).cast(pl.String).fill_null('__MISSING__'), pl.col('author_id').fill_null(-1),
118:         pl.any_horizontal(*[pl.col(c) == 1 for c in ['is_like', 'is_comment', 'is_follow', 'is_forward']]).cast(pl.Int8).alias('positive'))
119:     requests_saved = pl.scan_parquet(args.data/'requests_base.parquet').filter(pl.col('session_id') == session_id).sort('req_idx').collect()
120: 
121:     raw_user_profile = pl.read_csv(args.source/'user_features_1k.csv').filter(pl.col('user_id') == user_id)
122:     user_profile = raw_user_profile.select('user_id', *USER_CATEGORY_FIELDS).with_columns(
123:         pl.col('is_live_streamer').eq(1).cast(pl.Int8),
124:         *[pl.col(f'onehot_feat{i}').cast(pl.Int64).fill_null(-1) for i in range(18)])
125:     assert user_profile.height == 1
126:     for name in USER_CATEGORY_FIELDS:
127:         assert requests_saved[name][0] == user_profile[name][0], name
128: 
129:     requests_rebuilt = aggregate(joined_events).join(requests_saved.select('req_idx', 'request_id'), on='req_idx', validate='1:1').sort('req_idx')
130:     compare_frames(requests_rebuilt, requests_saved, [c for c in requests_rebuilt.columns if c != 'request_id'])
131:     event_membership = pl.scan_parquet(args.data/'request_events.parquet').filter(pl.col('session_id') == session_id).sort('event_idx').collect()
132:     assert event_membership.filter(pl.col('request_id').is_not_null()).group_by('request_id').len()['len'].to_list() == [6]*requests_saved.height
133:     print('\n[2 requests] 聚合重放与requests_base一致：')
134:     print(requests_rebuilt.select('request_id', 'req_idx', 'y_continue', 'elapsed_min', 'prev1_click_cnt', 'sess_cum_click_rate', 'slate_tags', 'hist_tags'))
135:     if 'requests' in args.stop_at:
136:         breakpoint()
137: 
138:     # DEBUG 3：全部历史→严格过去日/0506快照→平滑先验。
139:     feature_rows = features_scan.filter(pl.col('session_id') == session_id).sort('req_idx').collect()
140:     prior_query = requests_saved.select('request_id', 'user_id', pl.when(pl.col('split') == 'train').then(
141:         pl.col('session_date')).otherwise(pl.lit(20220507)).alias('date'))
142:     user_prior_rows = item_prior_rows = prior_observations = all_sessions = None
143:     if not args.skip_priors:
144:         all_sessions = pl.read_parquet(args.data/'sessions.parquet')
145:         user_prior_rows = user_priors(prior_query, all_sessions).sort('request_id')
146:         # 总体平滑率必须来自全体过去曝光，不能仅筛选当前6个视频就计算总体率。
147:         prior_observations = pl.scan_parquet(args.data/'events.parquet').filter(pl.col('date') <= 20220506).select(
148:             'video_id', 'date', 'is_click', 'long_view').collect()
149:         item_query = prior_query.select('request_id', 'date').join(requests_saved.select('request_id', 'slate_items'), on='request_id', validate='1:1')
150:         item_prior_rows = item_priors(item_query, prior_observations).sort('request_id')
151:         prior_rows = user_prior_rows.join(item_prior_rows, on='request_id', validate='1:1')
152:         compare_frames(prior_rows, feature_rows.sort('request_id'), [c for c in prior_rows.columns if c != 'request_id'])
153:     print('\n[3 priors] 查询date之前的历史可用；valid/test查询20220507，实际只用<=0506：')
154:     print(prior_query)
155:     print(feature_rows.select('request_id', 'user_prior_n', 'user_hist_session_min', 'slate_prior_ctr_mean', 'item_prior_cold_fraction'))
156:     if 'priors' in args.stop_at:
157:         breakpoint()
158: 
159:     # DEBUG 4：Float32/log1p→已train-fit的Schema.transform→编码parquet核对。
160:     # 不在这个小样本上重新fit，否则词表与正式checkpoint不一致。
161:     manager = request_manager(args.data/'schema', load=True)
162:     schema_input = feature_rows.with_columns(*[pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32) for c in LOG1P_FIELDS])
163:     output_fields = list(dict.fromkeys(manager.fields() + EXTRA_COLUMNS))
164:     encoded = manager.transform(schema_input.lazy()).select(output_fields).collect()
165:     encoded_saved = pl.scan_parquet(args.data/f'{split}.parquet').filter(pl.col('session_id') == session_id).sort('req_idx_raw').collect()
166:     compare_frames(encoded, encoded_saved, output_fields)
167:     tag_setting = manager.get_setting('slate_tags')
168:     tag_vocab = tag_setting.vocab
169:     tag_tokens = feature_rows['slate_tags'].str.split(',').to_list()
170:     encoded_tag_ids = encoded['slate_tags'].to_list()
171:     print('\n[4 encoded] Schema转换与正式parquet一致；标签先拆分，再转ID：')
172:     print(encoded.select('request_id', 'user_id', 'user_id_raw', 'slate_tags', 'hist_tags', 'elapsed_min'))
173:     if 'encoded' in args.stop_at:
174:         breakpoint()
175: 
176:     # DEBUG 5：正式ParquetStreamDataset + DataLoader(batch_size=None)→Interaction。
177:     # 用LazyFrame选取小片段，避免改动正式parquet；formatter/padding仍走同一实现。
178:     focus_position = encoded['request_id'].to_list().index(request_id)
179:     batch_start = max(0, focus_position - args.batch_size//2)
180:     encoded_slice = encoded.slice(batch_start, args.batch_size)
181:     dataset = ParquetStreamDataset(encoded_slice.lazy(), manager, batch_size=len(encoded_slice), shuffle=False,
182:                                    drop_last=False, extra_col_names=EXTRA_COLUMNS)
183:     interaction = next(iter(DataLoader(dataset, batch_size=None, num_workers=0)))
184:     batch = interaction.interaction
185:     batch_shapes = {name: (tuple(value.shape), str(value.dtype)) for name, value in batch.items()}
186:     assert len(FEATURES) == 54 and batch['slate_tags'].shape[1] == 12 and batch['hist_tags'].shape[1] == 15
187:     print('\n[5 batch] 重点变量：interaction、batch、batch_shapes；例如：')
188:     for name in ['user_id', 'slate_tags', 'hist_tags', 'elapsed_min', 'req_idx', 'y_continue']:
189:         print(name, batch_shapes[name], batch[name][:2])
190:     if 'batch' in args.stop_at:
191:         breakpoint()
192: 
193:     # DEBUG 6：实际checkpoint→逐字段Embedding→塔路由→scale/shape→gamma×stay。
194:     model = ExitModelV5(manager, mode='ours')
195:     checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
196:     model.load_state_dict(checkpoint['state_dict'])
197:     model.set_stage(2)
198:     model.eval()
199:     field_embeddings = model.omni_embedding(batch, split_by='name')
200:     budget_x = model.budget_view(batch)
201:     content_x = model.content_x(batch)
202:     # 给每个字段标出拼接后的切片；可直接看两个时钟切片是否全0。
203:     field_slices = {}
204:     offset = 0
205:     for name, value in field_embeddings.items():
206:         field_slices[name] = slice(offset, offset + value.shape[-1])
207:         offset += value.shape[-1]
208:     feature_categories = {name: category for category, names in FEATURE_CATEGORIES.items() for name in names}
209:     # routing仍表示实际加载的v5，target_routing单独展示新设计，不能把归属当成已消费。
210:     routing = {name: {'category': feature_categories[name], 'budget': name in BUDGET_PARAMETER_FIELDS,
211:                       'budget_formula': name == 'elapsed_min', 'content': name not in CONTENT_CLOCK_FIELDS,
212:                       'slice': (section.start, section.stop)} for name, section in field_slices.items()}
213:     target_routing = {name: {'category': feature_categories[name],
214:                             'budget_parameters': name in BUDGET_PARAMETER_FIELDS,
215:                             'budget_clock': name in SESSION_CLOCK_FIELDS,
216:                             'content': name in CONTENT_TOWER_FIELDS,
217:                             'clock_formula_pending': name == 'req_idx'} for name in FEATURES}
218:     for name in CONTENT_CLOCK_FIELDS:
219:         assert torch.count_nonzero(content_x[:, field_slices[name]]) == 0
220:     budget_raw = model.budget_net(budget_x)
221:     scale, shape = weibull_parameters(budget_raw)
222:     gamma, survival, log_survival = model.budget_components(batch)
223:     content_logits = model.content_net(content_x).squeeze(-1)
224:     stay = torch.sigmoid(content_logits)
225:     prediction = gamma.detach() * stay
226:     torch.testing.assert_close(prediction, model.predict(batch), rtol=0, atol=0)
227:     loss = model.calculate_loss(batch)
228:     prediction_table = pl.DataFrame({'request_id': batch['request_id'].numpy(), 'label': batch['y_continue'].numpy(),
229:         'elapsed_min': batch['elapsed_min'].numpy(), 'scale': scale.detach().numpy(), 'shape': shape.detach().numpy(),
230:         'gamma': gamma.detach().numpy(), 'stay': stay.detach().numpy(), 'prediction': prediction.detach().numpy()})
231:     print('\n[6 forward] loss =', loss.item())
232:     print(prediction_table)
233:     print('checkpoint selection:', checkpoint.get('selection'))
234:     # 训练入口仅保存test预测，因此仅对test样本做同Request核对。
235:     saved_predictions_path = args.checkpoint.parent.parent/'predictions.parquet'
236:     saved_predictions = None
237:     if split == 'test' and saved_predictions_path.exists():
238:         saved_predictions = prediction_table.select('request_id').join(
239:             pl.read_parquet(saved_predictions_path, columns=['request_id', 'prediction']),
240:             on='request_id', how='left', validate='1:1', maintain_order='left')
241:         assert saved_predictions['prediction'].null_count() == 0, '保存的test预测未覆盖当前Request'
242:         np.testing.assert_allclose(prediction.detach().numpy(), saved_predictions['prediction'].to_numpy(), rtol=1e-6, atol=1e-7)
243:     if 'forward' in args.stop_at:
244:         breakpoint()
245: 
246:     # DEBUG 7（可选）：反向梯度；不创建optimizer，不更新参数，不写回checkpoint。
247:     gradients = None
248:     if args.backward:
249:         model.zero_grad(set_to_none=True)
250:         loss.backward()
251:         gradients = {name: None if value.grad is None else value.grad.norm().item() for name, value in model.named_parameters()}
252:         assert all(p.grad is None for module in model.budget_modules for p in module.parameters())
253:         print('\n[7 backward] 预算路径无梯度；内容末层梯度norm =', gradients['content_net.4.weight'])
254:         if 'backward' in args.stop_at:
255:             breakpoint()
256:     print('\n完成。中间表与正式产物核对通过；未更新参数、未改写数据。')
257: 
258: 
259: if __name__ == '__main__':
260:     main()
````

## File: workspace/kr_time_budget/feature_categories.json
````json
  1: {
  2:   "contract": "shared30_budget6_content18",
  3:   "data": "workspace/kr_time_budget/data/v3",
  4:   "dataset_rebuild_required": false,
  5:   "dataset_manifest_sha256": "6b25631f161f19d6862c97b95021f3e345d2bba4e4f4fd6f487215c831ed1109",
  6:   "categories": {
  7:     "shared": [
  8:       "user_id",
  9:       "user_active_degree",
 10:       "is_live_streamer",
 11:       "is_video_author",
 12:       "friend_user_num_range",
 13:       "register_days_range",
 14:       "follow_user_num_range",
 15:       "fans_user_num_range",
 16:       "onehot_feat0",
 17:       "onehot_feat1",
 18:       "onehot_feat2",
 19:       "onehot_feat3",
 20:       "onehot_feat4",
 21:       "onehot_feat5",
 22:       "onehot_feat6",
 23:       "onehot_feat7",
 24:       "onehot_feat8",
 25:       "onehot_feat9",
 26:       "onehot_feat10",
 27:       "onehot_feat11",
 28:       "onehot_feat12",
 29:       "onehot_feat13",
 30:       "onehot_feat14",
 31:       "onehot_feat15",
 32:       "onehot_feat16",
 33:       "onehot_feat17",
 34:       "tab",
 35:       "day_of_week",
 36:       "is_weekend",
 37:       "start_hour"
 38:     ],
 39:     "budget_exclusive": [
 40:       "user_hist_session_min",
 41:       "user_hist_req_cnt",
 42:       "day_prev_sess_cnt",
 43:       "day_prev_elapsed_min",
 44:       "elapsed_min",
 45:       "req_idx"
 46:     ],
 47:     "content_exclusive": [
 48:       "is_first_req",
 49:       "prev1_click_cnt",
 50:       "prev1_longview_cnt",
 51:       "prev1_pos_cnt",
 52:       "sess_cum_click_rate",
 53:       "sess_cum_longview_rate",
 54:       "sess_cum_pos_rate",
 55:       "sess_cum_hate_cnt",
 56:       "hist_tags",
 57:       "slate_tags",
 58:       "slate_video_types",
 59:       "slate_music_types",
 60:       "slate_prior_ctr_mean",
 61:       "slate_prior_longview_mean",
 62:       "slate_dur_mean_min",
 63:       "slate_unique_tag_cnt",
 64:       "slate_unique_author_cnt",
 65:       "slate_has_rand"
 66:     ]
 67:   },
 68:   "subgroups": {
 69:     "shared_user": [
 70:       "user_id",
 71:       "user_active_degree",
 72:       "is_live_streamer",
 73:       "is_video_author",
 74:       "friend_user_num_range",
 75:       "register_days_range",
 76:       "follow_user_num_range",
 77:       "fans_user_num_range",
 78:       "onehot_feat0",
 79:       "onehot_feat1",
 80:       "onehot_feat2",
 81:       "onehot_feat3",
 82:       "onehot_feat4",
 83:       "onehot_feat5",
 84:       "onehot_feat6",
 85:       "onehot_feat7",
 86:       "onehot_feat8",
 87:       "onehot_feat9",
 88:       "onehot_feat10",
 89:       "onehot_feat11",
 90:       "onehot_feat12",
 91:       "onehot_feat13",
 92:       "onehot_feat14",
 93:       "onehot_feat15",
 94:       "onehot_feat16",
 95:       "onehot_feat17"
 96:     ],
 97:     "shared_context": [
 98:       "tab",
 99:       "day_of_week",
100:       "is_weekend",
101:       "start_hour"
102:     ],
103:     "budget_priors": [
104:       "user_hist_session_min",
105:       "user_hist_req_cnt",
106:       "day_prev_sess_cnt",
107:       "day_prev_elapsed_min"
108:     ],
109:     "clocks": [
110:       "elapsed_min",
111:       "req_idx"
112:     ],
113:     "content_history": [
114:       "is_first_req",
115:       "prev1_click_cnt",
116:       "prev1_longview_cnt",
117:       "prev1_pos_cnt",
118:       "sess_cum_click_rate",
119:       "sess_cum_longview_rate",
120:       "sess_cum_pos_rate",
121:       "sess_cum_hate_cnt",
122:       "hist_tags"
123:     ],
124:     "content_slate": [
125:       "slate_tags",
126:       "slate_video_types",
127:       "slate_music_types",
128:       "slate_prior_ctr_mean",
129:       "slate_prior_longview_mean",
130:       "slate_dur_mean_min",
131:       "slate_unique_tag_cnt",
132:       "slate_unique_author_cnt",
133:       "slate_has_rand"
134:     ]
135:   },
136:   "target_routes": {
137:     "budget_parameters": [
138:       "user_id",
139:       "user_active_degree",
140:       "is_live_streamer",
141:       "is_video_author",
142:       "friend_user_num_range",
143:       "register_days_range",
144:       "follow_user_num_range",
145:       "fans_user_num_range",
146:       "onehot_feat0",
147:       "onehot_feat1",
148:       "onehot_feat2",
149:       "onehot_feat3",
150:       "onehot_feat4",
151:       "onehot_feat5",
152:       "onehot_feat6",
153:       "onehot_feat7",
154:       "onehot_feat8",
155:       "onehot_feat9",
156:       "onehot_feat10",
157:       "onehot_feat11",
158:       "onehot_feat12",
159:       "onehot_feat13",
160:       "onehot_feat14",
161:       "onehot_feat15",
162:       "onehot_feat16",
163:       "onehot_feat17",
164:       "tab",
165:       "day_of_week",
166:       "is_weekend",
167:       "start_hour",
168:       "user_hist_session_min",
169:       "user_hist_req_cnt",
170:       "day_prev_sess_cnt",
171:       "day_prev_elapsed_min"
172:     ],
173:     "budget_scope": [
174:       "user_id",
175:       "user_active_degree",
176:       "is_live_streamer",
177:       "is_video_author",
178:       "friend_user_num_range",
179:       "register_days_range",
180:       "follow_user_num_range",
181:       "fans_user_num_range",
182:       "onehot_feat0",
183:       "onehot_feat1",
184:       "onehot_feat2",
185:       "onehot_feat3",
186:       "onehot_feat4",
187:       "onehot_feat5",
188:       "onehot_feat6",
189:       "onehot_feat7",
190:       "onehot_feat8",
191:       "onehot_feat9",
192:       "onehot_feat10",
193:       "onehot_feat11",
194:       "onehot_feat12",
195:       "onehot_feat13",
196:       "onehot_feat14",
197:       "onehot_feat15",
198:       "onehot_feat16",
199:       "onehot_feat17",
200:       "tab",
201:       "day_of_week",
202:       "is_weekend",
203:       "start_hour",
204:       "user_hist_session_min",
205:       "user_hist_req_cnt",
206:       "day_prev_sess_cnt",
207:       "day_prev_elapsed_min",
208:       "elapsed_min",
209:       "req_idx"
210:     ],
211:     "content": [
212:       "user_id",
213:       "user_active_degree",
214:       "is_live_streamer",
215:       "is_video_author",
216:       "friend_user_num_range",
217:       "register_days_range",
218:       "follow_user_num_range",
219:       "fans_user_num_range",
220:       "onehot_feat0",
221:       "onehot_feat1",
222:       "onehot_feat2",
223:       "onehot_feat3",
224:       "onehot_feat4",
225:       "onehot_feat5",
226:       "onehot_feat6",
227:       "onehot_feat7",
228:       "onehot_feat8",
229:       "onehot_feat9",
230:       "onehot_feat10",
231:       "onehot_feat11",
232:       "onehot_feat12",
233:       "onehot_feat13",
234:       "onehot_feat14",
235:       "onehot_feat15",
236:       "onehot_feat16",
237:       "onehot_feat17",
238:       "tab",
239:       "day_of_week",
240:       "is_weekend",
241:       "start_hour",
242:       "is_first_req",
243:       "prev1_click_cnt",
244:       "prev1_longview_cnt",
245:       "prev1_pos_cnt",
246:       "sess_cum_click_rate",
247:       "sess_cum_longview_rate",
248:       "sess_cum_pos_rate",
249:       "sess_cum_hate_cnt",
250:       "hist_tags",
251:       "slate_tags",
252:       "slate_video_types",
253:       "slate_music_types",
254:       "slate_prior_ctr_mean",
255:       "slate_prior_longview_mean",
256:       "slate_dur_mean_min",
257:       "slate_unique_tag_cnt",
258:       "slate_unique_author_cnt",
259:       "slate_has_rand"
260:     ],
261:     "content_mask": [
262:       "day_prev_elapsed_min",
263:       "day_prev_sess_cnt",
264:       "elapsed_min",
265:       "req_idx",
266:       "user_hist_req_cnt",
267:       "user_hist_session_min"
268:     ],
269:     "fusion": [
270:       "user_id",
271:       "user_active_degree",
272:       "is_live_streamer",
273:       "is_video_author",
274:       "friend_user_num_range",
275:       "register_days_range",
276:       "follow_user_num_range",
277:       "fans_user_num_range",
278:       "onehot_feat0",
279:       "onehot_feat1",
280:       "onehot_feat2",
281:       "onehot_feat3",
282:       "onehot_feat4",
283:       "onehot_feat5",
284:       "onehot_feat6",
285:       "onehot_feat7",
286:       "onehot_feat8",
287:       "onehot_feat9",
288:       "onehot_feat10",
289:       "onehot_feat11",
290:       "onehot_feat12",
291:       "onehot_feat13",
292:       "onehot_feat14",
293:       "onehot_feat15",
294:       "onehot_feat16",
295:       "onehot_feat17",
296:       "user_hist_session_min",
297:       "user_hist_req_cnt",
298:       "tab",
299:       "day_of_week",
300:       "is_weekend",
301:       "start_hour",
302:       "day_prev_sess_cnt",
303:       "day_prev_elapsed_min",
304:       "elapsed_min",
305:       "req_idx",
306:       "is_first_req",
307:       "prev1_click_cnt",
308:       "prev1_longview_cnt",
309:       "prev1_pos_cnt",
310:       "sess_cum_click_rate",
311:       "sess_cum_longview_rate",
312:       "sess_cum_pos_rate",
313:       "sess_cum_hate_cnt",
314:       "hist_tags",
315:       "slate_tags",
316:       "slate_video_types",
317:       "slate_music_types",
318:       "slate_prior_ctr_mean",
319:       "slate_prior_longview_mean",
320:       "slate_dur_mean_min",
321:       "slate_unique_tag_cnt",
322:       "slate_unique_author_cnt",
323:       "slate_has_rand"
324:     ]
325:   },
326:   "existing_v5_budget_formula": {
327:     "elapsed_min": "consumed, minutes, delta=1 minute",
328:     "req_idx": "not consumed; dual-clock formula pending"
329:   },
330:   "model_status": "classification complete; historical v5 routing unchanged; no new full training",
331:   "log1p_fields": [
332:     "user_hist_session_min",
333:     "user_hist_req_cnt",
334:     "day_prev_sess_cnt",
335:     "day_prev_elapsed_min"
336:   ],
337:   "splits": {
338:     "train": {
339:       "rows": 1721745,
340:       "request_order_sha256": "bf59bf8492385cb4358c89ba06781528f18f81541d89b04742facaf13f01b51a",
341:       "feature_count": 54,
342:       "null_features": 0,
343:       "native_batch_rows": 8
344:     },
345:     "valid": {
346:       "rows": 57872,
347:       "request_order_sha256": "5bfb9d2af0f10bbaba102934b1e969d3e6cd203ea775fc33a191f49877eb4917",
348:       "feature_count": 54,
349:       "null_features": 0,
350:       "native_batch_rows": 8
351:     },
352:     "test": {
353:       "rows": 48898,
354:       "request_order_sha256": "46ac632ce48e70f2ddb933866fe135b0cdf8d53f1178cc61ebf88e52329aeffc",
355:       "feature_count": 54,
356:       "null_features": 0,
357:       "native_batch_rows": 8
358:     }
359:   },
360:   "budget_parameter_fields_constant_within_session": {
361:     "user_id": 1,
362:     "user_active_degree": 1,
363:     "is_live_streamer": 1,
364:     "is_video_author": 1,
365:     "friend_user_num_range": 1,
366:     "register_days_range": 1,
367:     "follow_user_num_range": 1,
368:     "fans_user_num_range": 1,
369:     "onehot_feat0": 1,
370:     "onehot_feat1": 1,
371:     "onehot_feat2": 1,
372:     "onehot_feat3": 1,
373:     "onehot_feat4": 1,
374:     "onehot_feat5": 1,
375:     "onehot_feat6": 1,
376:     "onehot_feat7": 1,
377:     "onehot_feat8": 1,
378:     "onehot_feat9": 1,
379:     "onehot_feat10": 1,
380:     "onehot_feat11": 1,
381:     "onehot_feat12": 1,
382:     "onehot_feat13": 1,
383:     "onehot_feat14": 1,
384:     "onehot_feat15": 1,
385:     "onehot_feat16": 1,
386:     "onehot_feat17": 1,
387:     "tab": 1,
388:     "day_of_week": 1,
389:     "is_weekend": 1,
390:     "start_hour": 1,
391:     "user_hist_session_min": 1,
392:     "user_hist_req_cnt": 1,
393:     "day_prev_sess_cnt": 1,
394:     "day_prev_elapsed_min": 1
395:   }
396: }
````

## File: workspace/kr_time_budget/特征分类.md
````markdown
 1: # 当前特征分类：30共享＋6预算与时钟＋18内容
 2: 
 3: 已按本次设计完成分类，定义集中在[kr_request_schema.py](pipeline/kr_request_schema.py)。全量审计确认现有data/v3可直接复用，无需重建parquet、重拟合词表或改变列顺序。机器可读字段清单与验证结果：[feature_categories.json](feature_categories.json)。
 4: 
 5: ## 类别与目标路由
 6: 
 7: | 类别 | 数量 | 字段 | 预算参数网 | 内容塔 |
 8: | --- | ---: | --- | --- | --- |
 9: | 共享用户 | 26 | user_id、user_active_degree、is_live_streamer、is_video_author、friend_user_num_range、register_days_range、follow_user_num_range、fans_user_num_range、onehot_feat0…17 | 是 | 是 |
10: | 共享开场上下文 | 4 | tab、day_of_week、is_weekend、start_hour | 是 | 是 |
11: | 预算先验 | 4 | user_hist_session_min、user_hist_req_cnt、day_prev_sess_cnt、day_prev_elapsed_min | 是 | 否 |
12: | 会话时钟 | 2 | elapsed_min、req_idx | 否；供预算公式 | 否 |
13: | 前序内容反馈 | 9 | is_first_req、prev1_click_cnt、prev1_longview_cnt、prev1_pos_cnt、sess_cum_click_rate、sess_cum_longview_rate、sess_cum_pos_rate、sess_cum_hate_cnt、hist_tags | 否 | 是 |
14: | 当前列表 | 9 | slate_tags、slate_video_types、slate_music_types、slate_prior_ctr_mean、slate_prior_longview_mean、slate_dur_mean_min、slate_unique_tag_cnt、slate_unique_author_cnt、slate_has_rand | 否 | 是 |
15: 
16: 目标预算参数网：34字段；目标内容塔：48字段；Fusion：全部54字段。两塔交集恰好为30个共享背景；三大业务类别无交集并完整覆盖54字段。B3九字段按本次要求完整保留，没有额外屏蔽is_first_req或历史标签。
17: 
18: ## 可直接使用的代码常量
19: 
20: - `SHARED_FIELDS`：30；细分为`USER_PROFILE_FIELDS`与`SESSION_CONTEXT_FIELDS`。
21: - `BUDGET_EXCLUSIVE_FIELDS`：6；细分为`BUDGET_PRIOR_FIELDS`与`SESSION_CLOCK_FIELDS`。
22: - `CONTENT_EXCLUSIVE_FIELDS`：18；细分为`CONTENT_HISTORY_FIELDS`与`CONTENT_SLATE_FIELDS`。
23: - `BUDGET_PARAMETER_FIELDS`：34，预算参数网的明确输入清单。
24: - `BUDGET_SCOPE_FIELDS`：目标预算归属36，含两个时钟。
25: - `CONTENT_TOWER_FIELDS`：48；`CONTENT_MASK_FIELDS`为6个预算与时钟字段。
26: - `FEATURES`：全部54字段，Fusion直接使用，不再另建同义别名。
27: 
28: 已移除B1/B2/B3/B4和含义模糊的BUDGET_FIELDS/CONTENT_FIELDS别名，调用方统一使用明确的业务名称。历史Content-only基线使用`CONTENT_EXCLUSIVE_FIELDS`（18），新设计内容塔使用`CONTENT_TOWER_FIELDS`（48）；二者不混同。`FEATURES`仍保持已有数据的列顺序，`LOG1P_FIELDS`表示上游对数变换字段，`EXTRA_COLUMNS`表示Dataset附加读取列。FeatureSource仍是Rosen编码元信息，不用它替代业务路由。
29: 
30: ## 数据无需重建的依据
31: 
32: 本次仅修改业务分类，没有修改字段值、标签、切分、log1p、sum/mean或padding。data/v3已保存全部54字段，无需为不同塔各存一份数据。
33: 
34: 全量核对：train≤20220506，1,721,745条；valid=20220507，57,872条；test=20220508，48,898条。每份54字段无null；request_id顺序哈希与原manifest一致；全部34预算参数字段在同一Session内的唯一值数量都是1。每个split另用原生ParquetStreamDataset读取8条batch成功。
35: 
36: 继续复用train-fit Schema、原train物理shuffle和valid/test顺序。elapsed_min仍为原始分钟、req_idx仍为原始屏数，没有将物理时间改成对数。原manifest中的content_elapsed是v3当时的模型协议记录，当前分类以本文件与feature_categories.json为准。
37: 
38: 验证：[全量分类审计日志](logs/feature-categories.log)、[30项实验测试](logs/feature-categories-tests.log)。
39: 
40: 命名整理验证：[30项测试](logs/schema-naming-tests.log)、[Schema及预测逐值核对](logs/schema-naming-verification.log)。新建/加载Schema设置、54字段顺序与整理前完全相同；四个v5模型各64条真实test预测与整理前逐值相同。本次只整理定义和引用，没有重建数据或重训。
41: 
42: ## 目标分类与当前模型状态
43: 
44: 本次先完成分类，不重训，也不改变v5/v6已保存实验的实际路由。debug_dataflow.py仍加载v5权重：在forward断点查看`feature_categories`了解类别、`routing`了解v5实际消费、`target_routing`了解本次目标归属。
45: 
46: **req_idx公式尚需定义。** 现有单威布尔公式计算S(e+1分钟)/S(e)，只接收elapsed_min。屏数不是分钟；将req_idx列归为预算时钟，不会让原函数自动使用它。目标“34参数＋2公式＝36”当前是归属契约，尚不能说现有模型已消费36字段。不能简单将屏数加到分钟上，也不能仅扩充右塔掩码后宣称覆盖全部54字段。
47: 
48: 共享U/C符合本次特征设计意图，但特征路由本身不证明预算与内容条件独立或完成因果识别。这里将条件独立作为建模假设记录。
49: 
50: ## 后续双时钟方案评估
51: 
52: 用户提出将独立分钟、屏数威布尔条件存活率相乘：分钟输入e=elapsed_min_raw、delta=1分钟；屏数输入r=req_idx_raw-1、delta=1屏；logS按两项相加。该方案可作为明确的参数化建模假设和消融候选，不把屏数换算成分钟。
53: 
54: 实施时需遵守以下边界：
55: 
56: - 非负且不回退的两个坐标、固定会话参数下，IFR条件存活率的乘积非递增；实际数据已有1,682对分钟时钟回退，不能保证所有相邻屏单调。不能为通过断言静默改成cummax；按两个坐标都非递减的子集验证，并单列回退样本。
57: - 分钟步长固定为1并不等于真实下一屏耗时；这是两个坐标各前进一步的代理预算模型，不能无条件称为已识别的下一屏退出机制。独立假设也未由数据证明。
58: - 第1屏屏数坐标为0，但已有elapsed_min是屏末时钟，第1屏并非必为0。保留原来的屏末预测时点，不混同屏初曝光时点。
59: - 第一轮只把原budget_net末层从Linear(64,2)改成Linear(64,4)，不同时增加64→32隐藏层。否则预算网络容量变化与双时钟机制变化会混杂。
60: - 不提前对gamma和S做1e-6截断；logS保留原值，损失处沿用数值保护。提前截断会改变公式并产生梯度为0的区间。
61: - 引入数值缩放与独立elapsed_min_raw列后才需要派生新数据版本；分类本身仍复用v3。req_idx_raw已有，只新增raw列不计入54个模型特征。先对相同新编码重跑Fusion及单时钟基准，再比较双时钟。
62: 
63: 当前只记录该方案与实施边界，尚未改动预算公式或启动双时钟训练。
64: 
65: 复现审计：
66: 
67: ```bash
68: uv run python -m workspace.kr_time_budget.pipeline.audit_feature_categories
69: uv run pytest -q workspace/kr_time_budget/tests
70: ```
````

## File: 01_snapshot_manifest.json
````json
  1: {
  2:   "snapshot_utc": "2026-11-16T08:44:47.223292+00:00",
  3:   "files": [
  4:     {
  5:       "path": "workspace/kr_time_budget/pipeline/kr_request_schema.py",
  6:       "bytes": 6335,
  7:       "sha256": "79b191a0a8b7e65a80dd9327907a0b97b5946afc56c0e86cb09c832feb0125d3"
  8:     },
  9:     {
 10:       "path": "workspace/kr_time_budget/pipeline/build_requests_v3.py",
 11:       "bytes": 14541,
 12:       "sha256": "e64eb70a00dd07202c94b774b3f611346fb5d719fb79804ac66afe084c9f46b1"
 13:     },
 14:     {
 15:       "path": "workspace/kr_time_budget/pipeline/build_requests.py",
 16:       "bytes": 10458,
 17:       "sha256": "e603f64c42d0af1c9943e1bf3e057d81aca5161304e11fe6fa863cafb539bb82"
 18:     },
 19:     {
 20:       "path": "workspace/kr_time_budget/pipeline/features.py",
 21:       "bytes": 7515,
 22:       "sha256": "1d9a753554ea43455a1651d52ed031016c1736ba41b45e8c852cd79bc257186a"
 23:     },
 24:     {
 25:       "path": "workspace/kr_time_budget/pipeline/audit_feature_categories.py",
 26:       "bytes": 4298,
 27:       "sha256": "97e27765a1073ad60cba37c66fa532ed18ccb142c37fa9497c53e59512446bfd"
 28:     },
 29:     {
 30:       "path": "workspace/kr_time_budget/feature_categories.json",
 31:       "bytes": 9646,
 32:       "sha256": "f2ddd799f424a27fa7024e7c55bcc34126a68684d67cfd00686a6b96395182dd"
 33:     },
 34:     {
 35:       "path": "workspace/kr_time_budget/特征分类.md",
 36:       "bytes": 6773,
 37:       "sha256": "f6172ae47ddd7d0e9be4423159ce46806982a354e9bedaaf0194946638cdb237"
 38:     },
 39:     {
 40:       "path": "workspace/kr_time_budget/data/v3/manifest.json",
 41:       "bytes": 4391,
 42:       "sha256": "6b25631f161f19d6862c97b95021f3e345d2bba4e4f4fd6f487215c831ed1109"
 43:     },
 44:     {
 45:       "path": "workspace/kr_time_budget/data/v3/data_audit.json",
 46:       "bytes": 4639,
 47:       "sha256": "46d2321694149c544e04a81672ae63f4cb4a14617e29efe6b7ab65e9206dfcb5"
 48:     },
 49:     {
 50:       "path": "workspace/kr_time_budget/data/v3/schema/feature_meta.json",
 51:       "bytes": 23259,
 52:       "sha256": "43b9d3239c1e141c2559ab66b165b8a921ccba8050f913028085d1786d24fa99"
 53:     },
 54:     {
 55:       "path": "workspace/kr_time_budget/pipeline/time_budget_v9.py",
 56:       "bytes": 5411,
 57:       "sha256": "90e8bde20db72ae128f1474b6695c2749505aa592a2e61119031d748d7c74792"
 58:     },
 59:     {
 60:       "path": "workspace/kr_time_budget/pipeline/run_exit_v9_pooled.py",
 61:       "bytes": 12886,
 62:       "sha256": "4b2c1f58c7fc03989efbc6b220b5861dfce0e836336f49d5f3be31bc049e3db7"
 63:     },
 64:     {
 65:       "path": "workspace/kr_time_budget/pipeline/time_budget_v8.py",
 66:       "bytes": 7335,
 67:       "sha256": "3d1a4f3d19d3d6cc3d3e49cced141f82ce989b423b49febcb61eedd753c964a4"
 68:     },
 69:     {
 70:       "path": "workspace/kr_time_budget/pipeline/run_exit_v8_branches.py",
 71:       "bytes": 14706,
 72:       "sha256": "68f4594de590235f3e9a853b7a036d34e1db75d601965f32e3cc4274adee9f13"
 73:     },
 74:     {
 75:       "path": "workspace/kr_time_budget/pipeline/run_exit_v7_joint.py",
 76:       "bytes": 14106,
 77:       "sha256": "45005ddc740882fb7161f69769e6314e522a1c3e44735d037f82f1464767cc91"
 78:     },
 79:     {
 80:       "path": "workspace/kr_time_budget/pipeline/run_exit_v3.py",
 81:       "bytes": 11804,
 82:       "sha256": "2a383ec7f70aefa3306a67619a2487c7bcf558aa338a48004c8d5e9753d5a9bf"
 83:     },
 84:     {
 85:       "path": "workspace/kr_time_budget/pipeline/run_exit.py",
 86:       "bytes": 9626,
 87:       "sha256": "1a575fbce12d74e9b9ef052054ef882350147cfccf191fe1ee87cc2733fbdeba"
 88:     },
 89:     {
 90:       "path": "workspace/kr_time_budget/debug_dataflow.py",
 91:       "bytes": 17201,
 92:       "sha256": "21bd289f86fc1e6a4f2f76747e2b406188c08a8544300150b53f806c4bcd0525"
 93:     },
 94:     {
 95:       "path": "workspace/kr_time_budget/tests/test_feature_categories.py",
 96:       "bytes": 1065,
 97:       "sha256": "546f0c0a2203df92d0356f243f82536177070087c6cd8b17f7884f188764fcc7"
 98:     },
 99:     {
100:       "path": "workspace/kr_time_budget/tests/test_v3.py",
101:       "bytes": 5051,
102:       "sha256": "f83e459905933ce6b4a045acc6e13a1ea2feedb74dbdc8c2e64a2a13c68769fc"
103:     },
104:     {
105:       "path": "workspace/kr_time_budget/tests/test_v8.py",
106:       "bytes": 5281,
107:       "sha256": "003bfbf2062d7eb85888a6934ffcbafd504e0ef65fe2d6dda8002f447c3013ef"
108:     },
109:     {
110:       "path": "workspace/kr_time_budget/tests/test_v9.py",
111:       "bytes": 3897,
112:       "sha256": "cc04da5943f29ea3d9919d2fafcfe1a4f1f59ad683b9ac0375f5d93855d31f03"
113:     },
114:     {
115:       "path": "src/rosen/datasets/kuairand.py",
116:       "bytes": 2614,
117:       "sha256": "1e04354b2b2c825a3db93e9c1a141dc04993cb33abd4f525339c748afd89d98b"
118:     },
119:     {
120:       "path": "src/rosen/datasets/base.py",
121:       "bytes": 735,
122:       "sha256": "68a68ddbc1158fda912f57c121aacc8584a9c384a20b6ae2aa48c3c1038dd66f"
123:     },
124:     {
125:       "path": "src/rosen/emb/manager.py",
126:       "bytes": 26714,
127:       "sha256": "d4750707b783677c297ccbd7c3058fce18b25f7b713a2796f7b61f1ab18013ce"
128:     },
129:     {
130:       "path": "src/rosen/emb/emblayer.py",
131:       "bytes": 27165,
132:       "sha256": "9a4fd8fbe2d4f862a0d8828693344f50be8a81cfe518b608c63a36ecd9c7c550"
133:     },
134:     {
135:       "path": "src/rosen/emb/schema/__init__.py",
136:       "bytes": 779,
137:       "sha256": "bb9e60dc4d4ef348fc7d2a1ae43018129ad5f1aee5f5655539728d4351bc5a70"
138:     },
139:     {
140:       "path": "src/rosen/emb/schema/base.py",
141:       "bytes": 5579,
142:       "sha256": "30a3774f2a207b97e0333186506359669bb470abfc201e487c7802b4e6c77857"
143:     },
144:     {
145:       "path": "src/rosen/emb/schema/categorical.py",
146:       "bytes": 18570,
147:       "sha256": "a90a114cbdb85ed71fee45db530d0d55e5e005df80de1bda1af380686dfec0da"
148:     },
149:     {
150:       "path": "src/rosen/emb/schema/numerical.py",
151:       "bytes": 8924,
152:       "sha256": "6ee265f22cc0a8b6e7915d01936c16506cafc6ba4f893a6ae63210eb2b38a922"
153:     },
154:     {
155:       "path": "src/rosen/emb/schema/utils.py",
156:       "bytes": 2914,
157:       "sha256": "3857a67eff14c94d62d400d60ce318a17c1898089977e6d095da38f2b9a31e4c"
158:     },
159:     {
160:       "path": "src/rosen/data/dataset.py",
161:       "bytes": 29917,
162:       "sha256": "045a968ce4041079f056046e6f6a6558688aad46e182ad9866f87110bf9640bb"
163:     },
164:     {
165:       "path": "src/rosen/data/padding.py",
166:       "bytes": 9206,
167:       "sha256": "4de3ac8a386fe1178d86345e2a5ab16dbb734d90853657a56e4519ee69521428"
168:     },
169:     {
170:       "path": "src/rosen/core/enum_type.py",
171:       "bytes": 2493,
172:       "sha256": "b77c4dedcd724b67acf368a227c4194111cb2b18d1eaf2dc899a2035bfd1fcc8"
173:     },
174:     {
175:       "path": "src/rosen/core/interaction.py",
176:       "bytes": 14502,
177:       "sha256": "26d4e163edb5af0890576240464dce12567b910f9f85d1608b13a9073d00d263"
178:     },
179:     {
180:       "path": "src/rosen/models/base.py",
181:       "bytes": 1551,
182:       "sha256": "a8385f5ba7dedfd7a732f5bba6668859765b92ceb89092d47e7ed16ce0c4a8c6"
183:     },
184:     {
185:       "path": "src/rosen/models/msr/base.py",
186:       "bytes": 1402,
187:       "sha256": "1594afdcf431f2f704985a2fd4be45f85e6ec8881d1095ea76d9fb2f731122d5"
188:     },
189:     {
190:       "path": "src/rosen/core/trainers/base.py",
191:       "bytes": 10833,
192:       "sha256": "3c7afe6d1b75c811e9349e0d9094d42d25a079d4e6c9c5789d3e1a50700b6cae"
193:     },
194:     {
195:       "path": "src/rosen/utils/optimize.py",
196:       "bytes": 4340,
197:       "sha256": "9d7f657b48ae86c5edd32e79f6de1bdf4fe34337fce6c54a64dfc7cf78f179f5"
198:     },
199:     {
200:       "path": "src/rosen/utils/shuffle.py",
201:       "bytes": 15194,
202:       "sha256": "efeca61c081bc8ac7b0d7484893fe48316995751418537282251c3c7eeb130c4"
203:     },
204:     {
205:       "path": "src/rosen/utils/sort.py",
206:       "bytes": 4754,
207:       "sha256": "67a4b465d027f450f5facbf5faa788b57bb0e94f97869526db2cc9c7e2665cec"
208:     }
209:   ]
210: }
````

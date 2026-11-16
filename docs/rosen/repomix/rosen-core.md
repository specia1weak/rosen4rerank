This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where content has been formatted for parsing in markdown style.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
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
- Only files matching these patterns are included: docs/rosen/repomix/README.md, docs/rosen/repomix/repomix.config.json, pyproject.toml, src/rosen/__init__.py, src/rosen/core/enum_type.py, src/rosen/core/interaction.py, src/rosen/emb/*.py, src/rosen/emb/schema/*.py, src/rosen/data/*.py, src/rosen/core/train/*.py, src/rosen/core/trainers/*.py, src/rosen/evaluate/*.py, src/rosen/experiment/__init__.py, src/rosen/experiment/config.py, src/rosen/experiment/param.py, src/rosen/models/base.py, src/rosen/models/msr/__init__.py, src/rosen/models/msr/base.py, src/rosen/models/msr/sharedbottom.py, src/rosen/models/msr/components/__init__.py, src/rosen/models/msr/components/heads.py, src/rosen/models/msr/components/codebook.py, src/rosen/models/utils/activation.py, src/rosen/models/utils/common.py, src/rosen/models/utils/container.py, src/rosen/models/utils/general.py, src/rosen/utils/observatory/port.py, src/rosen/utils/shuffle.py, src/rosen/utils/sort.py, src/rosen/utils/sequential.py, src/rosen/utils/optimize.py
- Files matching these patterns are excluded: **/__pycache__/**, **/*.pyc, **/.DS_Store, docs/rosen/repomix/rosen-core.md
- Files matching default ignore patterns are excluded
- Content has been formatted for parsing in markdown style

# User Provided Header
rosen 核心数据流源码阅读包。先读 docs/rosen/repomix/README.md。入选源码全文保留，不压缩、不删除注释或空行；文件首尾空白可能由 Repomix 规范化。仅解释 rosen 当前实现，不包含 HAC 源码或移植方案。

# Directory Structure
````
docs/
  rosen/
    repomix/
      README.md
      repomix.config.json
src/
  rosen/
    core/
      train/
        __init__.py
        context.py
        early_stopper.py
        ema.py
        hooks.py
        loss.py
        two_stage.py
      trainers/
        __init__.py
        base.py
        ema.py
        two_stage.py
        with_life.py
      enum_type.py
      interaction.py
    data/
      __init__.py
      dataset.py
      padding.py
      profile.py
    emb/
      schema/
        __init__.py
        base.py
        categorical.py
        numerical.py
        sequence.py
        utils.py
      __init__.py
      emblayer.py
      manager.py
      split.py
    evaluate/
      __init__.py
      evaluator.py
      manager.py
      metrics.py
    experiment/
      __init__.py
      config.py
      param.py
    models/
      msr/
        components/
          __init__.py
          codebook.py
          heads.py
        __init__.py
        base.py
        sharedbottom.py
      utils/
        activation.py
        common.py
        container.py
        general.py
      base.py
    utils/
      observatory/
        port.py
      optimize.py
      sequential.py
      shuffle.py
      sort.py
    __init__.py
pyproject.toml
````

# Files

## File: docs/rosen/repomix/README.md
````markdown
# rosen 核心数据流源码包

导出文件：[`rosen-core.md`](rosen-core.md)。只收录 rosen 核心链路及其相关实现；入选文件完整保留，不删除注释、空行或函数体，不做代码压缩。源码中的历史注释、示例和现有行为也原样保留。这个文件是阅读包，不是可单独运行的源码发行包。

## 数据怎么走

```text
实验脚本读取、join、过滤原始表（Polars LazyFrame）
  -> 显式切分 train / valid / test，或 SchemaManager.split_dataset()
  -> SchemaManager.fit(train_raw)：拟合词表、数值统计量，保存 schema
  -> SchemaManager.transform(split)：按 schema 编码各 split
  -> SchemaManager.save_as_dataset()：写入 parquet
  -> ParquetStreamDataset：扫描、worker 分片、可选 shuffle、组 batch
  -> TensorFormatter：按字段 formatter 转 tensor、截断和 padding
  -> Interaction：字段名 -> batch tensor
  -> OmniEmbLayer：按字段/source/序列组取 embedding
  -> 模型 calculate_loss() / predict()
  -> Trainer：更新参数、评估、早停、checkpoint
```

原始数据 join、过滤、切分规则和特征选择由实验脚本明确决定。`EmbSetting` 子类定义每个字段的 fit、transform、tensor formatter 和 embedding 取值方式；`SchemaManager` 负责组织这些规则。

`fit()` 发现 `schema_dir/feature_meta.json` 已存在时直接加载。schema 保存各 setting 状态，词表另存到相对路径 `vocab/*.parquet`。`fields()` 汇总 setting 输出列，以及时间、label、domain 字段；dataset 还支持通过 `extra_col_names` / `extra_col_formatters` 保留额外列。

已编码 parquet 走 `ParquetStreamDataset`；原始 parquet 需要边读边编码时走 `RawParquetStreamDataset`。两者已经产出 batch，外层 `DataLoader` 使用 `batch_size=None`。验证、测试通常设置 `shuffle=False, drop_last=False`，避免丢失尾批。

profile 路径由 `ProfileDatasetBundle` 读取 manifest、schema 和 split。`ProfileParquetStreamDataset` 通过 `ProfileSequenceStore` 按 owner 和 cursor 取历史前缀，再按时间规则过滤、按 max_len 截断，最后格式化为 `Interaction`。这条路径复用侧表中的历史，避免每行重复存储整段历史。

## batch 之后怎么走

`BaseModel` 持有 manager、`OmniEmbLayer`、`LossPort` 和 `ObservationPort`。`OmniEmbLayer.whole` 汇合普通用户/物品/交互特征；序列通过 `seq_groups` 的视图单独读取。`SharedBottomModel` 是包内附带的具体例子：embedding -> backbone -> 按 domain 选择 tower -> logits，训练使用 BCEWithLogits。

`BaseTrainer` 的默认 step 是 `zero_grad -> calculate_loss -> lsp.settle -> backward -> clip_grad_norm -> optimizer.step`。模型实现 `CustomTrainStepProtocol.custom_train_step(batch, ctx)` 时接管该 step 的优化过程，并负责结算 loss、返回标量 Tensor。epoch/eval 周期方法由 `BaseTrainerWithLife` 解释；EMA 和两阶段运行也有独立实现。

默认评估从 batch 取 manager 的 uid 和首个 label，调用 `model.predict()`，以 `batch_preds_1d` 交给 `EvaluatorManager`；它按全量/domain 等过滤器分发到 evaluator，最后汇总、清空。`predict()` 的分数语义由模型决定。已有 Top-K evaluator 接收二维物品得分和目标物品索引，但默认 Trainer 不会自动调用 `full_sort_predict()` 或切换为 Top-K 收集。

checkpoint 保存模型、主 optimizer、epoch、global_step，以及模型通过 `TrainerStateProtocol` 提供的额外训练状态。`run()` 根据验证指标执行早停和 best 保存；不会自动重载 best 后测试，需要入口显式安排。

## 源码阅读顺序

下列路径均为仓库相对路径，可在导出文件中搜索对应的 `## File:`：

1. `src/rosen/emb/manager.py`、`emb/schema/*.py`、`emb/split.py`：字段规则、编码、切分、持久化。
2. `src/rosen/data/dataset.py`、`data/padding.py`、`data/profile.py`、`core/interaction.py`：parquet 到 batch，以及 profile 历史装配。
3. `src/rosen/emb/emblayer.py`、`models/base.py`、`models/msr/sharedbottom.py`：batch 到 embedding 和模型输出。
4. `src/rosen/core/trainers/base.py`、`core/train/{context,hooks,loss}.py`：优化步骤和扩展契约。
5. `src/rosen/evaluate/{manager,evaluator,metrics}.py`：分组汇总与指标口径。
6. `src/rosen/experiment/{config,param}.py`：工作目录、配置和 CLI 参数。

模型只附 SharedBottom 实现及其直接组件、MSR 基类和注册入口，没有打包整个模型库；观测只附 BaseModel 使用的 port。其他模型、观测历史存储、实验脚本和数据不在本包内。`src/rosen/models/rerank/` 在生成时为空，因此没有该目录的源码条目。

## 重新生成

在仓库根目录运行（本次使用 Repomix 1.18.1）：

```bash
repomix --config docs/rosen/repomix/repomix.config.json
```

选取范围以 [`repomix.config.json`](repomix.config.json) 为准。Repomix 使用 Markdown 代码围栏包装完整文件，可能规范化文件首尾空白；文件内部的注释、空行和代码保持原样。
````

## File: docs/rosen/repomix/repomix.config.json
````json
{
  "output": {
    "filePath": "docs/rosen/repomix/rosen-core.md",
    "style": "markdown",
    "parsableStyle": true,
    "headerText": "rosen 核心数据流源码阅读包。先读 docs/rosen/repomix/README.md。入选源码全文保留，不压缩、不删除注释或空行；文件首尾空白可能由 Repomix 规范化。仅解释 rosen 当前实现，不包含 HAC 源码或移植方案。",
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "git": {
      "sortByChanges": false
    }
  },
  "include": [
    "docs/rosen/repomix/README.md",
    "docs/rosen/repomix/repomix.config.json",
    "pyproject.toml",
    "src/rosen/__init__.py",
    "src/rosen/core/enum_type.py",
    "src/rosen/core/interaction.py",
    "src/rosen/emb/*.py",
    "src/rosen/emb/schema/*.py",
    "src/rosen/data/*.py",
    "src/rosen/core/train/*.py",
    "src/rosen/core/trainers/*.py",
    "src/rosen/evaluate/*.py",
    "src/rosen/experiment/__init__.py",
    "src/rosen/experiment/config.py",
    "src/rosen/experiment/param.py",
    "src/rosen/models/base.py",
    "src/rosen/models/msr/__init__.py",
    "src/rosen/models/msr/base.py",
    "src/rosen/models/msr/sharedbottom.py",
    "src/rosen/models/msr/components/__init__.py",
    "src/rosen/models/msr/components/heads.py",
    "src/rosen/models/msr/components/codebook.py",
    "src/rosen/models/utils/activation.py",
    "src/rosen/models/utils/common.py",
    "src/rosen/models/utils/container.py",
    "src/rosen/models/utils/general.py",
    "src/rosen/utils/observatory/port.py",
    "src/rosen/utils/shuffle.py",
    "src/rosen/utils/sort.py",
    "src/rosen/utils/sequential.py",
    "src/rosen/utils/optimize.py"
  ],
  "ignore": {
    "useGitignore": false,
    "useDotIgnore": false,
    "useDefaultPatterns": true,
    "customPatterns": [
      "**/__pycache__/**",
      "**/*.pyc",
      "**/.DS_Store",
      "docs/rosen/repomix/rosen-core.md"
    ]
  },
  "security": {
    "enableSecurityCheck": true
  }
}
````

## File: src/rosen/core/train/__init__.py
````python
"""Foundational training contracts, state and reusable controllers.

Trainer implementations live in :mod:`rosen.core.trainers`. This package
must not import or re-export that higher-level package.
"""

from rosen.core.train.context import (
    EvaluationFeedback,
    TrainContext,
    TrainerComponents,
    TrainerDataLoaders,
)
from rosen.core.train.early_stopper import EarlyStopper
from rosen.core.train.ema import EMAConfig, EMAController, ema_decay_tag
from rosen.core.train.hooks import (
    CustomTrainStepProtocol,
    ObservationSourceProtocol,
    TrainingCycleProtocol,
    TrainingPhaseProtocol,
    TrainerStateProtocol,
)
from rosen.core.train.loss import LossPort, LossTerm
from rosen.core.train.two_stage import (
    StageReport,
    StageSnapshot,
    TwoStagePhase,
    TwoStageTrainingProtocol,
)

__all__ = [
    "CustomTrainStepProtocol",
    "EMAConfig",
    "EMAController",
    "EarlyStopper",
    "EvaluationFeedback",
    "LossPort",
    "LossTerm",
    "ObservationSourceProtocol",
    "StageReport",
    "StageSnapshot",
    "TwoStagePhase",
    "TrainContext",
    "TrainingCycleProtocol",
    "TrainingPhaseProtocol",
    "TrainerComponents",
    "TrainerDataLoaders",
    "TrainerStateProtocol",
    "TwoStageTrainingProtocol",
    "ema_decay_tag",
]
````

## File: src/rosen/core/train/context.py
````python
from dataclasses import dataclass
from typing import Any, Optional, Iterable, Mapping
import torch

from rosen.core.interaction import Interaction
from rosen.core.train.early_stopper import EarlyStopper


@dataclass(frozen=True)
class EvaluationFeedback:
    """Typed evaluation targets available to model training-cycle methods."""

    metrics_by_target: Mapping[str, Optional[dict]]
    primary_target: str
    primary_metrics: Optional[dict]


@dataclass
class TrainContext:
    # 1. 训练进度统计（通常是只读的）
    epoch: int
    global_step: int
    batch_idx: int

    # 2. 核心操纵杆
    optimizer: torch.optim.Optimizer

    # 3. 实验与全局组件
    manager: Any  # 你的 SchemaManager
    cfg: Any  # 全局配置参数
    timer: Any  # 允许模型给特殊的内部步骤计时
    recorder: Any = None
    stage: str = "train"
    phase: str = "default"
    evaluation_feedback: Optional[EvaluationFeedback] = None


@dataclass
class TrainerDataLoaders:
    train: Iterable[Interaction]
    valid: Optional[Iterable[Interaction]]
    test: Optional[Iterable[Interaction]] = None  # 扩展性好，支持可选参数


@dataclass
class TrainerComponents:
    evaluator_manager: Any = None

    # Keep core/train independent from the concrete Observatory package.
    recorder: Any = None
    timer: Any = None
    early_stopper: Optional[EarlyStopper] = None
    health_monitor: Any = None

    def __post_init__(self) -> None:
        if self.early_stopper is not None and not isinstance(self.early_stopper, EarlyStopper):
            raise TypeError(
                "TrainerComponents.early_stopper must be an EarlyStopper or None, "
                f"got {type(self.early_stopper).__name__}."
            )
````

## File: src/rosen/core/train/early_stopper.py
````python
"""Metric selection and patience state for early stopping."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopper:
    patience: int = 5
    min_delta: float = 0.0
    preferred_evaluators: tuple[str, ...] = ("overall", "all")
    preferred_metrics: tuple[str, ...] = (
        "auc",
        "gauc",
        "ndcg@10",
        "ndcg@20",
        "ndcg",
        "hr@10",
        "hr@20",
        "hr",
        "recall@10",
        "recall@20",
        "recall",
        "logloss",
        "loss",
    )

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        """Explicitly restart this object's patience window."""
        self.best_metric_value: Optional[float] = None
        self.best_metric_name: Optional[str] = None
        self.best_evaluator_name: Optional[str] = None
        self.best_epoch: Optional[int] = None
        self.bad_epoch_count: int = 0
        self.last_metric_value: Optional[float] = None
        self.last_metric_name: Optional[str] = None
        self.last_evaluator_name: Optional[str] = None
        self.last_status_message: Optional[str] = None

    def _ordered_evaluator_names(self, summary_dict: dict[str, dict[str, float]]) -> list[str]:
        names = list(summary_dict.keys())
        ordered = [name for name in self.preferred_evaluators if name in summary_dict]
        ordered.extend(name for name in names if name not in ordered)
        return ordered

    def _pick_metric(self, summary_dict: dict[str, dict[str, float]]):
        if not summary_dict:
            return None

        for evaluator_name in self._ordered_evaluator_names(summary_dict):
            metrics_dict = summary_dict[evaluator_name]
            if not metrics_dict:
                continue
            lowered = {metric_name.lower(): metric_name for metric_name in metrics_dict}
            for preferred_metric in self.preferred_metrics:
                metric_name = lowered.get(preferred_metric.lower())
                if metric_name is not None:
                    return evaluator_name, metric_name, float(metrics_dict[metric_name])
            metric_name, metric_value = next(iter(metrics_dict.items()))
            return evaluator_name, metric_name, float(metric_value)
        return None

    @staticmethod
    def _metric_mode(metric_name: str) -> str:
        metric_name = metric_name.lower()
        if metric_name in {"loss", "logloss", "mae", "mse", "rmse"}:
            return "min"
        return "max"

    @staticmethod
    def _format_metric_value(metric_value: Optional[float]) -> str:
        if metric_value is None:
            return "None"
        return f"{metric_value:.6f}"

    def step(
        self,
        summary_dict: Optional[dict[str, dict[str, float]]],
        epoch: Optional[int] = None,
    ) -> tuple[bool, bool]:
        if summary_dict is None:
            return False, False

        early_metric = self._pick_metric(summary_dict)
        if early_metric is None:
            return False, False

        evaluator_name, metric_name, metric_value = early_metric
        mode = self._metric_mode(metric_name)
        self.last_evaluator_name = evaluator_name
        self.last_metric_name = metric_name
        self.last_metric_value = metric_value

        if self.best_metric_value is None:
            improved = True
        elif mode == "max":
            improved = metric_value > self.best_metric_value + self.min_delta
        else:
            improved = metric_value < self.best_metric_value - self.min_delta

        if improved:
            self.best_metric_value = metric_value
            self.best_metric_name = metric_name
            self.best_evaluator_name = evaluator_name
            self.best_epoch = epoch
            self.bad_epoch_count = 0
            self.last_status_message = (
                f"[EarlyStop] best {evaluator_name}/{metric_name}="
                f"{self._format_metric_value(metric_value)} at epoch {epoch}, patience 0/{self.patience}"
            )
            return True, False

        self.bad_epoch_count += 1
        self.last_status_message = (
            f"[EarlyStop] wait {self.bad_epoch_count}/{self.patience}, "
            f"current {evaluator_name}/{metric_name}={self._format_metric_value(metric_value)}, "
            f"best {self.best_evaluator_name}/{self.best_metric_name}="
            f"{self._format_metric_value(self.best_metric_value)} at epoch {self.best_epoch}"
        )
        return False, self.bad_epoch_count >= self.patience

    def status_text(self) -> Optional[str]:
        return self.last_status_message
````

## File: src/rosen/core/train/ema.py
````python
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Union

import torch


def ema_decay_tag(decay: float) -> str:
    return f"{float(decay):.6g}".replace(".", "p")


def _parse_ema_decays(value: Union[None, float, int, str, Iterable[float]]) -> tuple[float, ...]:
    """Normalize the supported EMA decay configuration forms."""
    if value is None:
        return ()
    if isinstance(value, (float, int)):
        return (float(value),)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ()
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = stripped.split(",")
        return _parse_ema_decays(parsed)
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class EMAConfig:
    decays: Sequence[float] | float | str | None = field(default_factory=lambda: (0.995,))
    start_step: int = 0
    update_every: int = 1
    include_buffers: bool = True
    eval_student: bool = True
    eval_ema: bool = True
    primary_checkpoint: str = ""
    backend: str = "teacher"


class EMAController:
    """Exponential moving averages over an arbitrary ``nn.Module``."""

    def __init__(
            self,
            model: torch.nn.Module,
            *,
            decays: Sequence[float],
            start_step: int = 0,
            update_every: int = 1,
            include_buffers: bool = True,
            backend: str = "teacher",
    ):
        self.decays = tuple(float(decay) for decay in decays)
        if not self.decays:
            raise ValueError("EMAController requires at least one decay.")
        for decay in self.decays:
            if not 0.0 <= decay < 1.0:
                raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

        self.start_step = max(0, int(start_step))
        self.update_every = max(1, int(update_every))
        self.include_buffers = bool(include_buffers)
        self.backend = str(backend).lower()
        if self.backend != "teacher":
            raise ValueError("EMAController currently supports backend='teacher'.")

        self.teachers: Dict[float, torch.nn.Module] = {}
        for decay in self.decays:
            teacher = copy.deepcopy(model)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            self.teachers[float(decay)] = teacher

    @torch.no_grad()
    def update(
            self,
            model: torch.nn.Module,
            *,
            step: int,
            state_names: Iterable[str] | None = None,
    ) -> bool:
        step = int(step)
        if step < self.start_step:
            return False
        if (step - self.start_step) % self.update_every != 0:
            return False

        source_state = model.state_dict()
        parameter_names = set(dict(model.named_parameters()).keys())
        selected_names = None if state_names is None else frozenset(str(name) for name in state_names)
        if selected_names is not None:
            unknown = selected_names.difference(source_state)
            if unknown:
                raise KeyError(f"EMA update received unknown state names: {sorted(unknown)}")
        for decay, teacher in self.teachers.items():
            target_state = teacher.state_dict()
            for name, source_value in source_state.items():
                if selected_names is not None and name not in selected_names:
                    continue
                target_value = target_state[name]
                if torch.is_floating_point(target_value):
                    if self.include_buffers or name in parameter_names:
                        target_value.mul_(decay).add_(source_value.detach(), alpha=1.0 - decay)
                    else:
                        target_value.copy_(source_value.detach())
                else:
                    target_value.copy_(source_value.detach())
            teacher.eval()
        return True

    def checkpoint_items(self) -> tuple[tuple[str, float, torch.nn.Module], ...]:
        return tuple(
            (f"ema{ema_decay_tag(decay)}", float(decay), self.teachers[decay])
            for decay in self.decays
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "decays": list(self.decays),
            "start_step": self.start_step,
            "update_every": self.update_every,
            "include_buffers": self.include_buffers,
            "backend": self.backend,
            "teachers": {
                f"ema{ema_decay_tag(decay)}": teacher.state_dict()
                for decay, teacher in self.teachers.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        teachers = state.get("teachers", {})
        if not isinstance(teachers, Mapping):
            raise TypeError("EMA state must contain a mapping under 'teachers'.")
        for decay, teacher in self.teachers.items():
            tag = f"ema{ema_decay_tag(decay)}"
            if tag in teachers:
                teacher.load_state_dict(teachers[tag])
                continue
            legacy_tag = ema_decay_tag(decay)
            if legacy_tag in teachers:
                teacher.load_state_dict(teachers[legacy_tag])


__all__ = [
    "EMAConfig",
    "EMAController",
    "ema_decay_tag",
]
````

## File: src/rosen/core/train/hooks.py
````python
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from rosen.core.train.context import TrainContext


@runtime_checkable
class CustomTrainStepProtocol(Protocol):
    def custom_train_step(self, batch_interaction, ctx: TrainContext):
        ...


@runtime_checkable
class TrainingCycleProtocol(Protocol):
    """Model behavior at the fixed epoch/evaluation boundaries."""

    def on_train_epoch_start(self, ctx: TrainContext) -> Optional[tuple[str, ...]]:
        ...

    def on_train_epoch_end(self, ctx: TrainContext) -> Optional[tuple[str, ...]]:
        ...

    def on_eval_epoch_end(
            self,
            metrics: Optional[dict[str, Any]],
            ctx: TrainContext,
    ) -> Optional[tuple[str, ...]]:
        ...


@runtime_checkable
class TrainerStateProtocol(Protocol):
    """Checkpoint state owned by model-side training behavior."""

    def trainer_state_dict(self) -> Mapping[str, Any]:
        ...

    def load_trainer_state_dict(self, state: Mapping[str, Any]) -> None:
        ...


@runtime_checkable
class TrainingPhaseProtocol(Protocol):
    """Model exposes a named phase only when its training is phase-aware."""

    @property
    def training_phase(self) -> str:
        ...


@runtime_checkable
class ObservationSourceProtocol(Protocol):
    def should_collect_observations(self, ctx: TrainContext, stage: str) -> bool:
        ...

    def collect_observations(self, ctx: TrainContext, stage: str) -> None:
        ...
````

## File: src/rosen/core/train/loss.py
````python
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import torch


LossWeight = float | int | torch.Tensor


@dataclass(frozen=True)
class LossTerm:
    """One named contribution to the active optimization objective."""

    value: torch.Tensor
    weight: LossWeight = 1.0

    @property
    def weighted(self) -> torch.Tensor:
        return self.value * self.weight


class LossPort:
    """Model-owned loss ledger and current-step summary port.

    ``LossPort`` never owns history. The code that owns a backward transaction
    owns a default ``main`` scope, model code may add named terms, and
    ``settle()`` returns the graph-bearing scalar Tensor. Additional named
    scopes are only needed for independent optimization objectives. A detached
    summary remains available until the next ``start_step()`` or an external
    caller clears it.
    """

    def __init__(self) -> None:
        self._scope = ""
        self._terms: OrderedDict[str, LossTerm] = OrderedDict()
        self._stack: list[tuple[str, OrderedDict[str, LossTerm]]] = []
        self._step_context: dict[str, Any] = {}
        self._settlements: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @property
    def active(self) -> bool:
        return bool(self._scope)

    @property
    def scope_name(self) -> str:
        return self._scope or "main"

    @property
    def has_terms(self) -> bool:
        return bool(self._terms)

    @property
    def has_settlements(self) -> bool:
        return bool(self._settlements)

    def start_step(
            self,
            *,
            step: int,
            epoch: int,
            batch_idx: int,
            stage: str,
    ) -> None:
        """Start one logical model step and discard the previous step summary."""

        if self.active:
            raise RuntimeError("LossPort.start_step() cannot run inside an active scope.")
        self._step_context = {
            "step": int(step),
            "epoch": int(epoch),
            "batch_idx": int(batch_idx),
            "stage": str(stage),
        }
        self.clear_summary()

    @contextmanager
    def scope(self, name: str = "main") -> Iterator["LossPort"]:
        scope_name = str(name).strip()
        if not scope_name:
            raise ValueError("LossPort scope name cannot be empty.")
        self._stack.append((self._scope, self._terms))
        self._scope = scope_name
        self._terms = OrderedDict()
        completed = False
        try:
            yield self
            completed = True
        finally:
            unsettled_scope = self._scope
            has_unsettled_terms = bool(self._terms)
            self._scope, self._terms = self._stack.pop()
            if completed and has_unsettled_terms:
                raise RuntimeError(
                    f"Loss scope {unsettled_scope!r} exited with unsettled terms; "
                    "the code that owns the optimization transaction must call settle()."
                )

    def add(self, name: str, value: torch.Tensor, *, weight: LossWeight = 1.0) -> torch.Tensor:
        if not self.active:
            raise RuntimeError("LossPort.add() requires an active LossPort.scope().")
        key = str(name)
        if key in self._terms:
            raise ValueError(f"Loss term {key!r} already exists in scope {self._scope!r}.")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Loss term {key!r} must be a Tensor, got {type(value).__name__}.")
        self._terms[key] = LossTerm(value=value, weight=weight)
        return value

    def total(self) -> torch.Tensor:
        if not self.has_terms:
            raise RuntimeError(f"Loss scope {self.scope_name!r} has no terms.")
        return sum(
            (term.weighted for term in self._terms.values()),
            start=next(iter(self._terms.values())).value.new_zeros(()),
        )

    @staticmethod
    def _scalar(value: torch.Tensor | float | int, *, label: str) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{label} must be scalar, got shape {tuple(value.shape)}.")
            return float(value.detach().float().cpu().item())
        return float(value)

    def settle(self, value: torch.Tensor | None = None) -> torch.Tensor:
        """Finalize the active scope, retain a detached summary, and return Tensor."""

        if not self.active:
            raise RuntimeError("LossPort.settle() requires an active LossPort.scope().")
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(
                "Training loss must be a Tensor or None, "
                f"got {type(value).__name__}."
            )
        total = self.total() if value is None else value
        if total.numel() != 1:
            raise ValueError(f"Settled loss must be scalar, got shape {tuple(total.shape)}.")

        scope = self.scope_name
        if scope in self._settlements:
            raise RuntimeError(
                f"Loss scope {scope!r} was settled more than once in the same step; "
                "use distinct scope names for distinct optimization objectives."
            )
        terms = OrderedDict()
        for name, term in self._terms.items():
            raw = self._scalar(term.value, label=f"Loss term {name!r}")
            weight = self._scalar(term.weight, label=f"Loss weight {name!r}")
            terms[name] = {
                "raw": raw,
                "weight": weight,
                "weighted": raw * weight,
            }
        self._settlements[scope] = {
            "total": self._scalar(total, label="Settled loss"),
            "terms": terms,
        }
        self._clear_terms()
        return total

    def _clear_terms(self) -> None:
        self._terms = OrderedDict()

    def summary(self, *, clear: bool = False) -> dict[str, Any]:
        """Return the detached settlements produced by the current model step."""

        result = {
            **self._step_context,
            "settlements": deepcopy(self._settlements),
        }
        if clear:
            self.clear_summary()
        return result

    def clear_summary(self) -> None:
        self._settlements.clear()


__all__ = ["LossPort", "LossTerm", "LossWeight"]
````

## File: src/rosen/core/train/two_stage.py
````python
"""Opt-in contract for two independent training stages.

Usage
-----
Only a stage-one model that genuinely needs two-stage training implements
``TwoStageTrainingProtocol``::

    class SearchModel(BaseModel):
        def should_finish_stage_one(self, report: StageReport) -> bool:
            return report.epoch >= self.search_epochs

        def initialize_stage_two(self, snapshot: StageSnapshot) -> BaseModel:
            model = RetrainModel(...)
            model.load_state_dict(snapshot.model_state)
            model.freeze_search_only_parameters()
            return model

The experiment supplies a factory that creates *fresh* Trainer state for each
model, then delegates orchestration to ``TwoStageRunner``::

    def trainer_factory(model, stage: TwoStagePhase):
        return EMATrainer(
            model=model,
            optimizer=build_optimizer(model),
            manager=manager,
            loaders=build_loaders(stage),
            components=build_fresh_components(),
            cfg=build_fresh_run_config(stage),
            ema_config=ema_config,
        )

    result = TwoStageRunner(trainer_factory).run(search_model)

The same factory is called with ``TwoStagePhase.STAGE_ONE`` and ``STAGE_TWO``;
the enum is for experiment output isolation and is never interpreted by the
model. ``initialize_stage_two`` must return a new model object, and the factory
must create fresh Trainer-owned state on every call. This keeps optimizer
parameter references, EMA teachers, EarlyStopper state, counters, evaluators,
and monitors out of the stage boundary. Ordinary and stage-two-only models do
not implement this protocol. If checkpoints are enabled, the two configs must
use different checkpoint directories so their fresh epoch/step counters
cannot produce the same filename. Reusing loaders is safe only when they are
re-iterable; one-shot iterators must also be rebuilt by the factory.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, TYPE_CHECKING, runtime_checkable

import torch

if TYPE_CHECKING:
    from rosen.models.base import BaseModel


class TwoStagePhase(str, Enum):
    """Typed factory input; models never need to inspect stage-name strings."""

    STAGE_ONE = "stage_one"
    STAGE_TWO = "stage_two"


@dataclass(frozen=True)
class StageReport:
    """Evaluation and progress visible to the stage-one stop condition."""

    epoch: int
    global_step: int
    metrics_by_target: Mapping[str, Optional[dict]]
    primary_target: str
    primary_metrics: Optional[dict]

    def metrics_for(self, target: str) -> Optional[dict]:
        target = str(target)
        if target == "primary":
            return self.primary_metrics
        return self.metrics_by_target.get(target)


@dataclass(frozen=True)
class StageSnapshot:
    """Detached stage-one model state passed to stage-two initialization."""

    model_state: Mapping[str, Any]
    report: StageReport

    @classmethod
    def capture(
            cls,
            model: torch.nn.Module,
            report: StageReport,
    ) -> "StageSnapshot":
        state: dict[str, Any] = {}
        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                state[name] = value.detach().cpu().clone()
            else:
                state[name] = copy.deepcopy(value)
        return cls(model_state=state, report=report)


@runtime_checkable
class TwoStageTrainingProtocol(Protocol):
    """Model-owned termination and stage-two construction rules."""

    def should_finish_stage_one(self, report: StageReport) -> bool:
        ...

    def initialize_stage_two(self, snapshot: StageSnapshot) -> "BaseModel":
        ...


__all__ = [
    "StageReport",
    "StageSnapshot",
    "TwoStagePhase",
    "TwoStageTrainingProtocol",
]
````

## File: src/rosen/core/trainers/__init__.py
````python
"""Trainer implementations built from :mod:`rosen.core.train` components."""

from rosen.core.trainers.base import BaseTrainer
from rosen.core.trainers.ema import EMATrainer
from rosen.core.trainers.two_stage import TwoStageRunResult, TwoStageRunner
from rosen.core.trainers.with_life import BaseTrainerWithLife

__all__ = [
    "BaseTrainer",
    "BaseTrainerWithLife",
    "EMATrainer",
    "TwoStageRunResult",
    "TwoStageRunner",
]
````

## File: src/rosen/core/trainers/base.py
````python
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional, Union

import torch

from rosen.core.interaction import Interaction
from rosen.core.train.context import (
    EvaluationFeedback,
    TrainContext,
    TrainerComponents,
    TrainerDataLoaders,
)
from rosen.core.train.hooks import (
    CustomTrainStepProtocol,
    TrainerStateProtocol,
)
from rosen.emb import SchemaManager
from rosen.experiment.param import ConfigBase
from rosen.models.base import BaseModel


class BaseTrainer:
    """Minimal trainer for ordinary train/eval/checkpoint flow.

    This class deliberately does not interpret model training-cycle methods or
    health monitoring. Those fixed boundaries live in ``BaseTrainerWithLife``;
    experiment audits live under :mod:`rosen.core.trainers.audit`.
    """

    def __init__(
            self,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            manager: SchemaManager,
            loaders: TrainerDataLoaders,
            components: TrainerComponents,
            cfg: ConfigBase,
    ):
        if components.health_monitor is not None:
            raise ValueError(
                "BaseTrainer does not run training-cycle health monitoring; "
                "use BaseTrainerWithLife."
            )
        self.model = model.to(cfg.device)
        self.optimizer = optimizer
        self.manager = manager
        self.train_loader = loaders.train
        self.valid_loader = loaders.valid
        self.test_loader = loaders.test

        self.evaluator = components.evaluator_manager
        self.timer = components.timer
        self.recorder = components.recorder
        self.early_stopper = components.early_stopper
        self.cfg = cfg

        self.epoch = 0
        self.global_step = 0
        self.last_evaluation_feedback: EvaluationFeedback | None = None

    def _timed(self, name: str):
        return self.timer(name) if self.timer is not None else nullcontext()

    def _context(
            self,
            *,
            batch_idx: int,
            stage: str = "train",
            evaluation_feedback: EvaluationFeedback | None = None,
    ) -> TrainContext:
        return TrainContext(
            epoch=self.epoch,
            global_step=self.global_step,
            batch_idx=int(batch_idx),
            optimizer=self.optimizer,
            manager=self.manager,
            cfg=self.cfg,
            timer=self.timer,
            recorder=self.recorder,
            stage=str(stage),
            phase="default",
            evaluation_feedback=evaluation_feedback,
        )

    def predict_step(self, batch_interaction: Interaction):
        return self.model.predict(batch_interaction)

    def _checkpoint_payload(self, metrics: Optional[dict] = None) -> dict:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_trainer_state_dict": (
                self.model.trainer_state_dict()
                if isinstance(self.model, TrainerStateProtocol) else None
            ),
            "metrics": metrics,
            "experiment_name": self.cfg.experiment_name,
            "dataset_name": self.cfg.dataset_name,
        }

    def save_checkpoint(self, tag: str = "", metrics: Optional[dict] = None) -> Union[Path, None]:
        ckpt_dir = getattr(self.cfg, "ckpt_dir", "")
        if not ckpt_dir:
            return None

        save_dir = Path(ckpt_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_tag = f"_{tag}" if tag else ""
        filename = f"epoch_{self.epoch:04d}_step_{self.global_step:08d}{safe_tag}.pt"
        save_path = save_dir / filename

        torch.save(self._checkpoint_payload(metrics), save_path)
        return save_path

    def load_checkpoint(self, path: Union[str, Path], *, map_location=None) -> dict:
        checkpoint = torch.load(path, map_location=map_location or self.cfg.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        model_trainer_state = checkpoint.get("model_trainer_state_dict")
        if isinstance(self.model, TrainerStateProtocol) and model_trainer_state is not None:
            self.model.load_trainer_state_dict(model_trainer_state)
        return checkpoint

    def calculate_train_loss(self, batch, ctx: TrainContext):
        del ctx
        return self.model.calculate_loss(batch)

    def _default_train_step(self, batch, ctx: TrainContext):
        raw_loss = self.calculate_train_loss(batch, ctx)
        loss = self.model.lsp.settle(raw_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        return loss

    def train_step(self, batch, ctx: TrainContext) -> torch.Tensor:
        """Run one train step inside the model's implicit ``main`` Loss scope."""
        with self.model.lsp.scope():
            if not isinstance(self.model, CustomTrainStepProtocol):
                return self._default_train_step(batch, ctx)

            loss = self.model.custom_train_step(batch, ctx)
            if not isinstance(loss, torch.Tensor):
                raise TypeError(
                    "custom_train_step() must settle and return a scalar Tensor, "
                    f"got {type(loss).__name__}."
                )
            if loss.numel() != 1:
                raise ValueError(
                    "custom_train_step() must return a scalar Tensor, "
                    f"got shape {tuple(loss.shape)}."
                )
            if self.model.lsp.has_terms:
                raise RuntimeError(
                    "custom_train_step() returned with unsettled LossPort terms."
                )
            if not self.model.lsp.has_settlements:
                raise RuntimeError(
                    "custom_train_step() must call model.lsp.settle() for each "
                    "optimization objective before returning."
                )
            return loss

    def should_stop_train_epoch(self, ctx: TrainContext, loss) -> bool:
        """Extension point for trainers that intentionally cap an epoch."""
        del loss
        limit = int(getattr(self.cfg, "max_train_steps", 0) or 0)
        return limit > 0 and int(ctx.global_step) + 1 >= limit

    def _has_reached_train_step_limit(self) -> bool:
        limit = int(getattr(self.cfg, "max_train_steps", 0) or 0)
        return limit > 0 and self.global_step >= limit

    def _run_train_batches(self) -> int:
        log_interval = max(1, int(getattr(self.cfg, "train_log_interval", 100)))
        last_batch_idx = -1
        for batch_idx, batch_interaction in enumerate(self.train_loader):
            batch_interaction = batch_interaction.to(self.cfg.device)
            self.optimizer.zero_grad(set_to_none=True)

            ctx = self._context(batch_idx=batch_idx, stage="train")
            last_batch_idx = batch_idx
            current_step = self.global_step + 1
            self.model.lsp.start_step(
                step=current_step,
                epoch=ctx.epoch,
                batch_idx=ctx.batch_idx,
                stage=ctx.stage,
            )
            loss = self.train_step(batch_interaction, ctx)

            loss_value = float(loss.detach().float().cpu().item())
            if current_step % log_interval == 0:
                print(f"step:{current_step} loss:{loss_value}")

            self.global_step += 1
            if self.should_stop_train_epoch(ctx, loss_value):
                break
        return last_batch_idx

    def train_epoch(self):
        with self._timed("train_epoch"):
            self.model.train()
            self._run_train_batches()
            self.epoch += 1

    @torch.no_grad()
    def _evaluate_loader(self, model, loader):
        if loader is None or self.evaluator is None:
            return None

        model.eval()
        for batch_idx, batch_interaction in enumerate(loader):
            batch_interaction = batch_interaction.to(self.cfg.device)
            uids = batch_interaction[self.manager.uid_field]
            labels = batch_interaction[self.manager.label_field]
            if model is self.model:
                scores = self.predict_step(batch_interaction)
            else:
                scores = model.predict(batch_interaction)
            self.evaluator.collect(uids, labels, batch_interaction, batch_preds_1d=scores)

        metrics_result = self.evaluator.summary(self.epoch)
        self.evaluator.clear()
        return metrics_result

    def _evaluate_model(self, model):
        return self._evaluate_loader(model, self.valid_loader)

    @torch.no_grad()
    def evaluate_epoch(self):
        with self._timed("valid_epoch"):
            return self._evaluate_model(self.model)

    @torch.no_grad()
    def evaluate_test(self):
        with self._timed("test_epoch"):
            return self._evaluate_loader(self.model, self.test_loader)

    def run(
            self,
            *,
            stop_condition: Callable[["BaseTrainer", EvaluationFeedback], bool] | None = None,
    ):
        last_metrics = None
        for _ in range(self.cfg.max_epochs):
            if self._has_reached_train_step_limit():
                break
            self.train_epoch()
            self.last_evaluation_feedback = None
            metrics = self.evaluate_epoch()
            last_metrics = metrics
            print(metrics)
            feedback = self.last_evaluation_feedback or EvaluationFeedback(
                metrics_by_target={"student": metrics},
                primary_target="student",
                primary_metrics=metrics,
            )
            should_stop = bool(
                stop_condition is not None and stop_condition(self, feedback)
            )
            if self.early_stopper:
                is_best, early_stop = self.early_stopper.step(metrics, epoch=self.epoch)
                status_text = self.early_stopper.status_text()
                if status_text:
                    print(status_text)
                if is_best:
                    save_path = self.save_checkpoint(tag="best", metrics=metrics)
                    if save_path is not None:
                        print(f"[Checkpoint] saved: {save_path}")
                if early_stop:
                    print(f"[EarlyStop] stop at epoch {self.epoch}")
                    should_stop = True
            if should_stop:
                break
        return last_metrics
````

## File: src/rosen/core/trainers/ema.py
````python
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch

from rosen.core.train.context import (
    EvaluationFeedback,
    TrainContext,
    TrainerComponents,
    TrainerDataLoaders,
)
from rosen.core.train.ema import EMAConfig, EMAController, _parse_ema_decays, ema_decay_tag
from rosen.core.trainers.with_life import BaseTrainerWithLife
from rosen.emb import SchemaManager
from rosen.experiment.param import ConfigBase
from rosen.models.base import BaseModel


class EMATrainer(BaseTrainerWithLife):
    """BaseTrainer variant with model-agnostic EMA teacher checkpoints."""

    def __init__(
            self,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            manager: SchemaManager,
            loaders: TrainerDataLoaders,
            components: TrainerComponents,
            cfg: ConfigBase,
            *,
            ema_config: EMAConfig | None = None,
    ):
        super().__init__(model, optimizer, manager, loaders, components, cfg)

        self.ema_config = ema_config or EMAConfig()
        resolved_decays = _parse_ema_decays(self.ema_config.decays)
        self.ema = None
        if resolved_decays:
            self.ema = EMAController(
                self.model,
                decays=resolved_decays,
                start_step=self.ema_config.start_step,
                update_every=self.ema_config.update_every,
                include_buffers=self.ema_config.include_buffers,
                backend=self.ema_config.backend,
            )
        self.last_eval_metrics_by_checkpoint: Dict[str, Optional[dict]] = {}

    @staticmethod
    def _update_step(ctx: TrainContext) -> int:
        return int(ctx.global_step) + (1 if ctx.stage == "train" else 0)

    def _update_ema(
            self,
            ctx: TrainContext,
            *,
            state_names: tuple[str, ...] | None = None,
    ) -> None:
        if self.ema is not None:
            self.ema.update(
                self.model,
                step=self._update_step(ctx),
                state_names=state_names,
            )

    def train_step(self, batch, ctx: TrainContext):
        loss = super().train_step(batch, ctx)
        self._update_ema(ctx)
        return loss

    def _after_cycle_model_update(
            self,
            ctx: TrainContext,
            state_names: tuple[str, ...],
    ) -> None:
        self._update_ema(ctx, state_names=state_names)

    def _default_primary_checkpoint(self) -> str:
        if self.ema_config.primary_checkpoint:
            return str(self.ema_config.primary_checkpoint)
        if self.ema_config.eval_ema and self.ema is not None and self.ema.decays:
            return f"ema{ema_decay_tag(self.ema.decays[-1])}"
        return "student"

    @torch.no_grad()
    def evaluate_epoch(self):
        if not self.valid_loader or not self.evaluator:
            return None

        metrics_by_checkpoint: Dict[str, Optional[dict]] = {}
        if self.ema_config.eval_student:
            metrics = self._evaluate_model(self.model)
            metrics_by_checkpoint["student"] = metrics
            print(f"[EMA Eval] checkpoint=student metrics={metrics}")

        if self.ema_config.eval_ema and self.ema is not None:
            for checkpoint, _decay, teacher in self.ema.checkpoint_items():
                metrics = self._evaluate_model(teacher)
                metrics_by_checkpoint[checkpoint] = metrics
                print(f"[EMA Eval] checkpoint={checkpoint} metrics={metrics}")

        self.last_eval_metrics_by_checkpoint = metrics_by_checkpoint
        primary = self._default_primary_checkpoint()
        if primary not in metrics_by_checkpoint:
            primary = next(iter(metrics_by_checkpoint), "")
        primary_metrics = metrics_by_checkpoint.get(primary)
        self._notify_evaluation_end(
            primary_metrics,
            feedback=EvaluationFeedback(
                metrics_by_target=metrics_by_checkpoint,
                primary_target=primary,
                primary_metrics=primary_metrics,
            ),
        )
        return primary_metrics

    def _checkpoint_payload(self, metrics: Optional[dict] = None) -> dict:
        payload = super()._checkpoint_payload(metrics)
        payload.update({
            "ema_state_dict": self.ema.state_dict() if self.ema is not None else None,
            "metrics_by_checkpoint": self.last_eval_metrics_by_checkpoint,
        })
        return payload

    def load_checkpoint(self, path: Union[str, Path], *, map_location=None) -> dict:
        checkpoint = super().load_checkpoint(path, map_location=map_location)
        ema_state = checkpoint.get("ema_state_dict")
        if self.ema is not None and ema_state is not None:
            self.ema.load_state_dict(ema_state)
        self.last_eval_metrics_by_checkpoint = dict(checkpoint.get("metrics_by_checkpoint") or {})
        return checkpoint
````

## File: src/rosen/core/trainers/two_stage.py
````python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rosen.core.train.context import EvaluationFeedback
from rosen.core.train.two_stage import (
    StageReport,
    StageSnapshot,
    TwoStagePhase,
    TwoStageTrainingProtocol,
)
from rosen.core.trainers.base import BaseTrainer
from rosen.models.base import BaseModel


TrainerFactory = Callable[[BaseModel, TwoStagePhase], BaseTrainer]


@dataclass(frozen=True)
class TwoStageRunResult:
    """Both independent training sessions and their transition snapshot."""

    stage_one_trainer: BaseTrainer
    stage_two_trainer: BaseTrainer
    stage_one_report: StageReport
    stage_snapshot: StageSnapshot
    stage_two_model: BaseModel
    stage_one_metrics: Optional[dict]
    stage_two_metrics: Optional[dict]


class TwoStageRunner:
    """Run two fresh Trainer instances around a model-owned transition."""

    def __init__(self, trainer_factory: TrainerFactory):
        if not callable(trainer_factory):
            raise TypeError("trainer_factory must be callable.")
        self.trainer_factory = trainer_factory

    @staticmethod
    def _report(trainer: BaseTrainer, feedback: EvaluationFeedback) -> StageReport:
        return StageReport(
            epoch=int(trainer.epoch),
            global_step=int(trainer.global_step),
            metrics_by_target=dict(feedback.metrics_by_target),
            primary_target=str(feedback.primary_target),
            primary_metrics=feedback.primary_metrics,
        )

    def _build_trainer(
            self,
            model: BaseModel,
            stage: TwoStagePhase,
    ) -> BaseTrainer:
        trainer = self.trainer_factory(model, stage)
        if not isinstance(trainer, BaseTrainer):
            raise TypeError(
                "trainer_factory must return a BaseTrainer, "
                f"got {type(trainer).__name__}."
            )
        if trainer.model is not model:
            raise ValueError("trainer_factory must attach the exact model object it receives.")
        return trainer

    @staticmethod
    def _validate_fresh_runtime(
            stage_one_trainer: BaseTrainer,
            stage_two_trainer: BaseTrainer,
    ) -> None:
        if stage_two_trainer is stage_one_trainer:
            raise ValueError("trainer_factory must create a fresh Trainer for stage two.")
        if stage_two_trainer.optimizer is stage_one_trainer.optimizer:
            raise ValueError("trainer_factory must create a fresh optimizer for stage two.")

        stage_one_checkpoint_dir = getattr(stage_one_trainer.cfg, "ckpt_dir", "")
        stage_two_checkpoint_dir = getattr(stage_two_trainer.cfg, "ckpt_dir", "")
        if stage_one_checkpoint_dir and stage_two_checkpoint_dir:
            stage_one_checkpoint_dir = Path(stage_one_checkpoint_dir).expanduser().resolve()
            stage_two_checkpoint_dir = Path(stage_two_checkpoint_dir).expanduser().resolve()
            if stage_one_checkpoint_dir == stage_two_checkpoint_dir:
                raise ValueError(
                    "two-stage Trainers need different checkpoint directories because "
                    "their epoch and step counters restart."
                )

        stage_two_parameter_ids = {
            id(parameter) for parameter in stage_two_trainer.model.parameters()
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in stage_two_trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        unknown_parameter_ids = optimizer_parameter_ids.difference(stage_two_parameter_ids)
        if unknown_parameter_ids:
            raise ValueError("stage-two optimizer contains parameters from another model.")

        for attribute in ("train_loader", "valid_loader"):
            stage_one_loader = getattr(stage_one_trainer, attribute)
            stage_two_loader = getattr(stage_two_trainer, attribute)
            if stage_one_loader is None or stage_two_loader is not stage_one_loader:
                continue
            if iter(stage_two_loader) is stage_two_loader:
                raise ValueError(
                    f"trainer_factory must rebuild one-shot {attribute} for stage two."
                )

        for attribute in (
            "early_stopper",
            "health_monitor",
            "evaluator",
            "timer",
            "recorder",
            "ema",
        ):
            stage_one_value = getattr(stage_one_trainer, attribute, None)
            stage_two_value = getattr(stage_two_trainer, attribute, None)
            if stage_one_value is not None and stage_two_value is stage_one_value:
                raise ValueError(
                    f"trainer_factory must create a fresh {attribute} for stage two."
                )

    def run(self, stage_one_model: BaseModel) -> TwoStageRunResult:
        if not isinstance(stage_one_model, TwoStageTrainingProtocol):
            raise TypeError(
                "stage_one_model must implement TwoStageTrainingProtocol, "
                f"got {type(stage_one_model).__name__}."
            )

        stage_one_trainer = self._build_trainer(
            stage_one_model,
            TwoStagePhase.STAGE_ONE,
        )
        latest_report: StageReport | None = None

        def should_stop(trainer: BaseTrainer, feedback: EvaluationFeedback) -> bool:
            nonlocal latest_report
            latest_report = self._report(trainer, feedback)
            return bool(stage_one_model.should_finish_stage_one(latest_report))

        stage_one_metrics = stage_one_trainer.run(stop_condition=should_stop)
        if latest_report is None:
            feedback = stage_one_trainer.last_evaluation_feedback or EvaluationFeedback(
                metrics_by_target={"student": stage_one_metrics},
                primary_target="student",
                primary_metrics=stage_one_metrics,
            )
            latest_report = self._report(stage_one_trainer, feedback)

        snapshot = StageSnapshot.capture(stage_one_model, latest_report)
        stage_two_model = stage_one_model.initialize_stage_two(snapshot)
        if not isinstance(stage_two_model, BaseModel):
            raise TypeError(
                "initialize_stage_two() must return a BaseModel, "
                f"got {type(stage_two_model).__name__}."
            )
        if stage_two_model is stage_one_model:
            raise ValueError("initialize_stage_two() must return a new model object.")

        stage_two_trainer = self._build_trainer(
            stage_two_model,
            TwoStagePhase.STAGE_TWO,
        )
        self._validate_fresh_runtime(stage_one_trainer, stage_two_trainer)
        stage_two_metrics = stage_two_trainer.run()

        return TwoStageRunResult(
            stage_one_trainer=stage_one_trainer,
            stage_two_trainer=stage_two_trainer,
            stage_one_report=latest_report,
            stage_snapshot=snapshot,
            stage_two_model=stage_two_model,
            stage_one_metrics=stage_one_metrics,
            stage_two_metrics=stage_two_metrics,
        )


__all__ = ["TrainerFactory", "TwoStageRunResult", "TwoStageRunner"]
````

## File: src/rosen/core/trainers/with_life.py
````python
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

from rosen.core.train.context import EvaluationFeedback, TrainContext
from rosen.core.train.hooks import TrainingCycleProtocol, TrainingPhaseProtocol
from rosen.core.trainers.base import BaseTrainer


class BaseTrainerWithLife(BaseTrainer):
    """BaseTrainer plus fixed model training-cycle and health boundaries."""

    def __init__(self, model, optimizer, manager, loaders, components, cfg):
        health_monitor = components.health_monitor
        super().__init__(
            model,
            optimizer,
            manager,
            loaders,
            replace(components, health_monitor=None),
            cfg,
        )
        self.health_monitor = health_monitor

    def _phase(self) -> str:
        if isinstance(self.model, TrainingPhaseProtocol):
            return str(self.model.training_phase)
        return "default"

    def _context(
            self,
            *,
            batch_idx: int,
            stage: str = "train",
            evaluation_feedback: EvaluationFeedback | None = None,
    ) -> TrainContext:
        ctx = super()._context(
            batch_idx=batch_idx,
            stage=stage,
            evaluation_feedback=evaluation_feedback,
        )
        ctx.phase = self._phase()
        return ctx

    @staticmethod
    def _updated_state_names(value) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, tuple) or any(not isinstance(name, str) for name in value):
            raise TypeError(
                "TrainingCycleProtocol methods must return tuple[str, ...] or None."
            )
        return value

    def _after_cycle_model_update(
            self,
            ctx: TrainContext,
            state_names: tuple[str, ...],
    ) -> None:
        del ctx, state_names

    def _call_cycle_method(self, method, ctx: TrainContext, *args) -> tuple[str, ...]:
        self.model.lsp.start_step(
            step=ctx.global_step,
            epoch=ctx.epoch,
            batch_idx=ctx.batch_idx,
            stage=ctx.stage,
        )
        capture = (
            self.health_monitor.capture_scope(self.model, ctx, force=True)
            if self.health_monitor is not None else nullcontext()
        )
        with capture:
            state_names = self._updated_state_names(method(*args, ctx))
        if state_names:
            self._after_cycle_model_update(ctx, state_names)
        if self.health_monitor is not None:
            self.health_monitor.record_training_cycle(
                self.model,
                ctx,
                state_names=state_names,
                phase=self._phase(),
            )
        return state_names

    def train_step(self, batch, ctx: TrainContext):
        capture = (
            self.health_monitor.capture_scope(self.model, ctx)
            if self.health_monitor is not None else nullcontext()
        )
        with capture:
            loss = super().train_step(batch, ctx)
        if self.health_monitor is not None:
            self.health_monitor.record_train_step(self.model, ctx)
        return loss

    def _run_train_batches(self) -> int:
        cycle_model = (
            self.model if isinstance(self.model, TrainingCycleProtocol) else None
        )
        if cycle_model is not None:
            start_ctx = self._context(batch_idx=-1, stage="epoch_start")
            self._call_cycle_method(cycle_model.on_train_epoch_start, start_ctx)

        last_batch_idx = super()._run_train_batches()

        if cycle_model is not None:
            end_ctx = self._context(batch_idx=last_batch_idx, stage="epoch_end")
            self._call_cycle_method(cycle_model.on_train_epoch_end, end_ctx)
        return last_batch_idx

    def _notify_evaluation_end(
            self,
            metrics,
            *,
            feedback: EvaluationFeedback | None = None,
    ) -> None:
        feedback = feedback or EvaluationFeedback(
            metrics_by_target={"student": metrics},
            primary_target="student",
            primary_metrics=metrics,
        )
        self.last_evaluation_feedback = feedback
        if not isinstance(self.model, TrainingCycleProtocol):
            return
        eval_ctx = self._context(
            batch_idx=-1,
            stage="eval_end",
            evaluation_feedback=feedback,
        )
        self._call_cycle_method(self.model.on_eval_epoch_end, eval_ctx, metrics)

    def evaluate_epoch(self):
        metrics = super().evaluate_epoch()
        if self.valid_loader is not None and self.evaluator is not None:
            self._notify_evaluation_end(metrics)
        return metrics

    def _checkpoint_payload(self, metrics=None) -> dict:
        payload = super()._checkpoint_payload(metrics)
        payload["health_monitor_state_dict"] = (
            self.health_monitor.state_dict() if self.health_monitor is not None else None
        )
        return payload

    def load_checkpoint(self, path, *, map_location=None) -> dict:
        checkpoint = super().load_checkpoint(path, map_location=map_location)
        health_state = checkpoint.get("health_monitor_state_dict")
        if self.health_monitor is not None and health_state is not None:
            self.health_monitor.load_state_dict(health_state)
        return checkpoint

    def run(self, *, stop_condition=None):
        metrics = super().run(stop_condition=stop_condition)
        if self.health_monitor is not None:
            end_ctx = self._context(batch_idx=-1, stage="train_end")
            self.health_monitor.flush(self.model, end_ctx)
        return metrics


__all__ = ["BaseTrainerWithLife"]
````

## File: src/rosen/core/enum_type.py
````python
# -*- coding: utf-8 -*-
# @Time   : 2020/8/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``GENERAL``: General Recommendation
    - ``SEQUENTIAL``: Sequential Recommendation
    - ``CONTEXT``: Context-aware Recommendation
    - ``KNOWLEDGE``: Knowledge-based Recommendation
    """

    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    TRADITIONAL = 5
    DECISIONTREE = 6


class KGDataLoaderState(Enum):
    """States for Knowledge-based DataLoader.

    - ``RSKG``: Return both knowledge graph information and user-item interaction information.
    - ``RS``: Only return the user-item interaction.
    - ``KG``: Only return the triplets with negative examples in a knowledge graph.
    """

    RSKG = 1
    RS = 2
    KG = 3


class EvaluatorType(Enum):
    """Type for evaluation metrics.

    - ``RANKING``: Ranking-based metrics like NDCG, Recall, etc.
    - ``VALUE``: Value-based metrics like AUC, etc.
    """

    RANKING = 1
    VALUE = 2


class InputType(Enum):
    """Type of Models' input.

    - ``POINTWISE``: Point-wise input, like ``uid, iid, label``.
    - ``PAIRWISE``: Pair-wise input, like ``uid, pos_iid, neg_iid``.
    """

    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3


class FeatureType(Enum):
    """Type of features.

    - ``TOKEN``: Token features like user_id and item_id.
    - ``FLOAT``: Float features like rating and timestamp.
    - ``TOKEN_SEQ``: Token sequence features like review.
    - ``FLOAT_SEQ``: Float sequence features like pretrained vector.
    """

    TOKEN = "token"
    FLOAT = "float"
    TOKEN_SEQ = "token_seq"
    FLOAT_SEQ = "float_seq"


class FeatureSource(Enum):
    """Source of features.

    - ``INTERACTION``: Features from ``.inter`` (other than ``user_id`` and ``item_id``).
    - ``USER``: Features from ``.user`` (other than ``user_id``).
    - ``ITEM``: Features from ``.item`` (other than ``item_id``).
    - ``USER_ID``: ``user_id`` feature in ``inter_feat`` and ``user_feat``.
    - ``ITEM_ID``: ``item_id`` feature in ``inter_feat`` and ``item_feat``.
    - ``KG``: Features from ``.kg``.
    - ``NET``: Features from ``.net``.
    """
    UNKNOWN = "unknown"
    INTERACTION = "inter"
    USER = "user"
    ITEM = "item"
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    SEQ = "seq"
    SEQ_GROUP = "seq_group"
    KG = "kg"
    NET = "net"
````

## File: src/rosen/core/interaction.py
````python
# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2022/7/8, 2020/9/15, 2020/9/16, 2020/8/12
# @Author  : Zhen Tian, Yupeng Hou, Yushuo Chen, Xingyu Pan
# @email   : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, panxy@ruc.edu.cn

"""
recbole.data.interaction
############################
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils

# 他只是对齐......
def _convert_to_tensor(data):
    """
    增强版转换函数：完美兼容 uint32 并自动对齐 PyTorch 常用类型。
    """
    # 1. 预处理：如果是含有 uint32 的 NumPy 数组，强制转为 int64 (Long)
    # 因为 PyTorch 不支持 uint32，且 ID 类数据转 int64 最保险（防止溢出）
    if isinstance(data, np.ndarray) and data.dtype == np.uint32:
        data = data.astype(np.int64)
    elif isinstance(data, pd.Series) and data.dtype == "uint32":
        data = data.astype(np.int64)

    elem = data[0]

    # 情况 A：单值/数值数组处理
    if isinstance(elem, (float, int, np.floating, np.integer)):
        new_data = torch.as_tensor(data)

    # 情况 B：序列处理 (例如 tags, history_videos)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        # 对每一个子序列进行递归或类型防御
        seq_data = []
        for d in data:
            # 防御子序列中的 uint32
            if isinstance(d, np.ndarray) and d.dtype == np.uint32:
                d = d.astype(np.int64)
            seq_data.append(torch.as_tensor(d))

        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True) # 这里对序列做了pad保证同长度！

    else:
        raise ValueError(f"[{type(elem)}] is not supported!")

    # 3. 类型后置对齐
    # float64 转 float32 是推荐系统的基操，节省一半显存
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    # uint8 虽然 PyTorch 支持，但在做 Embedding 索引时通常需要 Long
    elif new_data.dtype == torch.uint8:
        new_data = new_data.long()

    return new_data


class Interaction(object):
    """The basic class representing a batch of interaction records.

    Note:
        While training, there is no strict rules for data in one Interaction object.

        While testing, it should be guaranteed that all interaction records of one single
        user will not appear in different Interaction object, and records of the same user
        should be continuous. Meanwhile, the positive cases of one user always need to occur
        **earlier** than this user's negative cases.

        A correct example:
            =======     =======     =======
            user_id     item_id     label
            =======     =======     =======
            1           2           1
            1           6           1
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======

        Some wrong examples for Interaction objects used in testing:

        1.
            =======     =======     =======     ============
            user_id     item_id     label
            =======     =======     =======     ============
            1           2           1
            1           6           0           # positive cases of one user always need to

                                                occur earlier than this user's negative cases
            1           3           1
            1           1           0
            2           3           1
            ...         ...         ...
            =======     =======     =======     ============

        2.
            =======     =======     =======     ========
            user_id     item_id     label
            =======     =======     =======     ========
            1           2           1
            1           6           1
            1           3           1
            2           3           1           # records of the same user should be continuous.
            1           1           0
            ...         ...         ...
            =======     =======     =======     ========

    Attributes:
        interaction (dict or pandas.DataFrame): keys are meaningful str (also can be called field name),
            and values are Torch Tensor of numpy Array with shape (batch_size, \\*).
    """

    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(
                        f"The type of {key}[{type(value)}] is not supported!"
                    )
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(
                f"[{type(interaction)}] is not supported for initialize `Interaction`!"
            )
        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, self.interaction[k].unsqueeze(-1).shape[0])

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if "interaction" not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        if isinstance(index, (np.ndarray, torch.Tensor)):
            index = index.tolist()

        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k][index]
        return Interaction(ret)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f"{type(key)} object does not support item assigment")
        self.interaction[key] = value

    def __delitem__(self, key):
        if key not in self.interaction:
            raise KeyError(f"{type(key)} object does not in this interaction")
        del self.interaction[key]

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f"The batch_size of interaction: {self.length}"]
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
            info.append(temp_str)
        info.append("\n")
        return "\n".join(info)

    def __repr__(self):
        return self.__str__()

    @property
    def columns(self):
        """
        Returns:
            list of str: The columns of interaction.
        """
        return list(self.interaction.keys())

    def to(self, device, selected_field=None):
        """Transfer Tensors in this Interaction object to the specified device.

        Args:
            device (torch.device), str: target device.
            selected_field (str or iterable object, optional): if specified, only Tensors
            with keys in selected_field will be sent to device.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to
            the specified device.
        """
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]

        if selected_field is not None:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        else:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def cpu(self):
        """Transfer Tensors in this Interaction object to cpu.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to cpu.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].cpu()
        return Interaction(ret)

    def numpy(self):
        """Transfer Tensors to numpy arrays.

        Returns:
            dict: keys the same as Interaction object, are values are corresponding numpy
            arrays transformed from Tensor.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].numpy()
        return ret

    def repeat(self, sizes):
        """Repeats each tensor along the batch dim.

        Args:
            sizes (int): repeat times.

        Example:
            >>> a = Interaction({'k': torch.zeros(4)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12]), cpu

            >>> a = Interaction({'k': torch.zeros(4, 7)})
            >>> a.repeat(3)
            The batch_size of interaction: 12
                k, torch.Size([12, 7]), cpu

        Returns:
            a copyed Interaction object with repeated Tensors.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat(
                [sizes] + [1] * (len(self.interaction[k].shape) - 1)
            )
        return Interaction(ret)

    def repeat_interleave(self, repeats, dim=0):
        """Similar to repeat_interleave of PyTorch.

        Details can be found in:

            https://pytorch.org/docs/stable/tensors.html?highlight=repeat#torch.Tensor.repeat_interleave

        Note:
            ``torch.repeat_interleave()`` is supported in PyTorch >= 1.2.0.
        """
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat_interleave(repeats, dim=dim)
        return Interaction(ret)

    def update(self, new_inter):
        """Similar to ``dict.update()``

        Args:
            new_inter (Interaction): current interaction will be updated by new_inter.
        """
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]

    def drop(self, column):
        """Drop column in interaction.

        Args:
            column (str): the column to be dropped.
        """
        if column not in self.interaction:
            raise ValueError(f"Column [{column}] is not in [{self}].")
        del self.interaction[column]

    def _reindex(self, index):
        """Reset the index of interaction inplace.

        Args:
            index: the new index of current interaction.
        """
        for k in self.interaction:
            self.interaction[k] = self.interaction[k][index]

    def shuffle(self):
        """Shuffle current interaction inplace."""
        index = torch.randperm(self.length)
        self._reindex(index)

    def sort(self, by, ascending=True):
        """Sort the current interaction inplace.

        Args:
            by (str or list of str): Field that as the key in the sorting process.
            ascending (bool or list of bool, optional): Results are ascending if ``True``, otherwise descending.
                Defaults to ``True``
        """
        if isinstance(by, str):
            if by not in self.interaction:
                raise ValueError(f"[{by}] is not exist in interaction [{self}].")
            by = [by]
        elif isinstance(by, (list, tuple)):
            for b in by:
                if b not in self.interaction:
                    raise ValueError(f"[{b}] is not exist in interaction [{self}].")
        else:
            raise TypeError(f"Wrong type of by [{by}].")

        if isinstance(ascending, bool):
            ascending = [ascending]
        elif isinstance(ascending, (list, tuple)):
            for a in ascending:
                if not isinstance(a, bool):
                    raise TypeError(f"Wrong type of ascending [{ascending}].")
        else:
            raise TypeError(f"Wrong type of ascending [{ascending}].")

        if len(by) != len(ascending):
            if len(ascending) == 1:
                ascending = ascending * len(by)
            else:
                raise ValueError(
                    f"by [{by}] and ascending [{ascending}] should have same length."
                )

        for b, a in zip(by[::-1], ascending[::-1]):
            if len(self.interaction[b].shape) == 1:
                key = self.interaction[b]
            else:
                key = self.interaction[b][..., 0]
            index = np.argsort(key, kind="stable")
            if not a:
                index = torch.tensor(np.array(index)[::-1])
            self._reindex(index)

    def add_prefix(self, prefix):
        """Add prefix to current interaction's columns.

        Args:
            prefix (str): The prefix to be added.
        """
        self.interaction = {
            prefix + key: value for key, value in self.interaction.items()
        }


def cat_interactions(interactions):
    """Concatenate list of interactions to single interaction.

    Args:
        interactions (list of :class:`Interaction`): List of interactions to be concatenated.

    Returns:
        :class:`Interaction`: Concatenated interaction.
    """
    if not isinstance(interactions, (list, tuple)):
        raise TypeError(f"Interactions [{interactions}] should be list or tuple.")
    if len(interactions) == 0:
        raise ValueError(
            f"Interactions [{interactions}] should have some interactions."
        )

    columns_set = set(interactions[0].columns)
    for inter in interactions:
        if columns_set != set(inter.columns):
            raise ValueError(
                f"Interactions [{interactions}] should have some interactions."
            )

    new_inter = {
        col: torch.cat([inter[col] for inter in interactions]) for col in columns_set
    }
    return Interaction(new_inter)
````

## File: src/rosen/data/__init__.py
````python
"""Data utilities for rosen."""

from .dataset import (
    DataScanner,
    DataTransformer,
    ParquetStreamDataset,
    PipelineStreamDataset,
    ProfileParquetStreamDataset,
    ProfileSequenceStore,
    RawParquetStreamDataset,
    ShuffleBuffer,
)
from .padding import FeatureContext, TensorFormatter
from .profile import ProfileDatasetBundle, ProfileDatasetLayout

__all__ = [
    "DataScanner",
    "DataTransformer",
    "FeatureContext",
    "ParquetStreamDataset",
    "ProfileDatasetBundle",
    "ProfileDatasetLayout",
    "PipelineStreamDataset",
    "ProfileParquetStreamDataset",
    "ProfileSequenceStore",
    "RawParquetStreamDataset",
    "ShuffleBuffer",
    "TensorFormatter",
]
````

## File: src/rosen/data/dataset.py
````python
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from rosen.emb import SchemaManager
from .padding import ColumnFormatter, FeatureContext, TensorFormatter


ScannerSource = Union[str, Path, Sequence[Union[str, Path]], pl.LazyFrame]


def _resolve_parquet_paths(source: Union[str, Path, Sequence[Union[str, Path]]]) -> list[str]:
    if isinstance(source, (list, tuple)):
        paths = [str(Path(path)) for path in source]
    else:
        path_obj = Path(source)
        if path_obj.is_dir():
            paths = [str(path) for path in sorted(path_obj.glob("*.parquet"))]
        else:
            paths = [str(path_obj)]

    if not paths:
        raise FileNotFoundError(f"No parquet files found from input: {source}")
    return paths


def _parquet_top_level_columns(path: Union[str, Path]) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _fill_null_scalar(col_type: pa.DataType):
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return pa.scalar("", type=col_type)
    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        return pa.scalar([], type=col_type)
    if pa.types.is_boolean(col_type):
        return pa.scalar(False, type=col_type)
    return pa.scalar(0, type=col_type)


def arrow_table_to_batch_dict(
    table: pa.Table,
    selected_col_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    selected_set = set(selected_col_names) if selected_col_names is not None else None
    batch_dict: Dict[str, np.ndarray] = {}
    for col_name, col_data in zip(table.column_names, table.columns):
        if selected_set is not None and col_name not in selected_set:
            continue

        clean_col = pc.fill_null(col_data, _fill_null_scalar(col_data.type))
        batch_dict[col_name] = clean_col.to_numpy(zero_copy_only=False)
    return batch_dict


def _concat_tables(tables: Sequence[pa.Table]) -> Optional[pa.Table]:
    non_empty_tables = [table for table in tables if table is not None and table.num_rows > 0]
    if not non_empty_tables:
        return None
    if len(non_empty_tables) == 1:
        return non_empty_tables[0]
    return pa.concat_tables(non_empty_tables)


class DataScanner:
    def __init__(
        self,
        source: ScannerSource,
        read_cols: Optional[Iterable[str]] = None,
        filter_expr: Optional[pl.Expr] = None,
    ):
        self.source = source
        self.read_cols = list(dict.fromkeys(read_cols or [])) or None
        self.filter_expr = filter_expr
        self.parquet_paths = None if isinstance(source, pl.LazyFrame) else _resolve_parquet_paths(source)
        self.lazy_frame = self._build_lazy_frame(source)

    def _build_lazy_frame(self, source: ScannerSource) -> pl.LazyFrame:
        if isinstance(source, pl.LazyFrame):
            lf = source
        else:
            lf = pl.scan_parquet(self.parquet_paths)

        if self.filter_expr is not None:
            lf = lf.filter(self.filter_expr)
        if self.read_cols is not None:
            lf = lf.select(self.read_cols)
        return lf

    def _apply_worker_shard(self, lf: pl.LazyFrame, worker_id: int, num_workers: int) -> pl.LazyFrame:
        if num_workers <= 1:
            return lf
        worker_col = "__scanner_worker_row_id"
        return (
            lf.with_row_index(worker_col)
            .filter((pl.col(worker_col) % num_workers) == worker_id)
            .drop(worker_col)
        )

    def _iter_lazy_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        lf = self._apply_worker_shard(self.lazy_frame, worker_id=worker_id, num_workers=num_workers)
        for batch_df in lf.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine="streaming",
        ):
            if batch_df.height == 0:
                continue
            yield batch_df.to_arrow()

    def _iter_parquet_batches(self, batch_size: int, worker_id: int, num_workers: int) -> Iterator[pa.Table]:
        row_group_offset = 0
        for parquet_file in self.parquet_paths or []:
            parquet_file_reader = pq.ParquetFile(parquet_file)
            row_group_count = parquet_file_reader.metadata.num_row_groups
            if num_workers > 1:
                row_groups = [
                    row_group_idx
                    for row_group_idx in range(row_group_count)
                    if (row_group_offset + row_group_idx) % num_workers == worker_id
                ]
                if not row_groups:
                    row_group_offset += row_group_count
                    continue
            else:
                row_groups = None

            for record_batch in parquet_file_reader.iter_batches(
                batch_size=batch_size,
                row_groups=row_groups,
                columns=self.read_cols,
            ):
                yield pa.Table.from_batches([record_batch])
            row_group_offset += row_group_count

    def iter_batches(
        self,
        batch_size: int,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> Iterator[pa.Table]:
        if self.parquet_paths is not None and self.filter_expr is None:
            yield from self._iter_parquet_batches(batch_size, worker_id, num_workers)
            return

        yield from self._iter_lazy_batches(batch_size, worker_id, num_workers)


class DataTransformer:
    def __init__(
        self,
        manager: SchemaManager,
        preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
        filter_expr: Optional[pl.Expr] = None,
        output_col_names: Optional[Iterable[str]] = None,
    ):
        self.manager = manager
        self.preprocess_exprs = list(preprocess_exprs or [])
        self.filter_expr = filter_expr
        self.output_col_names = list(dict.fromkeys(output_col_names or manager.fields()))

    def __call__(self, raw_table: pa.Table) -> Optional[pa.Table]:
        if raw_table.num_rows == 0:
            return None

        raw_df = pl.from_arrow(raw_table)
        if self.preprocess_exprs:
            raw_df = raw_df.with_columns(self.preprocess_exprs)
        if self.filter_expr is not None:
            raw_df = raw_df.filter(self.filter_expr)
            if raw_df.height == 0:
                return None

        transformed_df = (
            self.manager.transform(raw_df.lazy())
            .select(self.output_col_names)
            .collect(engine="streaming")
        )
        if transformed_df.height == 0:
            return None
        return transformed_df.to_arrow()


class ShuffleBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = True,
        selected_col_names: Optional[Sequence[str]] = None,
    ):
        self.capacity = max(batch_size, int(capacity))
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.selected_col_names = list(selected_col_names) if selected_col_names is not None else None
        self.reset()

    def reset(self):
        self.buffer: list[pa.Table] = []
        self.current_rows = 0

    def push(self, table: Optional[pa.Table]):
        if table is None or table.num_rows == 0:
            return
        self.buffer.append(table)
        self.current_rows += table.num_rows

    def _drain(self, force: bool) -> Iterator[Dict[str, np.ndarray]]:
        table = _concat_tables(self.buffer)
        if table is None or table.num_rows == 0:
            self.reset()
            return

        total_rows = table.num_rows
        batch_dict = arrow_table_to_batch_dict(table, selected_col_names=self.selected_col_names)
        indices = np.random.permutation(total_rows) if self.shuffle else np.arange(total_rows)
        num_complete_batches = total_rows // self.batch_size

        for batch_idx in range(num_complete_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield {col: arr[batch_indices] for col, arr in batch_dict.items()}

        residual_start = num_complete_batches * self.batch_size
        residual_indices = indices[residual_start:]
        residual_count = len(residual_indices)

        self.reset()
        if residual_count == 0:
            return

        if force:
            if not self.drop_last:
                yield {col: arr[residual_indices] for col, arr in batch_dict.items()}
            return

        residual_table = table.take(pa.array(residual_indices, type=pa.int64()))
        self.buffer = [residual_table]
        self.current_rows = residual_table.num_rows

    def yield_batches(self) -> Iterator[Dict[str, np.ndarray]]:
        if self.current_rows < self.capacity:
            return
        yield from self._drain(force=False)

    def flush(self) -> Iterator[Dict[str, np.ndarray]]:
        yield from self._drain(force=True)


class PipelineStreamDataset(IterableDataset):
    def __init__(
        self,
        scanner: DataScanner,
        formatter: TensorFormatter,
        transformer: Optional[DataTransformer] = None,
        buffer: Optional[ShuffleBuffer] = None,
        scan_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.scanner = scanner
        self.transformer = transformer
        self.buffer = buffer
        self.formatter = formatter
        self.scan_batch_size = scan_batch_size or (buffer.batch_size if buffer is not None else 4096)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if self.buffer is not None:
            self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            table = self.transformer(raw_table) if self.transformer else raw_table
            if table is None or table.num_rows == 0:
                continue

            if self.buffer is None:
                yield self.formatter.format(
                    arrow_table_to_batch_dict(table, selected_col_names=self.formatter.context.output_col_names)
                )
                continue

            self.buffer.push(table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(batch_dict)

        if self.buffer is not None:
            for batch_dict in self.buffer.flush():
                yield self.formatter.format(batch_dict)


class ParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ):
        context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(parquet_path, read_cols=context.read_col_names)
        formatter = TensorFormatter(context)
        buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=None,
            buffer=buffer,
            scan_batch_size=batch_size * 4 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.col_formatters = formatter.col_formatters


def _empty_object_sequences(batch_size: int) -> np.ndarray:
    values = np.empty(batch_size, dtype=object)
    for idx in range(batch_size):
        values[idx] = []
    return values


def _default_missing_column(name: str, formatter: ColumnFormatter, batch_size: int) -> np.ndarray:
    formatter_name = formatter.__class__.__name__
    if formatter_name.startswith("Padded"):
        return _empty_object_sequences(batch_size)
    if formatter_name in {"DenseFormatter"}:
        return np.zeros(batch_size, dtype=np.float32)
    if formatter_name in {"VectorDenseFormatter"}:
        dim = int(getattr(formatter, "dim", 0))
        return np.zeros((batch_size, dim), dtype=np.float32)
    return np.zeros(batch_size, dtype=np.int64)


class ProfileSequenceStore:
    HISTORY_TIME_FILTERS = {
        "cursor_order",
        "strict_time",
        "drop_same_ts",
        "same_ts_only",
    }

    def __init__(
        self,
        manager: SchemaManager,
        profile_root: Union[str, Path],
        group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
    ):
        self.manager = manager
        self.profile_root = Path(profile_root)
        self.group_dirs = {str(k): Path(v) for k, v in dict(group_dirs or {}).items()}
        self.groups = self._compile_groups()
        self.tables = {
            group_name: self._load_group(group_name, spec)
            for group_name, spec in self.groups.items()
        }

    def _compile_groups(self) -> Dict[str, dict]:
        groups: Dict[str, dict] = {}
        for setting in self.manager.settings:
            if not setting.is_sequence_setting:
                continue
            group_name = setting.group_name
            group = setting.group
            spec = groups.setdefault(
                group_name,
                {
                    "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
                    "seq_len_col": setting.seq_len_field_name,
                    "max_len": int(setting.max_len),
                    "fields": [],
                    "time_field": setting.time_field_name,
                    "anchor_col": getattr(group, "anchor_field_name", None),
                    "history_time_filter": getattr(
                        group,
                        "history_time_filter",
                        "cursor_order",
                    ),
                    "truncate_mode": setting.truncate_mode,
                },
            )
            shared_values = {
                "owner_col": getattr(group, "owner_field_name", None) or self.manager.uid_field,
                "seq_len_col": setting.seq_len_field_name,
                "max_len": int(setting.max_len),
                "time_field": setting.time_field_name,
                "anchor_col": getattr(group, "anchor_field_name", None),
                "history_time_filter": getattr(group, "history_time_filter", "cursor_order"),
                "truncate_mode": setting.truncate_mode,
            }
            for key, value in shared_values.items():
                if spec[key] != value:
                    raise ValueError(
                        f"Profile sequence group '{group_name}' has inconsistent "
                        f"{key}: {spec[key]!r} != {value!r}."
                    )
            if setting.field_name not in spec["fields"]:
                spec["fields"].append(setting.field_name)
            if setting.time_field_name and setting.time_field_name not in spec["fields"]:
                spec["fields"].append(setting.time_field_name)
            if spec["history_time_filter"] not in self.HISTORY_TIME_FILTERS:
                raise ValueError(
                    f"Unknown history_time_filter {spec['history_time_filter']!r} "
                    f"for profile sequence group '{group_name}'."
                )
        return groups

    def _resolve_group_dir(self, group_name: str) -> Path:
        if group_name in self.group_dirs:
            return self.group_dirs[group_name]
        return self.profile_root / group_name

    @staticmethod
    def _profile_value_to_array(value) -> np.ndarray:
        if value is None:
            return np.asarray([], dtype=np.int64)
        try:
            return np.asarray(value, dtype=np.int64)
        except (TypeError, ValueError):
            return np.asarray(value, dtype=object)

    def _load_group(self, group_name: str, spec: dict) -> Dict[int, Dict[str, np.ndarray]]:
        group_dir = self._resolve_group_dir(group_name)
        if group_dir.is_dir():
            files = sorted(group_dir.glob("*.parquet"))
        else:
            files = [group_dir] if group_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No profile parquet files found for group '{group_name}' under {group_dir}")

        read_cols = [spec["owner_col"], *spec["fields"]]
        df = pl.read_parquet([str(path) for path in files], columns=read_cols)
        table: Dict[int, Dict[str, np.ndarray]] = {}
        for row in df.iter_rows(named=True):
            uid = int(row[spec["owner_col"]])
            table[uid] = {}
            for field in spec["fields"]:
                value = row[field]
                table[uid][field] = self._profile_value_to_array(value)
        return table

    def inject(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self.groups:
            return batch_dict
        batch_size = len(next(iter(batch_dict.values())))

        for group_name, spec in self.groups.items():
            owner_col = spec["owner_col"]
            seq_len_col = spec["seq_len_col"]
            if owner_col not in batch_dict or seq_len_col not in batch_dict:
                raise KeyError(
                    f"Profile group '{group_name}' requires columns "
                    f"'{owner_col}' and '{seq_len_col}' in main batch."
                )

            users = np.asarray(batch_dict[owner_col]).reshape(-1)
            cursors = np.asarray(batch_dict[seq_len_col]).reshape(-1)
            table = self.tables[group_name]
            max_len = int(spec["max_len"])
            history_time_filter = spec["history_time_filter"]
            time_field = spec["time_field"]
            anchor_col = spec["anchor_col"]
            anchors = None
            if history_time_filter != "cursor_order":
                if not time_field or not anchor_col:
                    raise ValueError(
                        f"Profile group '{group_name}' requires time_field_name and "
                        f"anchor_field_name for history_time_filter={history_time_filter!r}."
                    )
                if anchor_col not in batch_dict:
                    raise KeyError(
                        f"Profile group '{group_name}' requires anchor column "
                        f"'{anchor_col}' in main batch."
                    )
                anchors = np.asarray(batch_dict[anchor_col]).reshape(-1)

            sliced_fields = {
                field: np.empty(batch_size, dtype=object)
                for field in spec["fields"]
            }
            actual_lens = np.zeros(batch_size, dtype=np.int64)

            for row_idx, (uid, cursor) in enumerate(zip(users, cursors)):
                profile = table.get(int(uid))
                cursor_int = max(0, int(cursor))
                base_field = spec["fields"][0]
                base_seq = profile.get(base_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                end = min(cursor_int, len(base_seq))
                selected_indices = np.arange(end, dtype=np.int64)

                if history_time_filter != "cursor_order":
                    history_time = profile.get(time_field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                    if len(history_time) < end:
                        raise ValueError(
                            f"Profile group '{group_name}' has {len(history_time)} "
                            f"timestamps for cursor {end} (user={int(uid)})."
                        )
                    candidate_time = history_time[:end]
                    anchor = anchors[row_idx]
                    if history_time_filter in {"strict_time", "drop_same_ts"}:
                        selected_indices = selected_indices[candidate_time < anchor]
                    else:
                        selected_indices = selected_indices[candidate_time == anchor]

                if spec["truncate_mode"] == "head":
                    selected_indices = selected_indices[:max_len]
                elif spec["truncate_mode"] == "tail":
                    selected_indices = selected_indices[-max_len:]
                else:
                    raise ValueError(
                        f"Unknown truncate_mode {spec['truncate_mode']!r} "
                        f"for profile sequence group '{group_name}'."
                    )
                actual_lens[row_idx] = len(selected_indices)

                for field in spec["fields"]:
                    full_seq = profile.get(field, np.empty(0, dtype=np.int64)) if profile else np.empty(0, dtype=np.int64)
                    if len(full_seq) < end:
                        raise ValueError(
                            f"Profile group '{group_name}' field '{field}' has "
                            f"length {len(full_seq)} for cursor {end} (user={int(uid)})."
                        )
                    sliced_fields[field][row_idx] = full_seq[selected_indices]

            batch_dict.update(sliced_fields)
            batch_dict[seq_len_col] = actual_lens

        return batch_dict


class ProfileParquetStreamDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        profile_root: Union[str, Path],
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        group_dirs: Optional[Mapping[str, Union[str, Path]]] = None,
        profile_store: Optional[ProfileSequenceStore] = None,
    ):
        super().__init__()
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.context = FeatureContext.from_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        self.formatter = TensorFormatter(self.context)
        self.profile_store = (
            profile_store
            if profile_store is not None
            else ProfileSequenceStore(
                manager=manager,
                profile_root=profile_root,
                group_dirs=group_dirs,
            )
        )

        parquet_paths = _resolve_parquet_paths(parquet_path)
        available_cols = _parquet_top_level_columns(parquet_paths[0])
        read_cols = [col for col in self.context.output_col_names if col in available_cols]
        for spec in self.profile_store.groups.values():
            required_cols = [spec["owner_col"], spec["seq_len_col"]]
            if spec["history_time_filter"] != "cursor_order":
                required_cols.append(spec["anchor_col"])
            for required_col in required_cols:
                if required_col in available_cols and required_col not in read_cols:
                    read_cols.append(required_col)

        self.scanner = DataScanner(parquet_path, read_cols=read_cols)
        self.buffer = ShuffleBuffer(
            capacity=shuffle_buffer_size if shuffle else batch_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=None,
        )
        self.scan_batch_size = batch_size * 4 if shuffle else batch_size
        self.parquet_path = parquet_path
        self.parquet_paths = parquet_paths
        self.profile_root = Path(profile_root)
        self.output_col_names = list(self.context.output_col_names)
        self.valid_col_names = list(self.context.output_col_names)
        self.read_col_names = list(read_cols)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.trigger_size = self.buffer.capacity
        self.col_formatters = self.formatter.col_formatters

    def _complete_batch(self, batch_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not batch_dict:
            return batch_dict
        batch_size = len(next(iter(batch_dict.values())))
        self.profile_store.inject(batch_dict)
        for col in self.context.output_col_names:
            if col in batch_dict:
                continue
            formatter = self.formatter.col_formatters.get(col, self.formatter.fallback_formatter)
            batch_dict[col] = _default_missing_column(col, formatter, batch_size)
        return {col: batch_dict[col] for col in self.context.output_col_names if col in batch_dict}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        self.buffer.reset()

        for raw_table in self.scanner.iter_batches(
            batch_size=self.scan_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            self.buffer.push(raw_table)
            for batch_dict in self.buffer.yield_batches():
                yield self.formatter.format(self._complete_batch(batch_dict))

        for batch_dict in self.buffer.flush():
            yield self.formatter.format(self._complete_batch(batch_dict))


class RawParquetStreamDataset(PipelineStreamDataset):
    def __init__(
        self,
        parquet_path: Union[str, Path, Sequence[Union[str, Path]]],
        manager: SchemaManager,
        batch_size: int = 4096,
        shuffle: bool = True,
        drop_last: bool = True,
        shuffle_buffer_size: int = 2000000,
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
        transform_buffer_size: Optional[int] = None,
        raw_filter_expr: Optional[pl.Expr] = None,
        raw_preprocess_exprs: Optional[Iterable[pl.Expr]] = None,
    ):
        preprocess_exprs = list(raw_preprocess_exprs or [])
        scanner_filter_expr = raw_filter_expr if not preprocess_exprs else None
        transformer_filter_expr = None if scanner_filter_expr is not None else raw_filter_expr
        context = FeatureContext.from_raw_manager(
            manager,
            extra_col_names=extra_col_names,
            extra_col_formatters=extra_col_formatters,
        )
        scanner = DataScanner(
            parquet_path,
            read_cols=context.read_col_names,
            filter_expr=scanner_filter_expr,
        )
        formatter = TensorFormatter(context)
        transformer = DataTransformer(
            manager=manager,
            preprocess_exprs=preprocess_exprs,
            filter_expr=transformer_filter_expr,
            output_col_names=context.output_col_names,
        )
        capacity = transform_buffer_size
        if capacity is None:
            capped_rows = batch_size * (8 if shuffle else 1)
            capacity = max(batch_size, min(shuffle_buffer_size, capped_rows))
        buffer = ShuffleBuffer(
            capacity=capacity,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            selected_col_names=context.output_col_names,
        )
        super().__init__(
            scanner=scanner,
            formatter=formatter,
            transformer=transformer,
            buffer=buffer,
            scan_batch_size=batch_size * 2 if shuffle else batch_size,
        )

        self.parquet_path = parquet_path
        self.parquet_paths = list(scanner.parquet_paths or [])
        self.manager = manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trigger_size = buffer.capacity
        self.context = context
        self.output_col_names = list(context.output_col_names)
        self.valid_col_names = list(context.output_col_names)
        self.read_col_names = list(context.read_col_names)
        self.extra_col_names = list(extra_col_names or [])
        self.extra_col_formatters = dict(extra_col_formatters or {})
        self.raw_filter_expr = raw_filter_expr
        self.raw_preprocess_exprs = preprocess_exprs
        self.transform_buffer_size = buffer.capacity
        self.col_formatters = formatter.col_formatters
````

## File: src/rosen/data/padding.py
````python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Literal, Optional

import numpy as np
import torch

from rosen.core.interaction import Interaction

if TYPE_CHECKING:
    from rosen.emb import SchemaManager


PaddingSide = Literal["left", "right"]


def _is_missing_sequence(value) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _pad_list(
    sequences,
    max_len: int,
    padding_side: PaddingSide = "right",
    pad_val=0,
    dtype=np.int64,
) -> np.ndarray:
    padded = np.full((len(sequences), max_len), pad_val, dtype=dtype)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq) or len(seq) == 0:
            continue

        seq_list = list(seq)
        valid_len = min(len(seq_list), max_len)
        trunc_seq = seq_list[-valid_len:]
        if padding_side == "right":
            padded[i, :valid_len] = trunc_seq
        elif padding_side == "left":
            padded[i, max_len - valid_len:] = trunc_seq
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")

    return padded


def _pad_nested_list(
    sequences,
    max_seq_len: int,
    max_tag_len: int,
    padding_side: PaddingSide = "right",
    pad_val: int = 0,
) -> np.ndarray:
    padded = np.full((len(sequences), max_seq_len, max_tag_len), pad_val, dtype=np.int64)
    for i, seq in enumerate(sequences):
        if _is_missing_sequence(seq) or len(seq) == 0:
            continue

        seq_list = list(seq)
        valid_seq_len = min(len(seq_list), max_seq_len)
        trunc_seq = seq_list[-valid_seq_len:]
        start_idx = 0 if padding_side == "right" else max_seq_len - valid_seq_len

        for j in range(valid_seq_len):
            tags = trunc_seq[j]
            if _is_missing_sequence(tags) or len(tags) == 0:
                continue

            tags_list = list(tags)
            valid_tag_len = min(len(tags_list), max_tag_len)
            trunc_tags = tags_list[:valid_tag_len]
            padded[i, start_idx + j, :valid_tag_len] = trunc_tags

    return padded


class ColumnFormatter(ABC):
    @abstractmethod
    def format(self, data) -> torch.Tensor:
        pass


@dataclass(frozen=True)
class DenseFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)


@dataclass(frozen=True)
class VectorDenseFormatter(ColumnFormatter):
    dim: int
    zero_fill: bool = True

    def format(self, data) -> torch.Tensor:
        arr = np.zeros((len(data), self.dim), dtype=np.float32)
        for i, row in enumerate(data):
            if _is_missing_sequence(row) or len(row) == 0:
                if not self.zero_fill:
                    raise ValueError(
                        f"Expected non-empty vector dim={self.dim}, but got an empty value at row {i}."
                    )
                continue

            row_arr = np.asarray(row, dtype=np.float32).reshape(-1)
            use_len = min(row_arr.shape[0], self.dim)
            arr[i, :use_len] = row_arr[:use_len]

        return torch.tensor(arr, dtype=torch.float32)


@dataclass(frozen=True)
class IntFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        if data.dtype == np.uint32:
            data = data.astype(np.int64)
        return torch.tensor(data, dtype=torch.long)


@dataclass(frozen=True)
class FallbackFormatter(ColumnFormatter):
    def format(self, data) -> torch.Tensor:
        if data.dtype in (np.float32, np.float64):
            return torch.tensor(data, dtype=torch.float32)
        if data.dtype == np.uint32:
            return torch.tensor(data.astype(np.int64), dtype=torch.long)
        return torch.tensor(data)


@dataclass(frozen=True)
class PaddedIntSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0

    def format(self, data) -> torch.Tensor:
        arr = _pad_list(
            data,
            max_len=self.max_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
            dtype=np.int64,
        )
        return torch.tensor(arr, dtype=torch.long)


@dataclass(frozen=True)
class PaddedFloatSequenceFormatter(ColumnFormatter):
    max_len: int
    padding_side: PaddingSide = "right"
    pad_val: float = 0.0

    def format(self, data) -> torch.Tensor:
        arr = _pad_list(
            data,
            max_len=self.max_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
            dtype=np.float32,
        )
        return torch.tensor(arr, dtype=torch.float32)


@dataclass(frozen=True)
class PaddedNestedSequenceFormatter(ColumnFormatter):
    max_seq_len: int
    max_tag_len: int
    padding_side: PaddingSide = "right"
    pad_val: int = 0

    def format(self, data) -> torch.Tensor:
        arr = _pad_nested_list(
            data,
            max_seq_len=self.max_seq_len,
            max_tag_len=self.max_tag_len,
            padding_side=self.padding_side,
            pad_val=self.pad_val,
        )
        return torch.tensor(arr, dtype=torch.long)


DEFAULT_FALLBACK_FORMATTER = FallbackFormatter()


def _dedupe_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(name for name in names if name))


def _resolve_raw_read_col_names(
    manager: "SchemaManager",
    extra_col_names: Iterable[str],
    extra_col_formatters: Dict[str, ColumnFormatter],
) -> list[str]:
    raw_col_names: list[str] = []
    for setting in manager.settings:
        raw_col_names.extend(setting.get_raw_field_names())

    raw_col_names.extend(manager.label_fields)
    raw_col_names.extend(manager.domain_fields)
    if manager.time_field:
        raw_col_names.append(manager.time_field)

    raw_col_names.extend(extra_col_names)
    raw_col_names.extend(extra_col_formatters.keys())
    return _dedupe_names(raw_col_names)


@dataclass(frozen=True)
class FeatureContext:
    manager: "SchemaManager"
    read_col_names: tuple[str, ...]
    output_col_names: tuple[str, ...]
    extra_col_formatters: Dict[str, ColumnFormatter] = field(default_factory=dict)

    @classmethod
    def from_manager(
        cls,
        manager: "SchemaManager",
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ) -> "FeatureContext":
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        return cls(
            manager=manager,
            read_col_names=tuple(output_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )

    @classmethod
    def from_raw_manager(
        cls,
        manager: "SchemaManager",
        extra_col_names: Optional[Iterable[str]] = None,
        extra_col_formatters: Optional[Dict[str, ColumnFormatter]] = None,
    ) -> "FeatureContext":
        formatter_map = dict(extra_col_formatters or {})
        extra_names = list(extra_col_names or [])
        output_cols = _dedupe_names([*manager.fields(), *extra_names, *formatter_map.keys()])
        read_cols = _resolve_raw_read_col_names(manager, extra_names, formatter_map)
        return cls(
            manager=manager,
            read_col_names=tuple(read_cols),
            output_col_names=tuple(output_cols),
            extra_col_formatters=formatter_map,
        )


@dataclass
class TensorFormatter:
    context: FeatureContext
    fallback_formatter: ColumnFormatter = DEFAULT_FALLBACK_FORMATTER
    col_formatters: Dict[str, ColumnFormatter] = field(init=False)

    def __post_init__(self):
        self.col_formatters = self._compile_formatters()

    def _compile_formatters(self) -> Dict[str, ColumnFormatter]:
        formatters: Dict[str, ColumnFormatter] = {}
        for setting in self.context.manager.settings:
            formatters.update(setting.get_formatters())

        for ctx_col in (
            *self.context.manager.label_fields,
            *self.context.manager.domain_fields,
            self.context.manager.time_field,
        ):
            if ctx_col and ctx_col not in formatters:
                formatters[ctx_col] = self.fallback_formatter

        formatters.update(self.context.extra_col_formatters)
        return formatters

    def format_tensors(self, batch_dict) -> Dict[str, torch.Tensor]:
        tensor_dict = {}
        for col, data in batch_dict.items():
            formatter = self.col_formatters.get(col, self.fallback_formatter)
            try:
                tensor_dict[col] = formatter.format(data)
            except Exception as exc:
                raise RuntimeError(
                    f"列 [{col}] 在转换为 Tensor 时发生错误，匹配到的 formatter 为: {formatter}。错误信息: {exc}"
                ) from exc
        return tensor_dict

    def format(self, batch_dict) -> Interaction:
        return Interaction(self.format_tensors(batch_dict))
````

## File: src/rosen/data/profile.py
````python
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from rosen.core.interaction import Interaction
from rosen.emb import SchemaManager

from .dataset import ProfileParquetStreamDataset


_SPLIT_MANIFEST_KEYS = {
    "train": "train_dir",
    "valid": "valid_path",
    "test": "test_path",
}
_SPLIT_FALLBACK_NAMES = {
    "train": ("train.shuffle", "train.parquet", "train.global_shuffle.parquet"),
    "valid": ("valid.parquet",),
    "test": ("test.parquet",),
}


@dataclass(frozen=True)
class ProfileDatasetLayout:
    """Resolve the physical layout of one prepared profile dataset."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProfileDatasetLayout":
        return cls(Path(root).expanduser())

    @property
    def main_dir(self) -> Path:
        return self.root / "main"

    @property
    def manifest_path(self) -> Path:
        return self.main_dir / "split_manifest.json"

    @property
    def schema_dir(self) -> Path:
        return self.root / "schema"

    @property
    def schema_path(self) -> Path:
        return self.schema_dir / "feature_meta.json"

    @property
    def profile_root(self) -> Path:
        return self.root / "profile_seq" / "profiles"

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def split_path(self, split: str) -> Path:
        if split not in _SPLIT_MANIFEST_KEYS:
            raise ValueError(
                f"Unknown profile dataset split {split!r}; "
                f"expected one of {sorted(_SPLIT_MANIFEST_KEYS)}."
            )

        manifest_value = self._manifest().get(_SPLIT_MANIFEST_KEYS[split])
        if manifest_value:
            local_path = self.main_dir / Path(manifest_value).name
            if local_path.exists():
                return local_path

            manifest_path = Path(manifest_value).expanduser()
            if manifest_path.exists():
                return manifest_path

        candidates = [
            self.main_dir / name for name in _SPLIT_FALLBACK_NAMES[split]
        ]
        return next((path for path in candidates if path.exists()), candidates[0])

    def validate(self, splits: Sequence[str] = ("train", "valid", "test")) -> None:
        required = {
            "dataset root": self.root,
            "schema": self.schema_path,
            "profile store": self.profile_root,
            **{f"{split} split": self.split_path(split) for split in splits},
        }
        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not path.exists()
        ]
        if missing:
            formatted = "\n".join(f"  - {entry}" for entry in missing)
            raise FileNotFoundError(
                f"Profile dataset is incomplete under {self.root}:\n{formatted}"
            )


class _MappedInteractionLoader(Iterable[Interaction]):
    def __init__(
        self,
        source: Iterable[Interaction],
        transform: Callable[[Interaction], Interaction],
    ) -> None:
        self.source = source
        self.transform = transform

    def __iter__(self) -> Iterator[Interaction]:
        for interaction in self.source:
            yield self.transform(interaction)


def _identity_collate(batch):
    return batch


class ProfileDatasetBundle:
    """Load schema and logical splits without exposing storage subdirectories."""

    def __init__(self, layout: ProfileDatasetLayout) -> None:
        self.layout = layout

    @classmethod
    def open(cls, root: str | Path) -> "ProfileDatasetBundle":
        return cls(ProfileDatasetLayout.from_root(root))

    @property
    def root(self) -> Path:
        return self.layout.root

    def load_manager(
        self,
        settings,
        *,
        time_field: str,
        label_fields: str | Sequence[str],
        domain_fields: str | Sequence[str],
    ) -> SchemaManager:
        self.layout.validate(splits=())
        manager = SchemaManager(
            settings,
            str(self.layout.schema_dir),
            time_field=time_field,
            label_fields=label_fields,
            domain_fields=domain_fields,
        )
        manager.load_schema()
        return manager

    def loader(
        self,
        split: str,
        manager: SchemaManager,
        *,
        batch_size: int = 4096,
        shuffle: bool = False,
        drop_last: bool = False,
        shuffle_buffer_size: int = 2_000_000,
        num_workers: int = 0,
        prefetch_factor: int = 1,
        transform: Callable[[Interaction], Interaction] | None = None,
    ) -> Iterable[Interaction]:
        self.layout.validate(splits=(split,))
        source: Iterable[Interaction] = ProfileParquetStreamDataset(
            self.layout.split_path(split),
            manager,
            profile_root=self.layout.profile_root,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        worker_count = max(0, int(num_workers))
        if worker_count > 0:
            kwargs: dict[str, Any] = {
                "batch_size": None,
                "num_workers": worker_count,
                "collate_fn": _identity_collate,
            }
            if prefetch_factor > 0:
                kwargs["prefetch_factor"] = int(prefetch_factor)
            source = DataLoader(source, **kwargs)

        if transform is not None:
            source = _MappedInteractionLoader(source, transform)
        return source

    def rows(self, split: str) -> int:
        path = self.layout.split_path(split)
        parquet_paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
        if not parquet_paths or any(not parquet_path.is_file() for parquet_path in parquet_paths):
            raise FileNotFoundError(path)
        return sum(
            pq.ParquetFile(parquet_path).metadata.num_rows
            for parquet_path in parquet_paths
        )
````

## File: src/rosen/emb/schema/__init__.py
````python
from .base import EmbSetting, EmbType, ProfileSeqGroupConfig, SeqGroupConfig
from .categorical import BaseCategoricalSetting, MultiSparseSetting, QuantileEmbSetting, SparseEmbSetting
from .numerical import BaseNumericalSetting, MinMaxDenseSetting, VectorDenseSetting
from .sequence import SequenceSetting
from .utils import NULL_FALLBACK, clear_seq_expr, explode_expr, seq_length_expr

__all__ = [
    "NULL_FALLBACK",
    "EmbSetting",
    "EmbType",
    "SeqGroupConfig",
    "ProfileSeqGroupConfig",
    "BaseCategoricalSetting",
    "BaseNumericalSetting",
    "SparseEmbSetting",
    "MultiSparseSetting",
    "SequenceSetting",
    "QuantileEmbSetting",
    "MinMaxDenseSetting",
    "VectorDenseSetting",
    "clear_seq_expr",
    "explode_expr",
    "seq_length_expr",
]
````

## File: src/rosen/emb/schema/base.py
````python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

if TYPE_CHECKING:
    from rosen.data.padding import ColumnFormatter


class EmbType(Enum):
    UNKNOWN = "none"
    SPARSE = "sparse"
    MULTI_SPARSE = "multi_sparse"
    QUANTILE = "quantile"
    SPARSE_SEQ = "sparse_seq"
    SPARSE_SET = "sparse_set"
    DENSE = "dense"
    VECTOR_DENSE = "vector_dense"
    DENSE_SEQ = "dense_seq"
    SEQUENCE = "sequence"


@dataclass
class SeqGroupConfig:
    group_name: str
    seq_len_field_name: str
    max_len: int
    padding_side: Literal["left", "right"] = "right"
    time_field_name: Optional[str] = None


@dataclass
class ProfileSeqGroupConfig(SeqGroupConfig):
    owner_field_name: str = "user_id"
    id_field_name: Optional[str] = None
    anchor_field_name: Optional[str] = None
    profile_name: Optional[str] = None
    history_time_filter: Literal[
        "cursor_order",
        "strict_time",
        "drop_same_ts",
        "same_ts_only",
    ] = "cursor_order"
    backend: Literal["profile"] = "profile"


class EmbSetting(ABC):
    emb_type = EmbType.UNKNOWN

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource = FeatureSource.UNKNOWN,
        padding_zero: bool = True,
        use_oov: bool = True,
    ):
        self.field_name = field_name
        self.embedding_dim = embedding_dim
        self.source = source
        self.is_fitted = False

        self.padding_zero = padding_zero
        self.use_oov = use_oov
        self.vocab: Dict[str, int] = {}
        self.oov_idx: int = -1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + int(self.padding_zero) + int(self.use_oov)

    @property
    def serialized_type_name(self) -> str:
        return self.emb_type.name

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name}

    @property
    def requires_embedding_module(self) -> bool:
        return self.num_embeddings > 0

    @property
    def embedding_field_name(self) -> str:
        return self.field_name

    @property
    def is_sequence_setting(self) -> bool:
        return False

    def get_output_field_names(self) -> List[str]:
        return [self.field_name]

    def get_raw_field_names(self) -> List[str]:
        return [self.field_name]

    @abstractmethod
    def get_fit_exprs(self) -> List[pl.Expr]:
        pass

    @abstractmethod
    def parse_fit_result(self, result_df: pl.DataFrame):
        pass

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        return self.parse_fit_result(result_df)

    @abstractmethod
    def get_transform_expr(self):
        pass

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element fitting.")

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sequence element transform.")

    @abstractmethod
    def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
        pass

    def get_sequence_formatters(
        self,
        max_len: int,
        padding_side: Literal["left", "right"],
    ) -> Dict[str, "ColumnFormatter"]:
        from rosen.data.padding import (
            DenseFormatter,
            IntFormatter,
            PaddedFloatSequenceFormatter,
            PaddedIntSequenceFormatter,
        )

        base_formatters = self.get_formatters()
        fmt = base_formatters[self.field_name]
        if isinstance(fmt, DenseFormatter):
            return {self.field_name: PaddedFloatSequenceFormatter(max_len=max_len, padding_side=padding_side)}
        if isinstance(fmt, IntFormatter):
            return {self.field_name: PaddedIntSequenceFormatter(max_len=max_len, padding_side=padding_side)}
        raise NotImplementedError(
            f"Formatter {fmt.__class__.__name__} cannot be promoted to a sequence automatically."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.serialized_type_name,
            "field_name": self.field_name,
            "embedding_size": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "vocab_size": self.vocab_size,
            "feature_source": self.source.name,
            "is_fitted": self.is_fitted,
            "padding_zero": self.padding_zero,
            "use_oov": self.use_oov,
            "vocab": self.vocab,
            "oov_idx": self.oov_idx,
        }

    def load_state(self, state_dict: Dict[str, Any]):
        self.is_fitted = state_dict.get("is_fitted", True)
        self.vocab = state_dict.get("vocab", {})
        self.use_oov = state_dict.get("use_oov", self.use_oov)
        self.padding_zero = state_dict.get("padding_zero", self.padding_zero)
        self.oov_idx = state_dict.get("oov_idx", self.oov_idx)

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbSetting":
        pass

    @abstractmethod
    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        pass
````

## File: src/rosen/emb/schema/categorical.py
````python
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType
from .utils import NULL_FALLBACK, map_list_to_indices, mean_pooling


class BaseCategoricalSetting(EmbSetting):
    """离散特征基类：统一管理词表与 OOV 逻辑。"""
    NULL_TOKEN = "__NULL__"

    def __init__(
        self,
        field_name: str,
        embedding_dim: int,
        source: FeatureSource,
        padding_zero: bool = True,
        use_null: bool = True,
        use_oov: bool = True,
        min_freq: int = 1,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero, use_oov)
        if use_null and not padding_zero:
            raise ValueError("use_null=True requires padding_zero=True for sparse categorical features.")
        self.min_freq = min_freq
        self.use_null = use_null
        self.vocab = {}
        self.null_idx = -1
        self.oov_idx = -1

    def _build_vocab_indices(self, valid_vals: List[str]):
        start_idx = 1 if self.padding_zero else 0
        self.vocab = {str(val): idx + start_idx for idx, val in enumerate(valid_vals)}
        next_idx = len(self.vocab) + start_idx
        if self.use_null:
            self.null_idx = next_idx
            next_idx += 1
        else:
            self.null_idx = 0 if self.padding_zero else -1
        if self.use_oov:
            self.oov_idx = next_idx
        else:
            self.oov_idx = 0 if self.padding_zero else -1
        self.is_fitted = True

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return (
            self.vocab_size
            + int(self.padding_zero)
            + int(self.use_null)
            + int(self.use_oov)
        )


class SparseEmbSetting(BaseCategoricalSetting):
    emb_type = EmbType.SPARSE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        padding_zero: bool = True,
        use_null: bool = True,
        fill_null: str = "null",
        min_freq: int = 1,
        use_oov: bool = True,
    ):
        super().__init__(
            field_name,
            embedding_dim,
            source,
            padding_zero,
            use_null=use_null,
            use_oov=use_oov,
            min_freq=min_freq,
        )
        if fill_null not in {"null", "oov", "zero", None}:
            raise ValueError("fill_null must be one of {'null', 'oov', 'zero', None}")
        self.fill_null = fill_null

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name)
            .cast(pl.Utf8)
            .drop_nulls()
            .value_counts()
            .implode()
            .alias(self.field_name)
        ]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        exploded_raw = seq_expr.explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(alias)]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        rows = result_df.get_column(alias).to_list()[0]
        valid_vals = sorted(
            [
                row[alias]
                for row in rows
                if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        replace_kwargs = {"default": pl.lit(self.oov_idx, dtype=pl.UInt32)} if self.oov_idx >= 0 else {}
        mapped_expr = expr.cast(pl.Utf8).replace_strict(self.vocab, **replace_kwargs)
        if not self.use_null:
            return mapped_expr.cast(pl.UInt32)

        if self.fill_null == "zero":
            null_expr = pl.lit(0, dtype=pl.UInt32)
        elif self.fill_null == "oov":
            null_expr = pl.lit(self.oov_idx, dtype=pl.UInt32)
        else:
            null_expr = pl.lit(self.null_idx, dtype=pl.UInt32)
        return pl.when(expr.is_null()).then(null_expr).otherwise(mapped_expr).cast(pl.UInt32)

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def get_formatters(self):
        from rosen.data.padding import IntFormatter

        return {self.field_name: IntFormatter()}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        data["use_null"] = self.use_null
        data["null_idx"] = self.null_idx
        data["fill_null"] = self.fill_null
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.use_null = state_dict.get("use_null", self.use_null)
        self.null_idx = state_dict.get("null_idx", self.null_idx)
        self.fill_null = state_dict.get("fill_null", self.fill_null)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            padding_zero=data.get("padding_zero", True),
            use_null=data.get("use_null", True),
            fill_null=data.get("fill_null", "null"),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.embedding_field_name](interaction[self.field_name])


class QuantileEmbSetting(EmbSetting):
    emb_type = EmbType.QUANTILE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        bucket_count: int = 10,
        embedding_dim: int = 16,
        boundaries: Optional[List[float]] = None,
    ):
        super().__init__(field_name, embedding_dim, source, padding_zero=True, use_oov=False)
        self.bucket_count = bucket_count
        self.boundaries = boundaries if boundaries is not None else []
        if self.boundaries:
            self.is_fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self.boundaries) + 1 if self.boundaries else 0

    @property
    def num_embeddings(self) -> int:
        if not self.is_fitted:
            return -1
        return self.vocab_size + 1

    def get_fit_exprs(self) -> List[pl.Expr]:
        q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
        return [
            pl.col(self.field_name).drop_nulls().cast(pl.Float64).quantile(q).alias(f"{self.field_name}_q_{idx}")
            for idx, q in enumerate(q_list)
        ]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        q_list = np.linspace(0, 1, self.bucket_count + 1)[1:-1]
        exploded = seq_expr.explode().drop_nulls().cast(pl.Float64)
        return [
            exploded.quantile(q).alias(f"{alias}_q_{idx}")
            for idx, q in enumerate(q_list)
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        boundaries = []
        for idx in range(self.bucket_count - 1):
            value = result_df.get_column(f"{alias}_q_{idx}")[0]
            if value is not None and not np.isnan(value):
                boundaries.append(value)
        self.boundaries = sorted(set(boundaries))
        self.is_fitted = True

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        if not self.is_fitted or not self.boundaries:
            return pl.lit(0, dtype=pl.UInt32)

        bucket_count = len(self.boundaries) + 1
        labels = [str(idx) for idx in range(1, bucket_count + 1)]
        return (
            expr
            .cut(breaks=self.boundaries, labels=labels)
            .cast(pl.String)
            .cast(pl.UInt32)
            .fill_null(0)
        )

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def get_formatters(self):
        from rosen.data.padding import IntFormatter

        return {self.field_name: IntFormatter()}

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["bucket_count"] = self.bucket_count
        data["boundaries"] = self.boundaries
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.bucket_count = state_dict.get("bucket_count", self.bucket_count)
        self.boundaries = state_dict.get("boundaries", [])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantileEmbSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            bucket_count=data.get("bucket_count", 10),
            embedding_dim=data["embedding_size"],
            boundaries=data.get("boundaries", []),
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return emb_modules[self.embedding_field_name](interaction[self.field_name])


class MultiSparseSetting(BaseCategoricalSetting):
    emb_type = EmbType.MULTI_SPARSE

    def __init__(
        self,
        field_name: str,
        source: FeatureSource,
        embedding_dim: int = 16,
        max_tag_len: int = 5,
        is_string_format: bool = False,
        separator: str = ",",
        padding_zero: bool = True,
        min_freq: int = 1,
        use_oov: bool = True,
        agg: str = "sum",
    ):
        super().__init__(
            field_name,
            embedding_dim,
            source,
            padding_zero,
            use_null=False,
            use_oov=use_oov,
            min_freq=min_freq,
        )
        self.max_tag_len = max_tag_len
        self.is_string_format = is_string_format
        self.separator = separator
        self.agg = agg

    @property
    def compatible_type_names(self) -> set[str]:
        return {self.serialized_type_name, EmbType.SPARSE_SET.name}

    @property
    def fill_empty_sequence_with_fallback(self) -> bool:
        return False

    def _clean_list_expr(self, expr: pl.Expr, fill_empty_with_fallback: bool = False) -> pl.Expr:
        if self.is_string_format:
            cleaned = (
                expr.cast(pl.Utf8)
                .fill_null("")
                .str.split(self.separator)
                .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
            )
        else:
            cleaned = (
                expr.cast(pl.List(pl.Utf8))
                .fill_null([])
                .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
            )

        if not fill_empty_with_fallback:
            return cleaned

        return (
            pl.when(cleaned.list.len() > 0)
            .then(cleaned)
            .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
        )

    def _truncate_tag_expr(self, expr: pl.Expr) -> pl.Expr:
        return expr.list.head(self.max_tag_len)

    def get_fit_exprs(self) -> List[pl.Expr]:
        exploded_raw = self._truncate_tag_expr(self._clean_list_expr(pl.col(self.field_name))).explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(self.field_name)]

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        nested_expr = seq_expr.list.eval(
            self._truncate_tag_expr(self._clean_list_expr(pl.element(), fill_empty_with_fallback=False))
        )
        exploded_raw = nested_expr.explode().explode().cast(pl.Utf8)
        exploded = exploded_raw.filter(exploded_raw.is_not_null() & (exploded_raw != ""))
        return [exploded.value_counts().implode().alias(alias)]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        rows = result_df.get_column(alias).to_list()[0]
        valid_vals = sorted(
            [
                row[alias]
                for row in rows
                if row[alias] is not None and row.get("count", row.get("counts", 0)) >= self.min_freq
            ]
        )
        self._build_vocab_indices(valid_vals)

    def get_transform_expr(self) -> pl.Expr:
        expr = self._truncate_tag_expr(
            self._clean_list_expr(pl.col(self.field_name), fill_empty_with_fallback=False)
        )
        return map_list_to_indices(expr, self.vocab, self.oov_idx).alias(self.field_name)

    def get_element_transform_expr(self, expr: Optional[pl.Expr] = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        cleaned = self._truncate_tag_expr(
            self._clean_list_expr(expr, fill_empty_with_fallback=False)
        )
        return map_list_to_indices(cleaned, self.vocab, self.oov_idx)

    def get_formatters(self):
        from rosen.data.padding import PaddedIntSequenceFormatter

        return {self.field_name: PaddedIntSequenceFormatter(max_len=self.max_tag_len)}

    def get_sequence_formatters(self, max_len: int, padding_side: str):
        from rosen.data.padding import PaddedNestedSequenceFormatter

        return {
            self.field_name: PaddedNestedSequenceFormatter(
                max_seq_len=max_len,
                max_tag_len=self.max_tag_len,
                padding_side=padding_side,
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_freq"] = self.min_freq
        data["max_tag_len"] = self.max_tag_len
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["agg"] = self.agg
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_freq = state_dict.get("min_freq", 1)
        self.max_tag_len = state_dict.get("max_tag_len", state_dict.get("max_len", self.max_tag_len))
        self.is_string_format = state_dict.get("is_string_format", False)
        self.separator = state_dict.get("separator", ",")
        self.agg = state_dict.get("agg", "sum")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiSparseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data["embedding_size"],
            max_tag_len=data.get("max_tag_len", data.get("max_len", 5)),
            is_string_format=data.get("is_string_format", False),
            separator=data.get("separator", ","),
            padding_zero=data.get("padding_zero", True),
            min_freq=data.get("min_freq", 1),
            use_oov=data.get("use_oov", True),
            agg=data.get("agg", "sum"),
        )
        obj.load_state(data)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        idx_tensor = interaction[self.field_name]
        emb = emb_modules[self.embedding_field_name](idx_tensor)
        emb = torch.sum(emb, dim=-2)
        if self.agg == "mean":
            emb = mean_pooling(emb, idx_tensor, self.padding_zero)
        return emb


if __name__ == "__main__":
    source = FeatureSource.USER

    sparse_null = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=True,
        use_null=True,
        fill_null="null",
        use_oov=True,
    )
    sparse_null._build_vocab_indices(["a", "b"])
    expr = sparse_null.get_transform_expr()
    df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [sparse_null.null_idx, 1, sparse_null.oov_idx]
    assert sparse_null.num_embeddings == 5

    sparse_zero = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=True,
        use_null=True,
        fill_null="zero",
        use_oov=True,
    )
    sparse_zero._build_vocab_indices(["a", "b"])
    expr = sparse_zero.get_transform_expr()
    df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [0, 1, sparse_zero.oov_idx]
    assert sparse_zero.num_embeddings == 5

    sparse_oov = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=True,
        use_null=True,
        fill_null="oov",
        use_oov=True,
    )
    sparse_oov._build_vocab_indices(["a", "b"])
    expr = sparse_oov.get_transform_expr()
    df = pl.DataFrame({"demo": [None, "a", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [sparse_oov.oov_idx, 1, sparse_oov.oov_idx]
    assert sparse_oov.num_embeddings == 5

    sparse_strict = SparseEmbSetting(
        "demo",
        source=source,
        padding_zero=False,
        use_null=False,
        use_oov=False,
    )
    sparse_strict._build_vocab_indices(["1", "2"])
    expr = sparse_strict.get_transform_expr()
    df = pl.DataFrame({"demo": ["1", "2"]}).lazy().with_columns(expr).collect()
    assert df["demo"].to_list() == [0, 1]

    multi = MultiSparseSetting(
        "tags",
        source=source,
        padding_zero=True,
        use_oov=True,
        is_string_format=True,
        separator=",",
    )
    multi._build_vocab_indices(["a", "b"])
    expr = multi.get_transform_expr()
    df = pl.DataFrame({"tags": [None, "", "a,b", "zzz"]}).lazy().with_columns(expr).collect()
    assert df["tags"].to_list() == [[], [], [1, 2], [multi.oov_idx]]

    print("categorical.py self-check passed")
````

## File: src/rosen/emb/schema/numerical.py
````python
from typing import Any, Dict, List, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType


class BaseNumericalSetting(EmbSetting):
    """连续数值特征基类：显式声明无词表。"""

    def __init__(self, field_name: str, source: FeatureSource, embedding_dim: int = 1):
        super().__init__(
            field_name=field_name,
            embedding_dim=embedding_dim,
            source=source,
            padding_zero=False,
            use_oov=False,
        )
        self.is_fitted = False

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def num_embeddings(self) -> int:
        return -1

    @property
    def requires_embedding_module(self) -> bool:
        return False

    def get_formatters(self):
        from rosen.data.padding import DenseFormatter

        return {self.field_name: DenseFormatter()}


class MinMaxDenseSetting(BaseNumericalSetting):
    emb_type = EmbType.DENSE

    def __init__(self, field_name: str, source: FeatureSource, min_val: float = None, max_val: float = None):
        super().__init__(field_name=field_name, source=source, embedding_dim=1)
        self.min_val = min_val
        self.max_val = max_val
        self.is_fitted = min_val is not None and max_val is not None

    def get_fit_exprs(self) -> List[pl.Expr]:
        return [
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().min().alias(f"{self.field_name}_min"),
            pl.col(self.field_name).cast(pl.Float64).drop_nulls().max().alias(f"{self.field_name}_max"),
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        self.parse_element_fit_result(result_df, alias=self.field_name)

    def parse_element_fit_result(self, result_df: pl.DataFrame, alias: str):
        self.min_val = result_df.get_column(f"{alias}_min").to_list()[0]
        self.max_val = result_df.get_column(f"{alias}_max").to_list()[0]
        if self.min_val is None or self.max_val is None:
            self.min_val = 0.0
            self.max_val = 1.0
        elif self.min_val == self.max_val:
            self.max_val = self.min_val + 1e-6
        self.is_fitted = True

    def get_element_fit_exprs(self, seq_expr: pl.Expr, alias: str) -> List[pl.Expr]:
        exploded = seq_expr.explode().cast(pl.Float64)
        return [
            exploded.min().alias(f"{alias}_min"),
            exploded.max().alias(f"{alias}_max"),
        ]

    def get_element_transform_expr(self, expr: pl.Expr = None) -> pl.Expr:
        expr = expr if expr is not None else pl.element()
        range_val = self.max_val - self.min_val
        return ((expr.cast(pl.Float32, strict=False) - self.min_val) / range_val).fill_null(0.0).cast(pl.Float32)

    def get_transform_expr(self) -> pl.Expr:
        return self.get_element_transform_expr(pl.col(self.field_name)).alias(self.field_name)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["min_val"] = self.min_val
        data["max_val"] = self.max_val
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.min_val = state_dict.get("min_val")
        self.max_val = state_dict.get("max_val")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MinMaxDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            min_val=data.get("min_val"),
            max_val=data.get("max_val"),
        )
        obj.is_fitted = data.get("is_fitted", False)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].unsqueeze(-1)


class VectorDenseSetting(BaseNumericalSetting):
    emb_type = EmbType.VECTOR_DENSE

    def __init__(self, field_name: str, source: 'FeatureSource', embedding_dim: Optional[int] = None,
                 zero_fill: bool = True):
        super().__init__(field_name=field_name, source=source, embedding_dim=embedding_dim or 0)
        self.zero_fill = zero_fill
        self.is_fitted = embedding_dim is not None

    def get_fit_exprs(self) -> List[pl.Expr]:
        if self.is_fitted:
            return []

        len_expr = pl.col(self.field_name).drop_nulls().list.len()
        return [
            len_expr.value_counts().implode().alias(f"{self.field_name}_len_counts"),
            pl.col(self.field_name).is_null().sum().alias(f"{self.field_name}_null_count"),
            len_expr.eq(0).sum().alias(f"{self.field_name}_empty_count"),
        ]

    def parse_fit_result(self, result_df: pl.DataFrame):
        if self.is_fitted:
            return

        len_counts_key = f"{self.field_name}_len_counts"
        len_rows = result_df.get_column(len_counts_key).to_list()[0]
        null_count = int(result_df.get_column(f"{self.field_name}_null_count")[0] or 0)
        empty_count = int(result_df.get_column(f"{self.field_name}_empty_count")[0] or 0)

        length_distribution: Dict[int, int] = {}
        for row in len_rows or []:
            count = row.get("count", row.get("counts", 0))
            length = None
            for key, value in row.items():
                if key not in ("count", "counts"):
                    length = value
                    break
            if length is None:
                continue
            length_distribution[int(length)] = int(count)

        non_empty_distribution = {
            int(length): int(count)
            for length, count in length_distribution.items()
            if int(length) > 0 and int(count) > 0
        }

        if not non_empty_distribution:
            raise ValueError(
                f"Feature '{self.field_name}' has no non-empty vectors in training data. "
                f"null_rows={null_count}, empty_rows={empty_count}. "
                "Please set `embedding_dim` explicitly if this feature should be kept."
            )

        if len(non_empty_distribution) != 1:
            dist_lines = [
                f"  {length} -> {count} rows"
                for length, count in sorted(non_empty_distribution.items())
            ]
            distribution_text = "\n".join(dist_lines)
            raise ValueError(
                f"Feature '{self.field_name}' has inconsistent vector lengths on training data.\n"
                f"Observed non-empty length distribution:\n{distribution_text}\n"
                f"null_rows={null_count}, empty_rows={empty_count}\n"
                "Please set `embedding_dim` explicitly for this feature."
            )

        self.embedding_dim = next(iter(non_empty_distribution))
        self.is_fitted = True

    def get_transform_expr(self) -> pl.Expr:
        if not self.is_fitted or not self.embedding_dim:
            raise RuntimeError(f"Feature '{self.field_name}' is not fitted yet. embedding_dim is unknown.")

        # Keep the raw float list column. Fixed-width padding/truncation is
        # handled in the dataset formatter so checkpoints preserve the original
        # vector information instead of silently zeroing mismatched rows.
        return pl.col(self.field_name).cast(pl.List(pl.Float32)).alias(self.field_name)

    def get_formatters(self):
        from rosen.data.padding import VectorDenseFormatter

        return {
            self.field_name: VectorDenseFormatter(
                dim=int(self.embedding_dim),
                zero_fill=self.zero_fill,
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["embedding_dim"] = self.embedding_dim
        data["zero_fill"] = self.zero_fill
        return data

    @property
    def compatible_type_names(self) -> set[str]:
        # Backward compatibility: older checkpoints stored vector dense as DENSE.
        return {self.serialized_type_name, EmbType.DENSE.name}

    def load_state(self, state_dict: Dict[str, Any]):
        super().load_state(state_dict)
        self.embedding_dim = state_dict.get("embedding_dim", self.embedding_dim)
        self.zero_fill = state_dict.get("zero_fill", self.zero_fill)
        self.is_fitted = self.embedding_dim is not None and self.embedding_dim > 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorDenseSetting":
        obj = cls(
            field_name=data["field_name"],
            source=FeatureSource[data["feature_source"]],
            embedding_dim=data.get("embedding_dim", data.get("embedding_size")),
            zero_fill=data.get("zero_fill", True),
        )
        obj.is_fitted = data.get("is_fitted", obj.embedding_dim is not None)
        return obj

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        return interaction[self.field_name].to(torch.float32)
````

## File: src/rosen/emb/schema/sequence.py
````python
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

import polars as pl
import torch
from torch import nn

from rosen.core.enum_type import FeatureSource

from .base import EmbSetting, EmbType, SeqGroupConfig
from .utils import NULL_FALLBACK


def _dedupe_names(names: List[Optional[str]]) -> List[str]:
    return list(dict.fromkeys(name for name in names if name))


def _clean_outer_sequence_expr(
    expr: pl.Expr,
    is_string_format: bool,
    separator: str,
    fill_empty_with_fallback: bool = False,
) -> pl.Expr:
    if is_string_format:
        cleaned = (
            expr.cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        )
    elif fill_empty_with_fallback:
        cleaned = (
            expr.cast(pl.List(pl.Utf8))
            .fill_null([])
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
        )
    else:
        cleaned = expr.fill_null([]).list.eval(pl.element().filter(pl.element().is_not_null()))

    if not fill_empty_with_fallback:
        return cleaned

    return (
        pl.when(cleaned.list.len() > 0)
        .then(cleaned)
        .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
    )


def _clean_time_sequence_expr(expr: pl.Expr, is_string_format: bool, separator: str) -> pl.Expr:
    if is_string_format:
        return (
            expr.cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(
                pl.when(pl.element().str.strip_chars() != "")
                .then(pl.element().str.strip_chars().cast(pl.Int64, strict=False))
                .otherwise(None)
            )
            .list.drop_nulls()
        )
    return (
        expr.cast(pl.List(pl.Int64))
        .fill_null([])
        .list.eval(pl.element().cast(pl.Int64, strict=False).filter(pl.element().is_not_null()))
    )


class SequenceSetting(EmbSetting):
    emb_type = EmbType.SEQUENCE

    def __init__(
        self,
        field_name: str,
        element_setting: EmbSetting,
        group: SeqGroupConfig,
        truncate_mode: str = "tail",
        is_string_format: bool = False,
        separator: str = ",",
    ):
        super().__init__(
            field_name=field_name,
            embedding_dim=element_setting.embedding_dim,
            source=FeatureSource.SEQ,
            padding_zero=element_setting.padding_zero,
            use_oov=element_setting.use_oov,
        )
        self.element_setting = element_setting
        self.group = group
        self.max_len = group.max_len
        self.group_name = group.group_name
        self.seq_len_field_name = group.seq_len_field_name
        self.padding_side = group.padding_side
        self.time_field_name = group.time_field_name
        self.truncate_mode = truncate_mode
        self.is_string_format = is_string_format
        self.separator = separator
        self.is_fitted = element_setting.is_fitted

    @property
    def compatible_type_names(self) -> set[str]:
        legacy = {self.serialized_type_name, EmbType.SPARSE_SEQ.name, EmbType.DENSE_SEQ.name, "SHARE_SEQ"}
        return legacy

    @property
    def is_sequence_setting(self) -> bool:
        return True

    @property
    def embedding_field_name(self) -> str:
        return self.element_setting.embedding_field_name

    @property
    def vocab_size(self) -> int:
        return self.element_setting.vocab_size

    @property
    def num_embeddings(self) -> int:
        return self.element_setting.num_embeddings

    @property
    def requires_embedding_module(self) -> bool:
        return self.element_setting.requires_embedding_module

    def get_output_field_names(self) -> List[str]:
        return _dedupe_names([self.field_name, self.seq_len_field_name, self.time_field_name])

    def get_raw_field_names(self) -> List[str]:
        return _dedupe_names([self.field_name, self.time_field_name])

    def get_clean_expr(self, fill_empty_with_fallback: bool = False, field_name: Optional[str] = None) -> pl.Expr:
        seq_field = field_name or self.field_name
        return _clean_outer_sequence_expr(
            pl.col(seq_field),
            is_string_format=self.is_string_format,
            separator=self.separator,
            fill_empty_with_fallback=fill_empty_with_fallback,
        )

    def truncate_seq_expr(self, expr: pl.Expr) -> pl.Expr:
        if self.truncate_mode == "head":
            return expr.list.head(self.max_len)
        if self.truncate_mode == "tail":
            return expr.list.tail(self.max_len)
        raise ValueError(f"Unknown truncate_mode: {self.truncate_mode}")

    def get_truncated_len_expr(self, raw_len_expr: pl.Expr) -> pl.Expr:
        return (
            pl.min_horizontal(
                raw_len_expr.cast(pl.UInt32),
                pl.lit(self.max_len, dtype=pl.UInt32),
            )
            .alias(self.seq_len_field_name)
        )

    def get_seq_len_expr(self, field_name: Optional[str] = None) -> pl.Expr:
        raw_len = self.get_clean_expr(fill_empty_with_fallback=False, field_name=field_name).list.len()
        return self.get_truncated_len_expr(raw_len)

    def get_fit_seq_expr(self) -> pl.Expr:
        return self.truncate_seq_expr(
            self.get_clean_expr(fill_empty_with_fallback=False)
        )

    def get_fit_exprs(self) -> List[pl.Expr]:
        if self.element_setting.is_fitted:
            self.is_fitted = True
            return []
        return self.element_setting.get_element_fit_exprs(self.get_fit_seq_expr(), alias=self.field_name)

    def parse_fit_result(self, result_df: pl.DataFrame):
        if self.element_setting.is_fitted:
            self.is_fitted = True
            return
        self.element_setting.parse_element_fit_result(result_df, alias=self.field_name)
        self.is_fitted = self.element_setting.is_fitted

    def get_time_transform_expr(self) -> Optional[pl.Expr]:
        if not self.time_field_name:
            return None
        expr = _clean_time_sequence_expr(
            pl.col(self.time_field_name),
            is_string_format=self.is_string_format,
            separator=self.separator,
        )
        return self.truncate_seq_expr(expr).alias(self.time_field_name)

    def get_transform_expr(self) -> List[pl.Expr]:
        expr = self.truncate_seq_expr(
            self.get_clean_expr(fill_empty_with_fallback=False)
        )
        mapped_expr = expr.list.eval(self.element_setting.get_element_transform_expr()).alias(self.field_name)
        exprs: List[pl.Expr] = [mapped_expr, self.get_seq_len_expr()]
        time_expr = self.get_time_transform_expr()
        if time_expr is not None:
            exprs.append(time_expr)
        return exprs

    def get_formatters(self) -> Dict[str, "ColumnFormatter"]:
        from rosen.data.padding import IntFormatter, PaddedIntSequenceFormatter

        element_formatter = next(
            iter(
                self.element_setting.get_sequence_formatters(
                    max_len=self.max_len,
                    padding_side=self.padding_side,
                ).values()
            )
        )
        formatters = {
            self.field_name: element_formatter,
            self.seq_len_field_name: IntFormatter(),
        }
        if self.time_field_name:
            formatters[self.time_field_name] = PaddedIntSequenceFormatter(
                max_len=self.max_len,
                padding_side=self.padding_side,
            )
        return formatters

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["group_name"] = self.group_name
        data["seq_len_field_name"] = self.seq_len_field_name
        data["max_len"] = self.max_len
        data["padding_side"] = self.padding_side
        data["time_field_name"] = self.time_field_name
        if is_dataclass(self.group):
            data["group_config"] = asdict(self.group)
            data["group_config"]["type"] = self.group.__class__.__name__
        data["truncate_mode"] = self.truncate_mode
        data["is_string_format"] = self.is_string_format
        data["separator"] = self.separator
        data["element_setting_field_name"] = self.element_setting.field_name
        data["element_setting_type"] = self.element_setting.serialized_type_name
        data["element_setting_state"] = self.element_setting.to_dict()
        return data

    def load_state(self, state_dict: Dict[str, Any]):
        self.group_name = state_dict.get("group_name", self.group_name)
        self.seq_len_field_name = state_dict.get("seq_len_field_name", self.seq_len_field_name)
        self.max_len = state_dict.get("max_len", self.max_len)
        self.padding_side = state_dict.get("padding_side", self.padding_side)
        self.time_field_name = state_dict.get("time_field_name", self.time_field_name)
        self.truncate_mode = state_dict.get("truncate_mode", self.truncate_mode)
        self.is_string_format = state_dict.get("is_string_format", self.is_string_format)
        self.separator = state_dict.get("separator", self.separator)
        element_state = state_dict.get("element_setting_state")
        if element_state:
            self.element_setting.load_state(element_state)
        self.is_fitted = self.element_setting.is_fitted

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SequenceSetting":
        raise NotImplementedError("SequenceSetting 需在 Python 代码中显式注入 element_setting 依赖。")

    def compute_tensor(self, interaction: dict, emb_modules: nn.ModuleDict) -> torch.Tensor:
        mock_interaction = {
            self.field_name: interaction[self.field_name],
            self.element_setting.field_name: interaction[self.field_name],
        }
        return self.element_setting.compute_tensor(mock_interaction, emb_modules)
````

## File: src/rosen/emb/schema/utils.py
````python
from typing import Dict

import polars as pl
import torch


# Historical compatibility placeholder.
# We no longer use this in the main feature pipeline:
# - sparse nulls are mapped directly to a reserved null index
# - sequence empties are kept empty and padded by the dataset formatter
NULL_FALLBACK = "NULL_FALLBACK"


def explode_expr(field_name: str, is_string_format: bool = True, separator: str = ",") -> pl.Expr:
    expr = pl.col(field_name).drop_nulls()
    if is_string_format:
        exploded = (
            expr.cast(pl.Utf8)
            .str.split(separator)
            .explode()
            .cast(pl.Utf8)
            .str.strip_chars()
        )
    else:
        exploded = expr.explode().cast(pl.Utf8)
    valid_mask = exploded.is_not_null() & (exploded != "")
    return exploded.filter(valid_mask)


def clear_seq_expr(
    field_name: str,
    is_string_format: bool,
    separator: str,
    fill_empty_with_fallback: bool = False,
) -> pl.Expr:
    if is_string_format:
        expr = (
            pl.col(field_name)
            .cast(pl.Utf8)
            .fill_null("")
            .str.split(separator)
            .list.eval(pl.element().str.strip_chars().filter(pl.element() != ""))
        )
    else:
        expr = (
            pl.col(field_name)
            .cast(pl.List(pl.Utf8))
            .fill_null([])
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element() != "")))
        )

    if not fill_empty_with_fallback:
        return expr

    return (
        pl.when(expr.list.len() > 0)
        .then(expr)
        .otherwise(pl.lit([NULL_FALLBACK], dtype=pl.List(pl.Utf8)))
    )


def seq_length_expr(
    field_name: str,
    is_string_format: bool,
    separator: str,
    alias: str,
) -> pl.Expr:
    clean_expr = clear_seq_expr(
        field_name=field_name,
        is_string_format=is_string_format,
        separator=separator,
        fill_empty_with_fallback=False,
    )
    return clean_expr.list.len().cast(pl.UInt32).alias(alias)


def map_list_to_indices(expr: pl.Expr, vocab: Dict[str, int], oov_idx: int) -> pl.Expr:
    keys = pl.Series(list(vocab.keys()), dtype=pl.Utf8)
    vals = pl.Series(list(vocab.values()), dtype=pl.UInt32)
    default_expr = pl.lit(oov_idx, dtype=pl.UInt32) if oov_idx >= 0 else pl.lit(None, dtype=pl.UInt32)
    return expr.list.eval(
        pl.element()
        .replace_strict(old=keys, new=vals, default=default_expr)
        .cast(pl.UInt32)
    )


def mean_pooling(emb: torch.Tensor, idx_tensor: torch.Tensor, padding_zero: bool) -> torch.Tensor:
    if padding_zero:
        mask_sum = (idx_tensor != 0).sum(dim=-1, keepdim=True).clamp(min=1)
    else:
        mask_shape = idx_tensor.shape[:-1] + (1,)
        mask_sum = torch.full(mask_shape, idx_tensor.shape[-1], dtype=emb.dtype, device=idx_tensor.device)
        return emb / mask_sum
    return emb / mask_sum.to(dtype=emb.dtype)
````

## File: src/rosen/emb/__init__.py
````python
from .manager import SchemaManager
````

## File: src/rosen/emb/emblayer.py
````python
from typing import Iterable, List, Dict, Union, Literal, TYPE_CHECKING, Optional, Any

import numpy as np
import pandas as pd
import torch
from rosen.emb.schema import (
    EmbSetting,
    MultiSparseSetting,
    SequenceSetting,
)
from rosen.core.enum_type import FeatureSource
from torch import nn

from rosen.core.interaction import Interaction

if TYPE_CHECKING:
    from rosen.emb.manager import SchemaManager


# 将settings中的col分成用户侧和物品侧，BoleEmbLayer会把宽表Interaction 变成对应的Dict Embedding


SPLIT_METHODS = Literal["source", "name", "none", None]


class RecEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, init_method='normal', init_std=1e-4):
        super(RecEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.init_method = init_method
        self.init_std = init_std
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            pass
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def reinitialize(self, init_method: Optional[str] = None, init_std: Optional[float] = None):
        if init_method is not None:
            self.init_method = init_method
        if init_std is not None:
            self.init_std = init_std
        self.reset_parameters()

    def forward(self, x):
        return self.embedding(x)


class BoleEmbLayer(nn.Module):
    def __init__(self, emb_settings: Iterable[EmbSetting]):
        super(BoleEmbLayer, self).__init__()
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        for setting in self.settings:
            if self.source2settings.get(setting.source) is None:
                self.source2settings[setting.source] = []
            self.source2settings[setting.source].append(setting)

            if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
                self.emb_modules[setting.embedding_field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero]
                )

    def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor]:
        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "none", None):
            raise ValueError

        output_embs = {}
        for source, settings in self.source2settings.items():
            if not settings:
                continue
            emb_list = []
            for setting in settings:
                emb = setting.compute_tensor(interaction, self.emb_modules)
                emb_list.append([setting.field_name, emb])

            if emb_list:
                if split_by == "source":
                    output_embs[source] = torch.cat([emb for name, emb in emb_list], dim=-1)
                elif split_by == "name":
                    for name, emb in emb_list:
                        output_embs[name] = emb
                else:
                    output_embs["all"] = output_embs.get("all", [])
                    output_embs["all"].extend([emb for name, emb in emb_list])

        if split_by is None or split_by == "none":
            if output_embs.get("all", None) is None:
                return None
            else:
                return torch.cat(output_embs.get("all"), dim=-1)
        else:
            return output_embs


class SideEmb(nn.Module):
    def __init__(self, all_settings: Iterable[EmbSetting],
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]]=None):
        super(SideEmb, self).__init__()

        # 处理单 Source 和多 Source 的兼容

        if target_sources is not None:
            if not isinstance(target_sources, Iterable):
                target_sources = [target_sources]
            # 核心逻辑：过滤出属于自己这一侧的 settings
            side_settings = [s for s in all_settings if s.source in target_sources]
        else:
            side_settings = all_settings

        self.target_sources = target_sources  # 转为 tuple 方便后续固定顺序遍历
        self.embedding = BoleEmbLayer(side_settings)
        self.embedding_size = sum([s.embedding_dim for s in side_settings])

    def forward(self, interaction, split_by: SPLIT_METHODS="none") -> Union[torch.Tensor, Dict[FeatureSource, torch.Tensor]]:
        # 直接把大宽表吐出的 interaction 传给底层
        emb_dict = self.embedding(interaction, split_by)
        return emb_dict


# 具体化实现，极其干净优雅
class UserSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.USER_ID, FeatureSource.USER)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)


class ItemSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.ITEM_ID, FeatureSource.ITEM)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)


class InterSideEmb(SideEmb):
    VALID_SOURCE = (FeatureSource.INTERACTION,)

    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings, self.VALID_SOURCE)

# 无任何约束
class CustomSideEmb(SideEmb):
    def __init__(self, all_settings: Iterable[EmbSetting]):
        super().__init__(all_settings)

"""已废弃"""
class ProfileEncoder(nn.Module):
    """
    全自动静态画像编码器：告别硬编码，完全数据驱动！
    """

    def __init__(self, settings: List['EmbSetting'], profile_path: str,
                 id_source: FeatureSource, feature_source: FeatureSource):
        super().__init__()

        # 1. 解析图纸：找出谁是主键 (ID)，谁是属性 (Features)
        self.id_setting = next((s for s in settings if s.source == id_source), None)
        if not self.id_setting:
            raise ValueError(f"在 settings 中未找到主键源: {id_source}")

        self.feature_settings = [s for s in settings if s.source == feature_source]
        self.feature_names = [s.field_name for s in self.feature_settings]

        # 2. 初始化底层 Embedding 层 (直接复用你写好的 SideEmb)
        self.side_emb = SideEmb(settings, target_sources=[id_source, feature_source])

        # 3. 读取静态宽表
        df = pd.read_parquet(profile_path)

        # 4. 构建地基 (最大 ID 决定了查找表的行数)
        id_col = self.id_setting.field_name
        num_entities = df[id_col].max() + 1
        entity_ids = torch.tensor(df[id_col].astype(np.int64).values, dtype=torch.long)

        # 5. 全自动装配流水线
        for setting in self.feature_settings:
            field = setting.field_name

            # --- 分支 A：处理变长序列特征 (必须 Padding) ---
            if isinstance(setting, MultiSparseSetting):
                max_len = getattr(setting, 'max_tag_len', 50)
                padded_matrix = self._pad_sequences(df[field], max_len)

                # 初始化空矩阵 [num_entities, max_len]
                buffer_tensor = torch.zeros((num_entities, max_len), dtype=torch.long)
                vals = torch.tensor(padded_matrix, dtype=torch.long)

            # --- 分支 B：处理单值特征 ---
            else:
                # 初始化一维向量 [num_entities]
                buffer_tensor = torch.zeros(num_entities, dtype=torch.long)
                clean_series = df[field].fillna(0).astype(np.int64)
                vals = torch.tensor(clean_series.values, dtype=torch.long)

            # 散列赋值：自动对齐坑位 (完美解决残缺错位问题)
            buffer_tensor[entity_ids] = vals

            # 动态注册为模型的 Buffer (名字比如: buf_tag, buf_author_id)
            self.register_buffer(f"buf_{field}", buffer_tensor)

    def _pad_sequences(self, series: pd.Series, max_len: int, pad_val: int = 0) -> np.ndarray:
        """高性能 Numpy 序列补齐工具，自带数据清洗"""
        padded = np.full((len(series), max_len), pad_val, dtype=np.int64)
        for i, seq in enumerate(series):
            # 防御由于 left join 产生的 NaN/None
            if seq is None or (isinstance(seq, float) and np.isnan(seq)) or len(seq) == 0:
                continue

            valid_len = min(len(seq), max_len)
            # 【核心修复】不仅截断，还要强制清洗内部元素的类型为 int64
            padded[i, :valid_len] = np.array(seq[:valid_len], dtype=np.int64)

        return padded

    def forward(self, query_ids: torch.Tensor, split_by: SPLIT_METHODS= "none"):
        """
        前向传播：根据传入的 query_ids (比如一串 item_id)，
        瞬间从缓存中拉出所有特征，组装成 interaction 字典，喂给 SideEmb。
        """
        # 1. 构建虚拟 Interaction 字典
        interaction = {self.id_setting.field_name: query_ids}
        # 2. 动态查表
        for field in self.feature_names:
            # 取出注册好的 Buffer 矩阵
            buffer_tensor = getattr(self, f"buf_{field}")
            interaction[field] = buffer_tensor[query_ids]

        # 3. 完美接入你的 SideEmb
        return self.side_emb(interaction, split_by=split_by)

class UserProfileEncoder(ProfileEncoder):
    ID_SOURCE = FeatureSource.USER_ID
    FEATURE_SOURCE = FeatureSource.USER
    def __init__(self, settings: List['EmbSetting'], profile_path: str,):
        super().__init__(settings, profile_path,
                         id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)

class ItemProfileEncoder(ProfileEncoder):
    ID_SOURCE = FeatureSource.ITEM_ID
    FEATURE_SOURCE = FeatureSource.ITEM
    def __init__(self, settings: List['EmbSetting'], profile_path: str,):
        super().__init__(settings, profile_path,
                         id_source=self.ID_SOURCE, feature_source=self.FEATURE_SOURCE)


class SeqEmbedder(nn.Module):
    def __init__(self, seq_field_name, settings: List[EmbSetting], profile_encoder: ProfileEncoder):
        super().__init__()
        self.settings = settings
        filtered_list = [setting for setting in settings if setting.field_name == seq_field_name]
        if len(filtered_list) != 1:
            raise ValueError
        self.id_setting = filtered_list[0]
        if not isinstance(self.id_setting, SequenceSetting):
            raise ValueError
        self.id_setting: SequenceSetting = self.id_setting
        self.seq_field_name = seq_field_name
        self.seq_len_field_name = self.id_setting.seq_len_field_name

        self.profile_encoder = profile_encoder

    def forward(self, interaction: Interaction, split_by: SPLIT_METHODS="none"):
        inter_dict = interaction.interaction
        seq = inter_dict[self.seq_field_name]
        seq_len = inter_dict[self.seq_len_field_name]
        emb_seq = self.profile_encoder.forward(seq, split_by)
        return emb_seq, seq_len


class EmbView:
    """
    轻量级特征视图：
    本身没有任何参数，只负责将请求转发给底层的 OmniEmbLayer，并自动绑定 target_sources。
    """
    def __init__(self,
                 omni_layer: 'OmniEmbLayer',
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                 include_fields: Iterable[str] = None,
                 exclude_fields: Iterable[str] = None):
        super().__init__()
        # 引用同一个巨无霸底座
        self.omni = omni_layer
        self.target_sources = target_sources
        self.include_fields = tuple(include_fields) if include_fields is not None else None
        self.exclude_fields = tuple(exclude_fields) if exclude_fields is not None else None

    def forward(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    def __call__(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[
        Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def embedding_dim(self) -> int:
        return self.omni.get_output_dim(
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def selected_settings(self) -> tuple[EmbSetting, ...]:
        return tuple(
            self.omni.get_settings(
                target_sources=self.target_sources,
                include_fields=self.include_fields,
                exclude_fields=self.exclude_fields,
            )
        )

    @property
    def field_segments(self) -> tuple[tuple[str, int, int], ...]:
        cursor = 0
        segments = []
        for setting in self.selected_settings:
            start = cursor
            cursor += int(setting.embedding_dim)
            segments.append((setting.field_name, start, cursor))
        return tuple(segments)

    @property
    def field_slices(self) -> Dict[str, slice]:
        return {
            field_name: slice(start, stop)
            for field_name, start, stop in self.field_segments
        }


class SeqGroupView(EmbView):
    """
    专为序列组设计的强化版视图：
    不仅能获取序列特征，还能自动获取严格对齐的 Target 特征和序列长度。
    """

    def __init__(self, omni_layer, group_name: str, include_fields: Iterable[str]):
        super().__init__(omni_layer, include_fields=include_fields)
        self.group_name = group_name

        # 预加载对齐的 element 设置与 seq_len 字段
        self.target_settings = []
        self.seq_len_field = None
        self.time_field = None
        self._target_forward_valid = True

        for field in self.include_fields:
            setting = next((s for s in self.omni.settings if s.field_name == field), None)
            if setting is None:
                continue

            if hasattr(setting, 'element_setting'):
                self.target_settings.append(setting.element_setting)
            else:
                self._target_forward_valid = False

            if self.seq_len_field is None and hasattr(setting, 'seq_len_field_name'):
                self.seq_len_field = setting.seq_len_field_name
            if hasattr(setting, 'time_field_name'):
                setting_time_field = setting.time_field_name
                if setting_time_field is None:
                    continue
                if self.time_field is None:
                    self.time_field = setting_time_field
                elif self.time_field != setting_time_field:
                    raise ValueError(
                        f"Seq group '{self.group_name}' has inconsistent time_field_name: "
                        f"{self.time_field} vs {setting_time_field}"
                    )

    def forward_target(self, interaction, split_by: str = "none"):
        # 实际上所有 setting 都应暴露 element_setting，forward_target 才不会有歧义
        if split_by == "source":
            raise ValueError("Target sequence view strictly supports split_by='none' or 'name'.")

        computed_results = []
        for target_s in self.target_settings:
            emb = target_s.compute_tensor(interaction, self.omni.emb_modules)
            computed_results.append((target_s, emb))
        return self.omni.format_output(computed_results, split_by)

    def fetch_all(self, interaction, include_time: bool = False):
        """默认返回 (seq_emb, target_emb, seq_len)，需要时间时返回四元组。"""
        seq_emb = self.forward(interaction, split_by="none")
        tar_emb = (
            self.forward_target(interaction, split_by="none")
            if self._target_forward_valid
            else ValueError("存在部分 setting 没有 element_setting，函数中断")
        )
        seq_len = interaction[self.seq_len_field] if self.seq_len_field else None
        hist_time = interaction[self.time_field] if self.time_field else None
        if include_time:
            return seq_emb, tar_emb, seq_len, hist_time
        return seq_emb, tar_emb, seq_len

class OmniEmbLayer(nn.Module):
    """
    巨无霸统一 Embedding 层：
    内部纳管所有 settings。forward 时可通过 target_sources 动态指定范围。
    彻底取代 SideEmb、UserSideEmb、ItemSideEmb 等一众子类。
    """
    def __init__(self, emb_settings: Iterable[EmbSetting] = None, manager: 'SchemaManager' = None, init_method='normal', init_std=1e-3):
        super(OmniEmbLayer, self).__init__()
        if manager is not None:
            emb_settings = manager.settings
        if emb_settings is None:
            raise ValueError("OmniEmbLayer requires manager or emb_settings.")

        self.manager = manager
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        self.group2fields = {}
        self._init_embedding(init_method=init_method, init_std=init_std)

        self.seq_groups: Dict[str, SeqGroupView] = {}
        for g_name, fields in self.group2fields.items():
            self.seq_groups[g_name] = SeqGroupView(self, group_name=g_name, include_fields=fields)

        self.whole = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                                                   FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                   FeatureSource.INTERACTION))
        self.user_all = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER))
        self.item_all = EmbView(self, target_sources=(FeatureSource.ITEM_ID, FeatureSource.ITEM))
        self.inter = EmbView(self, target_sources=(FeatureSource.INTERACTION,))

        self.user_id = EmbView(self, target_sources=(FeatureSource.USER_ID,))
        self.item_id = EmbView(self, target_sources=(FeatureSource.ITEM_ID,))


        domain_fields = tuple(manager.domain_fields) if manager is not None else tuple()
        domain_field = manager.domain_field if manager is not None else None
        self.domain = EmbView(self, include_fields=domain_fields)
        self.domain_id = EmbView(
            self,
            include_fields=((domain_field,) if domain_field is not None else tuple()),
        )
        self.whole_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                            FeatureSource.ITEM_ID, FeatureSource.ITEM,
                            FeatureSource.INTERACTION),
            exclude_fields=domain_fields,
        )
        self.inter_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.INTERACTION,),
            exclude_fields=domain_fields,
        )

    def _init_embedding(self, init_method='normal', init_std=1e-3):
        for setting in self.settings:
            # A. 维护 Source 映射
            self.source2settings.setdefault(setting.source, []).append(setting)

            # B. 确定哪些特征需要物理权重 (nn.Embedding)
            if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
                self.emb_modules[setting.embedding_field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero],
                    init_method=init_method, init_std=init_std
                )

            # C. 自动收集带有组名标签的序列特征
            if hasattr(setting, 'group_name') and setting.group_name:
                self.group2fields.setdefault(setting.group_name, []).append(setting.field_name)

    def reinitialize_large_vocab_embeddings(
            self,
            vocab_size_threshold: int,
            *,
            init_method: Optional[str] = None,
            init_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reinitialize embedding modules whose linked setting vocab size exceeds the threshold.

        Notes:
        - This method is opt-in only. Nothing in the default training flow calls it.
        - When multiple settings share one embedding table, the table is reinitialized at most once.
        - The threshold compares `setting.vocab_size`, not `num_embeddings`, so padding/OOV slots
          do not affect the trigger.
        """
        vocab_size_threshold = int(vocab_size_threshold)
        embedding_meta: Dict[str, Dict[str, Any]] = {}

        for setting in self.settings:
            if not setting.requires_embedding_module:
                continue

            emb_field_name = setting.embedding_field_name
            meta = embedding_meta.setdefault(
                emb_field_name,
                {
                    "embedding_field_name": emb_field_name,
                    "field_names": [],
                    "vocab_size": 0,
                    "num_embeddings": int(setting.num_embeddings),
                },
            )
            if setting.field_name not in meta["field_names"]:
                meta["field_names"].append(setting.field_name)
            meta["vocab_size"] = max(int(meta["vocab_size"]), int(setting.vocab_size))
            meta["num_embeddings"] = max(int(meta["num_embeddings"]), int(setting.num_embeddings))

        reinitialized = []
        for emb_field_name, meta in embedding_meta.items():
            if meta["vocab_size"] <= vocab_size_threshold:
                continue

            if emb_field_name not in self.emb_modules:
                continue
            emb_module = self.emb_modules[emb_field_name]

            emb_module.reinitialize(init_method=init_method, init_std=init_std)
            reinitialized.append(meta)

        return sorted(reinitialized, key=lambda item: item["vocab_size"], reverse=True)


    def forward(
            self,
            interaction,
            split_by: SPLIT_METHODS = "none",
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:

        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "none", None):
            raise ValueError(f"Invalid split_by: {split_by}")

        # 1. 路由过滤：找出本次 forward 需要处理的所有 Setting
        valid_settings = self._filter_settings(target_sources, include_fields, exclude_fields, split_by)
        computed = [(s, s.compute_tensor(interaction, self.emb_modules)) for s in valid_settings]
        return self.format_output(computed, split_by)


    def _filter_settings(self, target_sources, include_fields, exclude_fields, split_by) -> List[EmbSetting]:
        """职责一：专门负责解析 target_sources 和 include/exclude_fields，返回合法的 Setting 列表"""
        target_sources = set(target_sources) if target_sources else None
        include_fields = set(include_fields) if include_fields else None
        exclude_fields = set(exclude_fields) if exclude_fields else None
        valid_settings = []
        for source, settings in self.source2settings.items():
            if target_sources and source not in target_sources:
                continue
            for setting in settings:
                if include_fields and setting.field_name not in include_fields:
                    continue
                if exclude_fields and setting.field_name in exclude_fields:
                    continue

                if (split_by == "none" or split_by is None) and setting.is_sequence_setting:
                    if not (include_fields and setting.field_name in include_fields):
                        continue

                valid_settings.append(setting)
        return valid_settings

    def format_output(self, computed_results: List[tuple], split_by: str):
        """职责三：根据 split_by 策略打包最终输出的 Tensor 或 Dict"""
        if not computed_results:
            return None

        if split_by == "name":
            return {setting.field_name: emb for setting, emb in computed_results}

        if split_by == "source":
            output_dict = {}
            for setting, emb in computed_results:
                output_dict.setdefault(setting.source, []).append(emb)
            return {source: torch.cat(embs, dim=-1) for source, embs in output_dict.items()}

        all_embs = [emb for _, emb in computed_results]
        return torch.cat(all_embs, dim=-1)

    def get_output_dim(self,
                       target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                       include_fields: Iterable[str] = None,
                       exclude_fields: Iterable[str] = None) -> int:
        """
        获取指定 target_sources 下拼接后 Embedding 的总维度大小。
        这是为了弥补移除 SideEmb 后，下游网络无法直接获取 self.embedding_size 的问题。
        """
        valid_settings = self.get_settings(
            target_sources=target_sources,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
        )
        return sum(setting.embedding_dim for setting in valid_settings)

    def get_settings(
            self,
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> tuple[EmbSetting, ...]:
        return tuple(
            self._filter_settings(
                target_sources=target_sources,
                include_fields=include_fields,
                exclude_fields=exclude_fields,
                split_by="none",
            )
        )
````

## File: src/rosen/emb/manager.py
````python
# ==========================================
# 2. 调度层 (Manager Layer) - 计算图统筹
# ==========================================
import gc
import json
import shutil
import ctypes
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union
import polars as pl

from rosen.emb.split import SplitConfig, SplitContext, create_split_strategy
from rosen.emb.schema import EmbSetting, EmbType, MultiSparseSetting, SparseEmbSetting
from rosen.core.enum_type import FeatureSource

from rosen.utils.sort import sort_parquet_inplace
from rosen.utils.shuffle import write_shuffled_parquet


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def collect_lazy(lf: pl.LazyFrame, streaming: bool = False) -> pl.DataFrame:
    if not streaming:
        return lf.collect()
    try:
        return lf.collect(streaming=True)
    except TypeError:
        return lf.collect(engine="streaming")
    except ValueError:
        return lf.collect()


def trim_process_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


VOCAB_DIR_NAME = "vocab"
VOCAB_VALUE_COL = "value"
VOCAB_ID_COL = "id"


class SchemaManager:
    ITEM_PROFILE_NAME = "item_profile.parquet"
    USER_PROFILE_NAME = "user_profile.parquet"
    SCHEMA_META_NAME = "feature_meta.json"
    WHOLE_DATA_NAME = "whole_dataframe.parquet"
    def __init__(self, settings_list: List[EmbSetting], schema_dir: str,
                 time_field=None,
                 label_fields: Union[str, Iterable[str]]=None,
                 domain_fields: Union[str, Iterable[str]]=None
                 ):
        self.settings = settings_list

        ## path & dir
        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.meta_filepath = self.schema_dir / self.SCHEMA_META_NAME

        ## field_name
        self.uid_field = next(
            (s.field_name for s in self.settings if s.source == FeatureSource.USER_ID),
            None
        )
        self.iid_field =next(
            (s.field_name for s in self.settings if s.source == FeatureSource.ITEM_ID),
            None
        )
        self.time_field = time_field
        self._label_fields = ensure_list(label_fields)
        self._domain_fields = ensure_list(domain_fields)

    @property
    def label_field(self):
        return next(iter(self._label_fields), None)
    @property
    def label_fields(self) -> tuple:
        return tuple(self._label_fields)
    @property
    def domain_field(self):
        return next(iter(self._domain_fields), None)
    @property
    def domain_fields(self) -> tuple:
        return tuple(self._domain_fields)


    def _parse_output_dir(self, output_dir=None):
        if output_dir is None:
            output_dir = self.schema_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir


    def prepare_data(self, lazy_df: pl.LazyFrame, output_dir: Union[Path, str]=None, redo=False):
        """
        不建议使用，全自动化流程：自动判断输入文件类型，自动构建计算图
        """
        output_parquet = self._parse_output_dir(output_dir) / self.WHOLE_DATA_NAME
        if self.meta_filepath.exists() and output_parquet.exists() and not redo:
            print("[Info] 检测到已存在元数据和目标文件，跳过处理直接加载 Schema。")
            self.load_schema()
            return pl.scan_parquet(output_parquet)

        print("[*] 正在执行计算图截断与中间态落盘，请稍候...")
        lazy_df = self._make_checkpoint(lazy_df)

        print("[*] 阶段一：构建 Fit 计算图...")
        fit_exprs = []
        for setting in self.settings:
            if not setting.is_fitted:
                fit_exprs.extend(setting.get_fit_exprs())

        if fit_exprs:
            print(f"[*] 启动单次全表扫描获取统计量，包含 {len(fit_exprs)} 个计算节点...")
            fit_result = lazy_df.select(fit_exprs).collect()
            for setting in self.settings:
                if not setting.is_fitted:
                    setting.parse_fit_result(fit_result)
            print("[+] 统计量计算与映射表构建完成。")
        else:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
        print("[*] 阶段二：构建 Transform 计算图...")
        transform_exprs = self._build_transform_exprs()
        print(f"[*] 启动流式转换计算，引擎正在将数据写入 {output_parquet} ...")
        lazy_df.with_columns(transform_exprs) \
            .sink_parquet(output_parquet)
        self.save_schema()
        print("[+] 预处理全流程结束！")
        return pl.scan_parquet(output_parquet)


    def fit(self, train_raw_lf: pl.LazyFrame, low_memory: bool = False):
        """
        阶段一：仅使用训练集拟合统计量和词表。

        :param train_raw_lf: Polars LazyFrame 格式的训练数据
        :param low_memory: 是否开启低内存模式。
                           - False (默认，推荐): 单次全表扫描，所有特征并行计算。速度极快，但内存开销较大。
                           - True (内存受限时): 逐个特征扫描。内存占用极低，但 I/O 放大，耗时成倍增加。
        """
        if self.meta_filepath.exists():
            self.load_schema()
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
            return
        unfitted_settings = [s for s in self.settings if not s.is_fitted]
        if not unfitted_settings:
            print("[+] 所有特征已就绪，跳过 Fit 扫描。")
            self.save_schema()
            return

        if low_memory:
            print(f"[*] 启动极低内存 Fit 模式，将对 {len(unfitted_settings)} 个特征进行逐列扫描 (耗时较长)...")
            for i, setting in enumerate(unfitted_settings):
                fit_exprs = setting.get_fit_exprs()
                if not fit_exprs:
                    continue
                print(f"    -> 正在拟合 [{i + 1}/{len(unfitted_settings)}]: {setting.field_name}")
                col_fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
                setting.parse_fit_result(col_fit_result)
                del col_fit_result # 显式释放哈希表内存
                trim_process_memory()
            print("[+] 逐列统计量与词表构建完成。")
        else:
            print(f"[*] 启动高性能并行 Fit 模式，单次扫描全表计算 {len(unfitted_settings)} 个统计量...")
            fit_exprs = [expr for s in unfitted_settings for expr in s.get_fit_exprs()]
            if fit_exprs:
                # 即使是并行，也建议开启 streaming=True，让 Polars 底层自己尽力优化和 Spilling
                fit_result = collect_lazy(train_raw_lf.select(fit_exprs), streaming=True)
                for setting in unfitted_settings:
                    setting.parse_fit_result(fit_result)
        self.save_schema()

    def _checkpoint_path(self, file_name: Union[str, Path]) -> Path:
        path = Path(file_name)
        if not path.is_absolute():
            path = self.schema_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def make_checkpoint(self, lf, file_name="custom_checkpoint.parquet", redo=True):
        path = self._checkpoint_path(file_name)
        if not path.exists() or redo:
            self._make_checkpoint(lf, file_name)
            return pl.scan_parquet(path)
        else:
            return pl.scan_parquet(path)

    def make_sorted_checkpoint(
        self,
        lf,
        file_name="custom_sorted_checkpoint.parquet",
        sort_by=None,
        descending=False,
        redo=True,
        memory_limit: str = "84GB",
    ):
        if sort_by is None:
            raise ValueError("sort_by is required for make_sorted_checkpoint.")
        path = self._checkpoint_path(file_name)
        if not path.exists() or redo:
            self._make_checkpoint(lf, file_name)
            sort_parquet_inplace(
                path,
                sort_by,
                descending=descending,
                memory_limit=memory_limit,
                temp_dir=path.parent / f"_{path.stem}_sort_tmp",
            )
        return pl.scan_parquet(path)

    def make_shuffled_checkpoint(
        self,
        lf,
        file_name="custom_shuffled_checkpoint.parquet",
        seed: int = 2026,
        bucket_count: int = 128,
        batch_size: int = 65536,
        redo: bool = True,
        exact: bool = False,
        mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
        compression: str = "zstd",
    ):
        path = self._checkpoint_path(file_name)
        if path.exists() and not redo:
            return pl.scan_parquet(path)

        source_path = path.parent / f"_{path.stem}_unshuffled.parquet"
        if source_path.exists():
            source_path.unlink()
        try:
            lf.sink_parquet(source_path, compression=compression)
            write_shuffled_parquet(
                [source_path],
                path,
                seed=seed,
                bucket_count=bucket_count,
                batch_size=batch_size,
                compression=compression,
                redo=True,
                exact=exact,
                mode=mode,
                temp_dir=path.parent / f"_{path.stem}_shuffle_tmp",
            )
        finally:
            if source_path.exists():
                source_path.unlink()
        return pl.scan_parquet(path)

    def transform(self, raw_lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        阶段二：根据已固化的 Schema 进行流式转换
        """
        transform_exprs = self._build_transform_exprs()
        return raw_lf.with_columns(transform_exprs)

    def transform_csv_partitioned(
        self,
        input_paths: Union[str, Path, Iterable[Union[str, Path]]],
        output_dir: Union[str, Path],
        uid_field: Optional[str] = None,
        n_parts: int = 100,
        batch_size: int = 50000,
        batches_per_read: int = 1,
        buffer_flush_rows: int = 500000,
        read_csv_kwargs: Optional[Mapping[str, Any]] = None,
        redo: bool = True,
    ) -> List[Path]:
        """
        分批读取 CSV，边 transform 边按原始 uid hash 分区落盘。

        输出结构:
            output_dir/p000/c0000.parquet
            output_dir/p000/c0001.parquet
            output_dir/p001/c0000.parquet
            ...
        """
        uid_field = uid_field or self.uid_field
        if uid_field is None:
            raise ValueError("uid_field is required when manager.uid_field is not set.")
        if n_parts <= 0:
            raise ValueError("n_parts must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batches_per_read <= 0:
            raise ValueError("batches_per_read must be positive.")
        if buffer_flush_rows <= 0:
            raise ValueError("buffer_flush_rows must be positive.")

        output_dir = Path(output_dir)
        if output_dir.exists() and redo:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        part_dirs = [output_dir / f"p{pid:03d}" for pid in range(n_parts)]
        for part_dir in part_dirs:
            part_dir.mkdir(parents=True, exist_ok=True)

        existing = [path for part_dir in part_dirs for path in part_dir.glob("*.parquet")]
        if existing and not redo:
            return part_dirs

        transform_exprs = self._build_transform_exprs()
        part_col = "__bb_part"
        buffers = {pid: [] for pid in range(n_parts)}
        buffer_rows = {pid: 0 for pid in range(n_parts)}
        file_counts = {pid: 0 for pid in range(n_parts)}

        def flush(pid: int):
            if not buffers[pid]:
                return
            out = pl.concat(buffers[pid], how="vertical")
            path = part_dirs[pid] / f"c{file_counts[pid]:04d}.parquet"
            out.write_parquet(path)
            file_counts[pid] += 1
            buffers[pid] = []
            buffer_rows[pid] = 0

        csv_kwargs = dict(read_csv_kwargs or {})
        csv_kwargs["batch_size"] = batch_size
        total_rows = 0

        for input_path in ensure_list(input_paths):
            input_path = Path(input_path)
            print(f"[*] transform csv partitioned: {input_path}")
            reader = pl.read_csv_batched(input_path, **csv_kwargs)

            while True:
                batches = reader.next_batches(batches_per_read)
                if batches is None:
                    break

                for raw_df in batches:
                    if raw_df.height == 0:
                        continue
                    if uid_field not in raw_df.columns:
                        raise ValueError(f"uid_field '{uid_field}' not found in {input_path}")
                    if part_col in raw_df.columns:
                        raise ValueError(f"reserved column '{part_col}' already exists in {input_path}")

                    df = (
                        raw_df.lazy()
                        .with_columns(
                            (pl.col(uid_field).hash(seed=0) % n_parts)
                            .cast(pl.UInt32)
                            .alias(part_col)
                        )
                        .with_columns(transform_exprs)
                        .pipe(collect_lazy, streaming=True)
                    )

                    for key, part_df in df.partition_by(
                        part_col,
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        pid = int(key[0])
                        part_df = part_df.drop(part_col)
                        buffers[pid].append(part_df)
                        buffer_rows[pid] += part_df.height
                        if buffer_rows[pid] >= buffer_flush_rows:
                            flush(pid)

                    total_rows += df.height
                    del raw_df, df

            print(f"    -> done, cumulative rows={total_rows:,}")

        for pid in range(n_parts):
            flush(pid)

        print(f"[+] partitioned transform done: rows={total_rows:,}, output={output_dir}")
        return part_dirs

    def extract_sequence_partitions(
        self,
        flat_part_dirs: Iterable[Union[str, Path]],
        output_dir: Union[str, Path],
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Mapping[str, str],
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1,
        select_fields: Optional[Iterable[str]] = None,
        redo: bool = True,
    ) -> List[Path]:
        from rosen.utils.sequential import extract_history_sequences

        output_dir = Path(output_dir)
        if output_dir.exists() and redo:
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[Path] = []
        fields = list(select_fields) if select_fields is not None else None

        for idx, part in enumerate(flat_part_dirs):
            part = Path(part)
            out_path = output_dir / f"{part.name}.parquet"
            outputs.append(out_path)
            if out_path.exists() and not redo:
                continue

            if part.is_dir():
                files = sorted(part.glob("*.parquet"))
            else:
                files = [part] if part.exists() else []

            if not files:
                continue

            seq_lf = extract_history_sequences(
                pl.scan_parquet([str(path) for path in files]),
                max_seq_len=max_seq_len,
                user_col=user_col,
                time_col=time_col,
                feature_mapping=dict(feature_mapping),
                seq_len_col=seq_len_col,
                label_col=label_col,
                positive_label=positive_label,
            )
            if fields is not None:
                seq_lf = seq_lf.select(fields)
            seq_lf.sink_parquet(out_path)
            print(f"    -> seq part {idx + 1}: {out_path}")

        return outputs

    def _build_transform_exprs(self) -> List[pl.Expr]:
        transform_exprs: List[pl.Expr] = []
        seen_output_names = set()
        for setting in self.settings:
            if setting.emb_type == EmbType.UNKNOWN:
                continue
            expr = setting.get_transform_expr()
            expr_list = expr if isinstance(expr, list) else [expr]
            for item in expr_list:
                try:
                    output_name = item.meta.output_name()
                except Exception:
                    output_name = None
                if output_name is not None and output_name in seen_output_names:
                    continue
                if output_name is not None:
                    seen_output_names.add(output_name)
                transform_exprs.append(item)
        return transform_exprs

    def _make_checkpoint(self, lazy_df, file_name="temp_checkpoint.parquet"):
        temp_checkpoint_path = self._checkpoint_path(file_name)
        lazy_df.sink_parquet(temp_checkpoint_path)
        lazy_df = pl.scan_parquet(temp_checkpoint_path)
        return lazy_df

    # NOTE 不推荐使用
    def generate_profiles(self, lazy_df: pl.LazyFrame, output_dir: str=None, redo=False):
        """
        利用 FeatureSource 统筹提取并保存静态画像表 (Profile)
        """
        item_profile_path = self._parse_output_dir(output_dir) / self.ITEM_PROFILE_NAME
        user_profile_path = self._parse_output_dir(output_dir) / self.USER_PROFILE_NAME

        print("[*] 阶段三：开始提取静态特征查找表 (Profiles)...")
        # 1. Item Profile
        item_features = [setting.field_name for setting in self.settings
                         if setting.source == FeatureSource.ITEM]
        if self.iid_field and (not item_profile_path.exists() or redo):
            # 确保主键在提取列表中
            if self.iid_field not in item_features:
                item_features.insert(0, self.iid_field)

            print(f"[*] 正在提取 Item 画像，包含特征: {item_features}")
            (
                lazy_df
                .select(item_features)
                .drop_nulls(subset=[self.iid_field])  # 剔除由于 left join 产生的空物品
                .unique(subset=[self.iid_field])  # 按物品 ID 去重
                .sort(self.iid_field)  # 【极其关键】按 ID 从小到大排序！
                .sink_parquet(item_profile_path)
            )
        # 2. 提取 User Profile (同理)
        user_features = [
            setting.field_name
            for setting in self.settings
            if setting.source == FeatureSource.USER
        ]

        if self.uid_field and (not user_profile_path.exists() or redo):
            if self.uid_field not in user_features:
                user_features.insert(0, self.uid_field)

            print(f"[*] 正在提取 User 画像，包含特征: {user_features}")
            (
                lazy_df
                .select(user_features)
                .drop_nulls(subset=[self.uid_field])
                .unique(subset=[self.uid_field])
                .sort(self.uid_field)
                .sink_parquet(user_profile_path)
            )
        print("[+] 静态查找表提取完成！")

    def split_dataset(
        self,
        lf: pl.LazyFrame,
        strategy: Literal['loo', 'time', 'sequential_ratio', 'random_ratio'] = "random_ratio",
        output_dir: str = None,
        redo: bool = False,
        config: Optional[SplitConfig] = None,
        **kwargs,
    ):
        output_dir = self._parse_output_dir(output_dir)
        split_ctx = SplitContext(
            uid_field=self.uid_field,
            iid_field=self.iid_field,
            time_field=self.time_field,
            checkpoint_fn=self._make_checkpoint  # 把方法当做变量传递进去 (Callback)
        )
        splitter = create_split_strategy(strategy=strategy, context=split_ctx, config=config, **kwargs)
        return splitter.split(lf, output_dir, redo)

    def save_as_dataset(self, train_lf, valid_lf=None, test_lf=None, output_dir: str = None, redo: bool = False):
        """
        如果预设的split_dataset不和你的想法，那么把三个lf传过来，这边直接保存。利好Kuairand等预设test的数据集
        """
        output_dir = self._parse_output_dir(output_dir)
        train_path = output_dir / "train.parquet"
        valid_path = output_dir / "valid.parquet"
        test_path = output_dir / "test.parquet"
        split_paths = str(train_path), str(valid_path), str(test_path)
        if not redo and (train_path.exists() or valid_path.exists() or test_path.exists()):
            return split_paths
        gc.collect()
        for lf, save_path, name in zip([train_lf, valid_lf, test_lf], split_paths, ["Train", "Valid", "Test"]):
            if lf is not None:
                print(f"    -> 正在写入 {name} 集合到 {save_path}")
                lf.sink_parquet(str(save_path),)
        print("[+] 所有数据落盘完成！")
        return split_paths

    def save_schema(self):
        vocab_dir = self.meta_filepath.parent / VOCAB_DIR_NAME
        vocab_dir.mkdir(parents=True, exist_ok=True)
        meta_data = []
        for setting in self.settings:
            data = setting.to_dict()
            for state in (data, data.get("element_setting_state")):
                if not isinstance(state, dict) or not isinstance(state.get("vocab"), dict) or not state["vocab"]:
                    continue
                ref = f"{VOCAB_DIR_NAME}/{state['field_name']}.parquet"
                vocab = state["vocab"]
                rows = sorted(vocab.items(), key=lambda item: int(item[1]))
                pl.DataFrame(
                    {
                        VOCAB_VALUE_COL: [str(key) for key, _ in rows],
                        VOCAB_ID_COL: [int(value) for _, value in rows],
                    }
                ).write_parquet(self.meta_filepath.parent / ref)
                state["vocab"] = ref
            meta_data.append(data)
        with open(self.meta_filepath, 'w') as f:
            json.dump(meta_data, f, indent=4)

    def load_schema(self):
        if not self.meta_filepath.exists():
            print(f"[Warning] 找不到元数据文件 {self.meta_filepath}，无法加载 schema。")
            return

        with open(self.meta_filepath, 'r', encoding='utf-8') as f:
            meta_data_list = json.load(f)

        for data in meta_data_list:
            for state in (data, data.get("element_setting_state")):
                if not isinstance(state, dict) or not isinstance(state.get("vocab"), str):
                    continue
                vocab_df = pl.read_parquet(
                    self.meta_filepath.parent / state["vocab"],
                    columns=[VOCAB_VALUE_COL, VOCAB_ID_COL],
                )
                state["vocab"] = {
                    str(row[VOCAB_VALUE_COL]): int(row[VOCAB_ID_COL])
                    for row in vocab_df.iter_rows(named=True)
                }

        meta_dict = {item["field_name"]: item for item in meta_data_list}
        for setting in self.settings:
            if setting.field_name in meta_dict:
                saved_state = meta_dict[setting.field_name]
                saved_type = saved_state.get("type")
                if saved_type not in setting.compatible_type_names:
                    raise ValueError(
                        f"特征 {setting.field_name} 的类型不匹配: "
                        f"期望 {sorted(setting.compatible_type_names)}, JSON 中为 {saved_type}"
                    )

                setting.load_state(saved_state)
                print(f"[-] 成功恢复特征状态: {setting.field_name}")
            else:
                print(f"[Warning] 特征 {setting.field_name} 在 JSON 元数据中未找到，将保持默认/未拟合状态。")

    def source2emb_dim(self, *sources: FeatureSource):
        emb_dim = 0
        for setting in self.settings:
            if setting.source in sources:
                emb_dim += setting.embedding_dim
        return emb_dim

    def get_setting(self, field_name):
        for setting in self.settings:
            if setting.field_name == field_name:
                return setting
        return None

    def fields(self):
        fields = []
        for setting in self.settings:
            for field_name in setting.get_output_field_names():
                if field_name not in fields:
                    fields.append(field_name)

        for ctx_field in (self.time_field, *self.label_fields, *self.domain_fields):
            if ctx_field is not None and ctx_field not in fields:
                fields.append(ctx_field)
        return fields


# ==========================================
# 3. 使用示例 (假设使用 KuaiRand)
# ==========================================
if __name__ == "__main__":
    # 假设特征注册
    user_id_setting = SparseEmbSetting("user_id", FeatureSource.USER)

    # 高级序列特征映射：无论原数据是 "1,5,6" 还是 ["1", "5", "6"]，瞬间处理完毕
    tags = MultiSparseSetting(
        field_name="tag",
        source=FeatureSource.ITEM,
        is_string_format=True,
        separator=","
    )

    manager = SchemaManager([user_id_setting, tags])

    from rosen.datasets.kuairand import KuaiRand
    item_lf = pl.scan_csv(KuaiRand.VIDEO_FEATURES)
    inter_lf = pl.scan_csv(KuaiRand.STD_LOG_FORMER_DATA)
    user_lf = pl.scan_csv(KuaiRand.USER_FEATURES)

    whole_lf = inter_lf.join(item_lf, on="video_id", how="left")
    whole_lf = whole_lf.join(user_lf, on="user_id", how="left")
    # 只需要喂入路径，不用 pd.read_csv。即使 kuairand_log.csv 有 50GB 也能在普通笔记本上跑完
    # manager.prepare_data("kuairand_log.csv", "kuairand_processed.parquet")
    manager.prepare_data(whole_lf, "kuairand_processed.parquet", redo=True)
    print("架构编译成功，可供调用。")

    import pandas as pd
    df = pd.read_parquet("kuairand_processed.parquet")
    print(df.head(3).to_string())
````

## File: src/rosen/emb/split.py
````python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union

import polars as pl


@dataclass(frozen=True)
class SplitContext:
    uid_field: Optional[str]
    iid_field: Optional[str]
    time_field: Optional[str]
    checkpoint_fn: Callable[[pl.LazyFrame], pl.LazyFrame]


@dataclass(frozen=True)
class LooConfig:
    k_core: int = 3


@dataclass(frozen=True)
class SequentialRatioConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1


@dataclass(frozen=True)
class TimeSplitConfig:
    valid_start: Any
    test_start: Any


@dataclass(frozen=True)
class RandomRatioConfig:
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    group_by: Optional[str] = None


SplitConfig = Union[LooConfig, SequentialRatioConfig, TimeSplitConfig, RandomRatioConfig]
ConfigT = TypeVar("ConfigT")


class BaseSplitStrategy(Generic[ConfigT], ABC):
    config_type: Type[ConfigT]

    def __init__(self, context: SplitContext, config: ConfigT):
        self.ctx = context
        self.config = config

    @abstractmethod
    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        pass


class LooSplitStrategy(BaseSplitStrategy[LooConfig]):
    config_type = LooConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool):
        print(f"[*] 正在执行 LOO 切分，K-core={self.config.k_core}...")
        if self.ctx.time_field is None:
            raise RuntimeError("执行 loo 切分必须在 SplitContext 中指定 time_field！")

        processed_lf = lf.with_columns(
            [
                pl.count(self.ctx.iid_field).over(self.ctx.uid_field).alias("user_seq_len"),
                pl.col(self.ctx.time_field)
                .rank(descending=True, method="ordinal")
                .over(self.ctx.uid_field)
                .alias("reverse_rank"),
            ]
        ).filter(pl.col("user_seq_len") >= self.config.k_core)

        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        drop_cols = ["user_seq_len", "reverse_rank"]
        train_lf = checkpoint_lf.filter(pl.col("reverse_rank") >= 3).select(pl.all().exclude(drop_cols))
        valid_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 2).select(pl.all().exclude(drop_cols))
        test_lf = checkpoint_lf.filter(pl.col("reverse_rank") == 1).select(pl.all().exclude(drop_cols))
        return train_lf, valid_lf, test_lf


class SequentialRatioStrategy(BaseSplitStrategy[SequentialRatioConfig]):
    config_type = SequentialRatioConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        if self.ctx.time_field is None:
            raise RuntimeError("执行 sequential_ratio 切分必须在 SplitContext 中指定 time_field！")

        train_ratio = self.config.train_ratio
        valid_ratio = self.config.valid_ratio
        test_ratio = 1 - train_ratio - valid_ratio
        print(
            f"[*] 正在执行顺序时序比例 (Sequential Ratio) 切分，比例: "
            f"{train_ratio}:{valid_ratio}:{test_ratio:.2f}..."
        )

        time_col = self.ctx.time_field
        time_count_lf = (
            lf.group_by(time_col)
            .len()
            .sort(time_col)
            .with_columns(pl.col("len").cum_sum().alias("cum_rows"))
        )
        total_rows = time_count_lf.select(pl.col("len").sum().alias("total_rows")).collect(engine="streaming").item()
        train_cut = int(total_rows * train_ratio)
        valid_cut = train_cut + int(total_rows * valid_ratio)

        boundary_df = (
            time_count_lf.select(
                [
                    pl.col(time_col).filter(pl.col("cum_rows") >= train_cut).first().alias("valid_start"),
                    pl.col(time_col).filter(pl.col("cum_rows") >= valid_cut).first().alias("test_start"),
                ]
            ).collect(engine="streaming")
        )
        valid_start = boundary_df["valid_start"][0]
        test_start = boundary_df["test_start"][0]

        train_lf = lf.filter(pl.col(time_col) < valid_start)
        valid_lf = lf.filter((pl.col(time_col) >= valid_start) & (pl.col(time_col) < test_start))
        test_lf = lf.filter(pl.col(time_col) >= test_start)
        return train_lf, valid_lf, test_lf


class TimeSplitStrategy(BaseSplitStrategy[TimeSplitConfig]):
    config_type = TimeSplitConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        if self.ctx.time_field is None:
            raise RuntimeError("执行 time 切分必须在 SplitContext 中指定 time_field！")

        print(
            f"[*] 正在执行绝对时间阈值 (Time) 切分，"
            f"train_end={self.config.valid_start}, valid_end={self.config.test_start}..."
        )
        time_expr = pl.col(self.ctx.time_field)
        train_lf = lf.filter(time_expr < self.config.valid_start)
        valid_lf = lf.filter((time_expr >= self.config.valid_start) & (time_expr < self.config.test_start))
        test_lf = lf.filter(time_expr >= self.config.test_start)
        return train_lf, valid_lf, test_lf


class RandomRatioStrategy(BaseSplitStrategy[RandomRatioConfig]):
    config_type = RandomRatioConfig

    def split(self, lf: pl.LazyFrame, output_dir: Path, redo: bool) -> tuple:
        train_ratio = self.config.train_ratio
        valid_ratio = self.config.valid_ratio
        group_by = self.config.group_by

        group_msg = f"按 '{group_by}' 分组" if group_by else "全局"
        print(
            f"[*] 正在执行{group_msg}随机比例 (Random Ratio) 切分，"
            f"比例: {train_ratio}:{valid_ratio}:{1 - train_ratio - valid_ratio:.2f}..."
        )

        len_expr = pl.len().over(group_by) if group_by else pl.len()
        row_idx_expr = pl.int_range(0, pl.len()).shuffle(seed=42)
        if group_by:
            row_idx_expr = row_idx_expr.over(group_by)

        train_idx_expr = (len_expr * train_ratio).cast(pl.Int64)
        valid_idx_expr = train_idx_expr + (len_expr * valid_ratio).cast(pl.Int64)
        processed_lf = lf.with_columns(
            row_idx_expr.alias("row_idx"),
            train_idx_expr.alias("train_idx"),
            valid_idx_expr.alias("valid_idx"),
        )

        checkpoint_lf = self.ctx.checkpoint_fn(processed_lf)
        temp_cols = ["row_idx", "train_idx", "valid_idx"]
        train_lf = checkpoint_lf.filter(pl.col("row_idx") < pl.col("train_idx")).drop(temp_cols)
        valid_lf = checkpoint_lf.filter(
            (pl.col("row_idx") >= pl.col("train_idx")) & (pl.col("row_idx") < pl.col("valid_idx"))
        ).drop(temp_cols)
        test_lf = checkpoint_lf.filter(pl.col("row_idx") >= pl.col("valid_idx")).drop(temp_cols)
        return train_lf, valid_lf, test_lf


SPLIT_STRATEGIES: Dict[str, Type[BaseSplitStrategy]] = {
    "loo": LooSplitStrategy,
    "time": TimeSplitStrategy,
    "sequential_ratio": SequentialRatioStrategy,
    "random_ratio": RandomRatioStrategy,
}

SPLIT_CONFIG_TYPES: Dict[str, Type[Any]] = {
    "loo": LooConfig,
    "time": TimeSplitConfig,
    "sequential_ratio": SequentialRatioConfig,
    "random_ratio": RandomRatioConfig,
}


def build_split_config(strategy: str, **kwargs) -> SplitConfig:
    config_cls = SPLIT_CONFIG_TYPES.get(strategy)
    if config_cls is None:
        raise ValueError(f"未知的切分策略: {strategy}")

    allowed_fields = {item.name for item in fields(config_cls)}
    unknown_fields = sorted(set(kwargs) - allowed_fields)
    if unknown_fields:
        raise TypeError(
            f"切分策略 '{strategy}' 不接受参数: {unknown_fields}。"
            f"允许参数为: {sorted(allowed_fields)}"
        )

    return config_cls(**kwargs)


def create_split_strategy(
    strategy: str,
    context: SplitContext,
    config: Optional[SplitConfig] = None,
    **kwargs,
) -> BaseSplitStrategy:
    strategy_cls = SPLIT_STRATEGIES.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"未知的切分策略: {strategy}")

    if config is None:
        config = build_split_config(strategy, **kwargs)
    elif kwargs:
        raise TypeError("请在 split_dataset 中二选一：传入显式 config，或传入兼容 kwargs。")

    if not isinstance(config, strategy_cls.config_type):
        raise TypeError(
            f"切分策略 '{strategy}' 期望配置类型为 {strategy_cls.config_type.__name__}，"
            f"实际得到 {type(config).__name__}"
        )

    return strategy_cls(context=context, config=config)
````

## File: src/rosen/evaluate/__init__.py
````python
"""Evaluation metrics and evaluators."""
````

## File: src/rosen/evaluate/evaluator.py
````python
import json
from typing import Iterable, Union, Literal

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
import torch


class BaseEvaluator:
    VALID_METRICS = ["auc", "logloss", "gauc", "hr", "ndcg", "recall"]
    def __init__(self, metrics):
        # 统一转小写并过滤非法指标
        self.metrics = [m.lower() for m in metrics]
        for m in self.metrics:
            base_m = m.split('@')[0] if '@' in m else m
            assert base_m in self.VALID_METRICS, f"不支持的指标: {m}"
        self.clear()

    def collect(self, batch_users, batch_targets, batch_preds):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


import numpy as np
from rosen.evaluate.metrics import METRIC_REGISTRY

class PointWiseEvaluator(BaseEvaluator):
    def __init__(self, metrics):
        super().__init__(metrics)

    def collect(self, batch_users, batch_targets, batch_preds):
        # 粗暴地把当前 Batch 的数据怼进大列表
        self.all_users.extend(batch_users.tolist())
        self.all_targets.extend(batch_targets.tolist())
        self.all_preds.extend(batch_preds.tolist())

    def summary(self):
        results = {}
        if not self.all_targets:
            return results

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

        # 遍历指标，通过注册表调用静态计算方法
        for m in self.metrics:
            calc_class = METRIC_REGISTRY[m]
            results[m] = calc_class.calculate(y_true=y_true, y_pred=y_pred, users=self.all_users)

        return results

    def clear(self):
        self.all_users = []
        self.all_targets = []
        self.all_preds = []


class TopKEvaluator(BaseEvaluator):
    def __init__(self, metrics, history_dict):
        self.history_dict = history_dict
        super().__init__(metrics)

        # 解析 K 值 (例如将 'hit@5', 'ndcg@10' 解析出 k_list = [5, 10])
        self.k_list = sorted(list(set([int(m.split('@')[1]) for m in self.metrics])))
        self.max_k = max(self.k_list) if self.k_list else 0

    def collect(self, batch_users, batch_targets, batch_scores):
        device = batch_scores.device
        batch_size = batch_users.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        target_original_scores = batch_scores[batch_indices, batch_targets].clone()
        # 1. 瞬间 Mask 历史物品
        rows, cols = [], []
        for i, uid in enumerate(batch_users.tolist()):
            hist = self.history_dict.get(uid, [])
            rows.extend([i] * len(hist))
            cols.extend(hist)
        if rows:
            batch_scores[rows, cols] = -float('inf')
        batch_scores[batch_indices, batch_targets] = target_original_scores # 将目标物品的分数还原
        # 2. 取 Top-Max_K 并生成命中矩阵 [Batch_size, max_k]
        _, topk_indices = torch.topk(batch_scores, self.max_k, dim=-1)
        hits = (topk_indices == batch_targets.unsqueeze(1)).float()

        # 3. 现结现清：调用指标类算总和并累加
        for k in self.k_list:
            hits_k = hits[:, :k]
            for m in self.metrics:
                metric_name, metric_k = m.split('@')
                if int(metric_k) == k:
                    calc_class = METRIC_REGISTRY[metric_name]
                    # 调用静态方法获取当前 Batch 的总得分
                    batch_sum = calc_class.calculate(hits_k=hits_k, k=k)
                    self.metric_sums[m] += batch_sum

        self.total_users += batch_size

    def summary(self):
        results = {}
        if self.total_users == 0:
            return results
        for m in self.metrics:
            results[m] = self.metric_sums[m] / self.total_users
        return results

    def clear(self):
        self.metric_sums = {m: 0.0 for m in self.metrics}
        self.total_users = 0

import polars as pl
ValidMetrics = Literal["auc", "logloss", "gauc", "HR@*", "NDCG@*"]
class Evaluator:
    def __init__(self, *metrics: ValidMetrics, history_dict=None):
        """
        metrics: list of str, 例如 ['auc', 'logloss', 'hit@10', 'ndcg@20']
        history_dict: dict, TopK 评估必须传入的静态历史黑名单
        """
        self.metrics = [m.lower() for m in metrics]
        # 1. 自动解析和分流指标
        self.point_metrics = [m for m in self.metrics if '@' not in m]
        self.topk_metrics = [m for m in self.metrics if '@' in m]

        # 2. 实例化底层 Evaluator
        self.point_evaluator = None
        self.topk_evaluator = None

        if self.point_metrics:
            self.point_evaluator = PointWiseEvaluator(self.point_metrics)

        if self.topk_metrics:
            assert history_dict is not None, "错误：计算 TopK 指标必须传入 history_dict 以进行历史剔除！"
            self.topk_evaluator = TopKEvaluator(self.topk_metrics, history_dict)

    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None):
        if self.point_evaluator:
            self.collect_pointwise(batch_users, batch_targets, batch_preds_1d)
        if self.topk_evaluator:
            self.collect_topk(batch_users, batch_targets, batch_scores_2d)

    def collect_pointwise(self, batch_users, batch_targets, batch_preds_1d):
        """
        收集 CTR 等 Point-wise 任务的预测结果。
        - batch_preds_1d: shape [Batch_Size]，通常为 0~1 的概率值。
        """
        if self.point_evaluator is None:
            raise RuntimeError("未初始化 PointWise 评估器。请检查是否传入了 auc, logloss 等指标。")
        self.point_evaluator.collect(batch_users, batch_targets, batch_preds_1d)

    def collect_topk(self, batch_users, batch_targets, batch_scores_2d):
        """
        收集序列推荐/召回等 Top-K 任务的全排序得分。
        - batch_scores_2d: shape [Batch_Size, Num_Items]，通常为 Raw Logits。
        """
        if self.topk_evaluator is None:
            raise RuntimeError("未初始化 TopK 评估器。请检查是否传入了 hit@k, ndcg@k 等指标。")
        self.topk_evaluator.collect(batch_users, batch_targets, batch_scores_2d)

    def summary(self):
        """
        Epoch 结束时调用，合并所有底层 Evaluator 的成绩单并保存
        epoch/step: 可选，用于记录当前是第几轮或第几步
        """
        final_results = {}

        if self.point_evaluator:
            final_results.update(self.point_evaluator.summary())

        if self.topk_evaluator:
            final_results.update(self.topk_evaluator.summary())

        return final_results

    def clear(self):
        """
        清空缓存和累加器，准备下一个 Epoch
        """
        if self.point_evaluator:
            self.point_evaluator.clear()

        if self.topk_evaluator:
            self.topk_evaluator.clear()

class EvaluatorDecorator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
    def collect_pointwise(self, batch_users, batch_targets, batch_preds_1d):
        self.evaluator.collect_pointwise(batch_users, batch_targets, batch_preds_1d)
    def collect_topk(self, batch_users, batch_targets, batch_scores_2d):
        self.evaluator.collect_topk(batch_users, batch_targets, batch_scores_2d)
    def collect(self, batch_users, batch_targets, batch_preds_1d=None, batch_scores_2d=None):
        self.evaluator.collect(batch_users, batch_targets, batch_preds_1d, batch_scores_2d)
    def summary(self, epoch=None, step=None):
        return self.evaluator.summary()
    def clear(self):
        self.evaluator.clear()

import os
from datetime import datetime
class LogDecorator(EvaluatorDecorator):
    def __init__(self, evaluator, save_path=None, title=None):
        """
        evaluator: 原始的 Evaluator 实例
        save_path: 日志保存路径
        """
        super().__init__(evaluator)
        self.save_path = save_path
        self.title = title
        self._is_first_summary = True

        if self.save_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)

    def summary(self, epoch=None, step=None):
        """
        拦截原有的 summary 方法，在拿到结果后增加纯文本写入的逻辑
        同时扩展了接受 epoch 和 step 的能力
        """
        # 1. 调用原 Evaluator 的 summary 获取干净的指标字典
        final_results = self.evaluator.summary()

        # 2. 拦截并执行纯文本日志写入逻辑
        if self.save_path:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(self.save_path, 'a', encoding='utf-8') as f:
                # 首次写入打分割线
                if self._is_first_summary:
                    title = self.title or ""
                    f.write(f"\n========== {title}实验进程开始 | 首次评估时间: {current_time} ==========\n")
                    self._is_first_summary = False

                # 拼接易读字符串
                log_parts = [f"[{current_time}]"]
                if epoch is not None:
                    log_parts.append(f"Epoch: {epoch:02d}")
                if step is not None:
                    log_parts.append(f"Step: {step}")

                # 遍历原始指标写入
                for metric_name, value in final_results.items():
                    if isinstance(value, float):
                        log_parts.append(f"{metric_name}: {value:.4f}")
                    else:
                        log_parts.append(f"{metric_name}: {value}")

                f.write(" | ".join(log_parts) + '\n')

        return final_results

if __name__ == '__main__':
    ...
````

## File: src/rosen/evaluate/manager.py
````python
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import polars as pl


class EvaluatorFilter(ABC):
    @abstractmethod
    def __call__(self, inter):
        raise NotImplementedError


class AllowAllFilter(EvaluatorFilter):
    def __call__(self, inter):
        return True


class DomainFilter(EvaluatorFilter):
    def __init__(self, field_name: str, domain_id: int):
        self.field_name = field_name
        self.domain_id = domain_id

    def __call__(self, inter):
        return inter[self.field_name] == self.domain_id

class EvaluatorManager:
    def __init__(self, log_path=None, title="untitled"):
        self.registry = {}
        self.log_path = log_path
        self.title = title
        self._is_first_summary = True

        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def register(self, name, evaluator, filter_fn=None):
        """
        注册评估器。
        filter_fn: 接收 batch_inter，返回布尔掩码
        """
        if filter_fn is None:
            filter_fn = AllowAllFilter()

        self.registry[name] = {
            "evaluator": evaluator,
            "filter_fn": filter_fn,
        }

    def collect(self, batch_users, batch_targets, batch_inter, batch_preds_1d=None, batch_scores_2d=None, ):
        """
        前4个参数完全对齐你的基础 Evaluator。
        唯一多出的 batch_inter 仅作为计算 mask 的上下文，不参与任何解包。
        """
        for name, config in self.registry.items():
            evaluator = config["evaluator"]
            filter_fn = config["filter_fn"]

            # 1. 仅把 inter 交给 filter_fn 算掩码，绝不从 inter 里取真实数据
            mask = filter_fn(batch_inter)

            # 2. 全量不过滤的情况
            if isinstance(mask, bool) and mask is True:
                evaluator.collect(batch_users, batch_targets, batch_preds_1d, batch_scores_2d)
                continue

            # 提前拦截空掩码
            if isinstance(mask, bool) and mask is False:
                continue
            if hasattr(mask, 'any') and not mask.any():
                continue

            # 3. 对传入的张量进行原生的布尔切片
            e_users = batch_users[mask]
            e_targets = batch_targets[mask]
            e_preds_1d = batch_preds_1d[mask] if batch_preds_1d is not None else None
            e_scores_2d = batch_scores_2d[mask] if batch_scores_2d is not None else None

            # 4. 完美对齐调用底层的 Evaluator
            evaluator.collect(e_users, e_targets, e_preds_1d, e_scores_2d)

    def _safe_summary(self, evaluator, epoch=None, step=None):
        try:
            return evaluator.summary(epoch=epoch, step=step)
        except TypeError:
            return evaluator.summary()

    def summary(self, epoch=None, step=None):
        """
        返回每个 evaluator 的完整指标：
        {
            "overall": {"auc": 0.73, "logloss": 0.41},
            "domain_a": {"gauc": 0.69},
        }
        """
        summary_by_name = {}
        log_metrics = {}
        for name, config in self.registry.items():
            res = self._safe_summary(config["evaluator"], epoch=epoch, step=step)
            summary_by_name[name] = dict(res)
            for metric_name, metric_value in res.items():
                log_metrics[f"{name}.{metric_name}"] = metric_value

        if self.log_path:
            self._write_polars_log(log_metrics, epoch, step)

        return summary_by_name

    def _write_polars_log(self, metrics_dict, epoch, step):
        """内部方法：专门负责将指标转成 Polars 字符串并追加到文件"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将字典转为纵向结构，方便阅读
        df = pl.DataFrame({
            "Metric": list(metrics_dict.keys()),
            "Value": list(metrics_dict.values())
        })

        # 临时配置 Polars 渲染参数：
        # tbl_rows=100：防止指标太多中间被折叠成 "..."
        # tbl_hide_dataframe_shape=True：隐藏底部的 (shape: 10, 2) 这种无关信息
        with pl.Config(tbl_rows=100, tbl_hide_dataframe_shape=True):
            table_str = str(df)

        # 写入文件
        with open(self.log_path, 'a', encoding='utf-8') as f:
            # 首次记录打印实验 Header
            if self._is_first_summary:
                f.write(f"\n========== 🚀 [{self.title}] 开始记录 | {current_time} ==========\n")
                self._is_first_summary = False

            # 打印当前 Step/Epoch 的信息头
            header_parts = [f"\n[{current_time}]"]
            if epoch is not None: header_parts.append(f"Epoch: {epoch:02d}")
            if step is not None: header_parts.append(f"Step: {step}")

            f.write(" | ".join(header_parts) + "\n")

            # 写入 Polars 格式化好的表格
            f.write(table_str + "\n")

    def clear(self):
        for config in self.registry.values():
            config["evaluator"].clear()
````

## File: src/rosen/evaluate/metrics.py
````python
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

# ==================== 指标基类 ====================
class BaseMetric:
    @classmethod
    def calculate(cls, *args, **kwargs):
        raise NotImplementedError

# ==================== PointWise 指标簇 ====================
class AUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.0

class LogLoss(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        return log_loss(y_true, y_pred)

class GAUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, users, **kwargs):
        user_data = {}
        for u, t, p in zip(users, y_true, y_pred):
            user_data.setdefault(u, {'t': [], 'p': []})
            user_data[u]['t'].append(t)
            user_data[u]['p'].append(p)

        total_weight, weighted_auc = 0.0, 0.0
        for data in user_data.values():
            t_arr, p_arr = np.array(data['t']), np.array(data['p'])
            weight = len(t_arr)
            if len(set(t_arr)) > 1:
                try:
                    weighted_auc += roc_auc_score(t_arr, p_arr) * weight
                    total_weight += weight
                except ValueError:
                    continue
        return weighted_auc / total_weight if total_weight > 0 else 0.0

# ==================== TopK 指标簇 ====================
class HR(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, **kwargs):
        # hits_k shape: [Batch_size, K]
        return hits_k.sum().item()

class NDCG(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, k, **kwargs):
        # 针对 Leave-One-Out 的极速 NDCG 计算
        device = hits_k.device
        ranks = torch.arange(1, k + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)
        return (hits_k * dcg_weights).sum().item()

# 注册表：将字符串映射到对应的计算类
METRIC_REGISTRY = {
    "auc": AUC, "logloss": LogLoss, "gauc": GAUC,
    "hr": HR, "ndcg": NDCG
}
````

## File: src/rosen/experiment/__init__.py
````python
from pathlib import Path
from .config import ROOT_DIR, WORKSPACE

def ignore_future_warning():
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

def change_root_workdir():
    import os
    print("切换到: ", ROOT_DIR)
    os.chdir(ROOT_DIR)

def set_all():
    ignore_future_warning()
    change_root_workdir()

if __name__ == '__main__':
    print(ROOT_DIR)
````

## File: src/rosen/experiment/config.py
````python
"""Experiment paths belong to the calling project, not the installed package."""
import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("ROSEN_ROOT_DIR", Path.cwd())).expanduser().resolve()
WORKSPACE = ROOT_DIR / "workspace"
````

## File: src/rosen/experiment/param.py
````python
import argparse
import ast
import dataclasses
from dataclasses import dataclass, fields, field, asdict
from typing import get_args, get_origin, Literal, get_type_hints
from  .config import WORKSPACE, ROOT_DIR
@dataclass
class ConfigBase:
    experiment_name: str = "untitled" # 提示这次实验的名字
    dataset_name: str = "unknown" # 影响workdir的设置
    seed: int = 2026
    device: str = "cpu"
    max_epochs: int = 100
    ckpt_dir: str = ""

    extras: dict = field(default_factory=dict)

    @property
    def preset_workspace(self):
        return WORKSPACE

    @property
    def preset_dataset_dir(self):
        return self.preset_workspace / self.dataset_name

    @property
    def preset_output_dir(self):
        return self.preset_workspace / self.dataset_name / "experiments" / self.experiment_name

    def __getattr__(self, name):
        if name in self.extras:
            return self.extras[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self):
        config_dict = asdict(self)
        lines = [f"\n🚀 [Active Config State]: <{self.__class__.__name__}>"]
        lines.append("-" * 50)
        max_key_len = max([len(str(k)) for k in config_dict.keys()] + [10])

        for key, value in config_dict.items():
            if isinstance(value, type):
                display_val = f"<class '{value.__name__}'>"
            elif callable(value):
                display_val = f"<callable '{getattr(value, '__name__', str(value))}'>"
            elif isinstance(value, dict) and key == "extras":
                if not value:
                    display_val = "{}"
                else:
                    display_val = "{" + ", ".join(f"'{k}': {repr(v)}" for k, v in value.items()) + "}"
            else:
                display_val = repr(value)  # 使用 repr 可以让字符串带上引号，更严谨
            lines.append(f"  {key:<{max_key_len}} = {display_val}")
        lines.append("-" * 50)
        return "\n".join(lines)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        annotations = getattr(cls, '__annotations__', {})
        missing_annotations = []
        for key, value in cls.__dict__.items():
            if key.startswith('_') or callable(value) or isinstance(value, (property, classmethod, staticmethod)):
                continue
            if key not in annotations:
                missing_annotations.append(key)
        if missing_annotations:
            missing_str = ", ".join(f"'{f}'" for f in missing_annotations)
            raise TypeError(
                f"🚨 拦截: 数据类 '{cls.__name__}' 存在未加类型注解的属性！\n"
                f"缺失注解的字段共 {len(missing_annotations)} 个: [{missing_str}]\n"
                f"👉 修复建议: 请修改为 '字段名: 类型 = ...'，如无特定类型请使用 'Any'。"
            )

def seed_everything(seed=42):
    import os
    import numpy as np
    import random
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""
调用build方法可以从任何地方获取参数
"""


class ParamManager:
    def __init__(self, config_class):
        self.config_class = config_class
        self.resolved_type_hints = get_type_hints(config_class)

        # 1. 基础类型转换注册表：消除对 bool 等基本类型的 if-else
        self.type_parsers = {
            bool: lambda x: str(x).lower() in ('true', '1', 'yes', 'y', 't'),
            list: lambda x: x.split(','),
        }

        # 2. 实例工厂注册表：{ 字段名: { 字符串key: 实例化方法/类 } }
        self.instance_registry = {}

    def register(self, field_name: str, mapping_dict: dict):
        """
        优雅的字符串到实例注册方法
        :param field_name: dataclass 中的字段名
        :param mapping_dict: 字符串参数到类/工厂函数的映射字典
        """
        self.instance_registry[field_name] = mapping_dict
        return self

    def _get_field_type(self, field):
        return self.resolved_type_hints.get(field.name, field.type)

    def _get_parser_action(self, field):
        """决定如何将命令行的字符串解析为目标对象"""
        field_type = self._get_field_type(field)

        # 优先级1：如果字段被注册了实例映射器 (例如 optimizer 字段)
        if field.name in self.instance_registry:
            mapping = self.instance_registry[field.name]
            def instance_factory(x):
                if x not in mapping:
                    raise ValueError(f"Invalid option '{x}' for {field.name}. Available: {list(mapping.keys())}")
                return mapping[x]
            return instance_factory
        if get_origin(field_type) is Literal:
            literal_values = get_args(field_type)
            if not literal_values:
                raise ValueError(f"Literal field '{field.name}' has no candidate values")
            return type(literal_values[0])
        if get_origin(field_type) is list:
            return self.type_parsers[list]
        if field_type in self.type_parsers:
            return self.type_parsers[field_type]
        return field_type

    @staticmethod
    def _parse_extra_value(raw_value: str):
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            return ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return raw_value

    def _parse_unknown_cli_args(self, unknown_args):
        extras = {}
        index = 0
        while index < len(unknown_args):
            token = unknown_args[index]
            if not token.startswith("--"):
                raise ValueError(
                    f"Unexpected CLI token '{token}'. Extra parameters must use '--key value' or '--key=value'."
                )

            if "=" in token:
                key, raw_value = token[2:].split("=", 1)
                extras[key.replace("-", "_")] = self._parse_extra_value(raw_value)
                index += 1
                continue

            key = token[2:].replace("-", "_")
            if not key:
                raise ValueError("Encountered an empty extra CLI parameter name.")

            if index + 1 < len(unknown_args) and not unknown_args[index + 1].startswith("--"):
                extras[key] = self._parse_extra_value(unknown_args[index + 1])
                index += 2
            else:
                extras[key] = True
                index += 1
        return extras

    def build(self, **code_kwargs):
        # 1. 【修复点】安全提取基类默认字段（同时兼容 default 和 default_factory）
        config_dict = {}
        for f in fields(self.config_class):
            if f.default is not dataclasses.MISSING:
                config_dict[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                config_dict[f.name] = f.default_factory()  # 这里会安全地生成 {}

        converters = {f.name: self._get_parser_action(f) for f in fields(self.config_class)}

        extras_dict = {}

        # 2. 处理代码直接传入的参数 (code_kwargs)
        for k, v in code_kwargs.items():
            if k in self.instance_registry and isinstance(v, str):
                mapping = self.instance_registry[k]
                if v not in mapping:
                    raise ValueError(f"Invalid option '{v}' for '{k}'. Available: {list(mapping.keys())}")
                processed_v = mapping[v]
            else:
                processed_v = v

            # 分发结果
            if k in config_dict:
                config_dict[k] = processed_v
            elif k != "extras":  # 防止覆盖本身的 extras 字段
                extras_dict[k] = processed_v

        # 3. 处理命令行参数
        parser = argparse.ArgumentParser()
        for f in fields(self.config_class):
            parser.add_argument(f"--{f.name}", type=converters[f.name], default=argparse.SUPPRESS)

        known_field_names = {f.name for f in fields(self.config_class)}
        for reg_k, mapping in self.instance_registry.items():
            if reg_k not in known_field_names:
                def dynamic_factory(x, m=mapping, key=reg_k):
                    if x not in m: raise ValueError(f"Invalid '{key}': {x}")
                    return m[x]

                parser.add_argument(f"--{reg_k}", type=dynamic_factory, default=argparse.SUPPRESS)

        cli_args, unknown_args = parser.parse_known_args()

        # 4. 合并命令行参数解析结果
        for k, v in vars(cli_args).items():
            if k in config_dict:
                config_dict[k] = v
            elif k != "extras":
                extras_dict[k] = v

        extras_dict.update(self._parse_unknown_cli_args(unknown_args))

        # 5. 【现在的 config_dict["extras"] 绝对是一个字典了】
        if "extras" in config_dict:
            config_dict["extras"].update(extras_dict)

        # 6. 实例化
        cfg = self.config_class(**config_dict)
        if hasattr(cfg, "seed"):
            seed_everything(cfg.seed)
        return cfg
    def __str__(self):
        lines = [f"ParamManager managing <{self.config_class.__name__}>"]
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<18} | {'Type':<12} | {'Default'}")
        lines.append("-" * 60)
        for f in fields(self.config_class):
            field_type = self._get_field_type(f)
            type_name = getattr(field_type, '__name__', str(field_type))
            if f.default is dataclasses.MISSING and f.default_factory is not dataclasses.MISSING:
                default_val = "<factory>"
            elif f.default is dataclasses.MISSING:
                default_val = "<required>"
            else:
                default_val = repr(f.default)
            lines.append(f"{f.name:<18} | {type_name:<12} | {default_val}")
        lines.append("-" * 60)
        return "\n".join(lines)

if __name__ == '__main__':
    from rosen.models.msr.sharedbottom import SharedBottomLessModel
    manager = ParamManager(ConfigBase) # 必须是出现在ConfigBase通过doc声明的方式才能被探测到
    manager.register("model", {
        "sharedbottom": SharedBottomLessModel
    })
    cfg = manager.build(model="sharedbottom")
    print(manager)
    print(cfg)
````

## File: src/rosen/models/msr/components/__init__.py
````python
from rosen.models.msr.components.codebook import SharedCodebookBottleneck
from rosen.models.msr.components.heads import DomainTowerHead

__all__ = [
    "SharedCodebookBottleneck",
    "DomainTowerHead",
]
````

## File: src/rosen/models/msr/components/codebook.py
````python
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _l2norm(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, p=2, dim=-1)


def _ema_inplace(moving_avg: torch.Tensor, new_value: torch.Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new_value, alpha=(1.0 - decay))


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    num_samples = int(samples.size(0))
    if num_samples == 0:
        raise ValueError("Cannot initialize codebook from an empty batch")
    if num_samples >= int(num):
        indices = torch.randperm(num_samples, device=samples.device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=samples.device)
    return samples[indices]


def _kmeans_cosine(
        samples: torch.Tensor,
        num_clusters: int,
        num_iters: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    means = _sample_vectors(samples, num_clusters)
    means = _l2norm(means)

    for _ in range(max(1, int(num_iters))):
        similarities = samples @ means.t()
        buckets = similarities.argmax(dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        safe_bins = bins.masked_fill(bins == 0, 1)

        new_means = torch.zeros_like(means)
        scatter_index = buckets.unsqueeze(-1).expand(-1, samples.size(-1))
        new_means.scatter_add_(0, scatter_index, samples)
        new_means = new_means / safe_bins.unsqueeze(-1)
        new_means = _l2norm(new_means)
        means = torch.where((bins == 0).unsqueeze(-1), means, new_means)
    return means, bins


class _EmbeddingEMA(nn.Module):
    def __init__(
            self,
            num_tokens: int,
            codebook_dim: int,
            *,
            decay: float = 0.99,
            kmeans_init: bool = False,
    ):
        super().__init__()
        if kmeans_init:
            weight = torch.zeros(num_tokens, codebook_dim)
        else:
            weight = _l2norm(torch.randn(num_tokens, codebook_dim))

        self.num_tokens = int(num_tokens)
        self.codebook_dim = int(codebook_dim)
        self.decay = float(decay)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.register_buffer("initted", torch.tensor([not kmeans_init], dtype=torch.bool))

    @torch.jit.ignore
    def init_embed_(self, data: torch.Tensor) -> None:
        if bool(self.initted.item()):
            return
        embed, _ = _kmeans_cosine(data, self.num_tokens, num_iters=10)
        self.weight.data.copy_(embed)
        self.initted.data.copy_(torch.tensor([True], dtype=torch.bool, device=self.initted.device))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.weight)


class NormEMAVectorQuantizer(nn.Module):
    def __init__(
            self,
            num_embed: int,
            embedding_dim: int,
            *,
            decay: float = 0.99,
            kmeans_init: bool = False,
    ):
        super().__init__()
        self.num_tokens = int(num_embed)
        self.codebook_dim = int(embedding_dim)
        self.decay = float(decay)
        self.embedding = _EmbeddingEMA(
            self.num_tokens,
            self.codebook_dim,
            decay=decay,
            kmeans_init=kmeans_init,
        )
        self.register_buffer("cluster_size", torch.zeros(self.num_tokens))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = _l2norm(z)
        flattened = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(flattened)

        distances = (
            flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * flattened @ self.embedding.weight.t()
        )
        encoding_indices = distances.argmin(dim=1)
        z_q = self.embedding(encoding_indices).view_as(z)

        encodings = F.one_hot(encoding_indices, self.num_tokens).type(flattened.dtype)
        bins = encodings.sum(dim=0)
        _ema_inplace(self.cluster_size, bins, self.decay)

        if self.training:
            zero_mask = bins == 0
            safe_bins = bins.masked_fill(zero_mask, 1.0)

            embed_sum = flattened.t() @ encodings
            embed_normalized = (embed_sum / safe_bins.unsqueeze(0)).t()
            embed_normalized = _l2norm(embed_normalized)
            embed_normalized = torch.where(
                zero_mask.unsqueeze(-1),
                self.embedding.weight,
                embed_normalized,
            )
            _ema_inplace(self.embedding.weight, embed_normalized, self.decay)
            self.embedding.weight.data.copy_(_l2norm(self.embedding.weight.data))

        loss = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices


class SharedCodebookBottleneck(nn.Module):
    def __init__(
            self,
            input_dim: int,
            *,
            projected_dim: Optional[int] = None,
            num_tokens: int = 256,
            codebook_dim: int = 8,
            decay: float = 0.99,
            kmeans_init: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.projected_dim = int(projected_dim or input_dim)
        self.codebook_dim = int(codebook_dim)
        if self.projected_dim <= 0:
            raise ValueError("projected_dim must be positive")
        if self.codebook_dim <= 0:
            raise ValueError("codebook_dim must be positive")
        if self.projected_dim % self.codebook_dim != 0:
            raise ValueError(
                f"projected_dim={self.projected_dim} must be divisible by codebook_dim={self.codebook_dim}"
            )

        self.output_dim = self.projected_dim
        self.num_chunks = self.projected_dim // self.codebook_dim
        self.input_proj = (
            nn.Identity()
            if self.projected_dim == self.input_dim
            else nn.Linear(self.input_dim, self.projected_dim)
        )
        self.quantizer = NormEMAVectorQuantizer(
            num_embed=num_tokens,
            embedding_dim=self.codebook_dim,
            decay=decay,
            kmeans_init=kmeans_init,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 2:
            raise ValueError(f"SharedCodebookBottleneck expects [B, D] input, got shape={tuple(x.shape)}")

        projected = self.input_proj(x)
        chunks = projected.view(projected.size(0), self.num_chunks, self.codebook_dim)
        quantized, quantize_loss, encoding_indices = self.quantizer(chunks)
        quantized_flat = quantized.reshape(projected.size(0), self.output_dim)

        usage = self.quantizer.cluster_size.detach().clone()
        usage_prob = usage / usage.sum().clamp_min(1e-12)
        perplexity = torch.exp(
            -(usage_prob * usage_prob.clamp_min(1e-12).log()).sum()
        )
        aux = {
            "projected": projected,
            "quantized": quantized_flat,
            "quantize_loss": quantize_loss,
            "encoding_indices": encoding_indices.view(projected.size(0), self.num_chunks),
            "code_usage": usage,
            "perplexity": perplexity,
        }
        return quantized_flat, aux
````

## File: src/rosen/models/msr/components/heads.py
````python
from typing import Iterable

import torch
from torch import nn

from rosen.models.utils.common import to_dims
from rosen.models.utils.container import MultiScenarioContainer
from rosen.models.utils.general import MLP


class DomainTowerHead(nn.Module):
    def __init__(
            self,
            num_domains: int,
            input_dim: int,
            hidden_dims: Iterable[int] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
    ):
        super().__init__()
        default_hidden = (max(1, input_dim // 2),)
        tower_hidden_dims = to_dims(hidden_dims, default_hidden)
        self.head = MultiScenarioContainer(
            num_domains,
            lambda: MLP(
                input_dim,
                *tower_hidden_dims,
                1,
                dropout_rate=dropout_rate,
                activation=activation,
                batch_norm=batch_norm,
            )
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(x, domain_ids).squeeze(-1)
````

## File: src/rosen/models/msr/__init__.py
````python
from typing import Dict, Type, Union

from rosen.emb import SchemaManager
from rosen.models.msr.all_expert_mixer import AllDomainExpertMixerModel
from rosen.models.msr.automtl import AutoMTLModel
from rosen.models.msr.base import MSRModel
from rosen.models.msr.crocodile import CrocodileModel
from rosen.models.msr.din import ADSDINModel, DINModel
from rosen.models.msr.hamur import HAMURModel
from rosen.models.msr.hierrec import HierRec
from rosen.models.msr.hypergate import HyperGateModel
from rosen.models.msr.m2m import M2MModel
from rosen.models.msr.m3oe import M3oEModel, M3oEVersion1Model, M3oEVersion2Model
from rosen.models.msr.mmoe import GradientBalancedMMoEModel, MMoEModel
from rosen.models.msr.multiemb import MultiEmbModel
from rosen.models.msr.pareto import ParetoModel
from rosen.models.msr.pepnet import EPNetModel, PEPNetModel, PPNetModel
from rosen.models.msr.ple import PLEModel, PLEVersion1Model
from rosen.models.msr.representation_contexture import ContexturePrototypeSharedBottomModel
from rosen.models.msr.riple import RIPLEModel
from rosen.models.msr.satrans import SATransModel
from rosen.models.msr.sdsp import SDSPMMoEModel, SDSPPLEModel
from rosen.models.msr.sharedbottom import (
    SharedBottomCodebookModel,
    SharedBottomLessModel,
    SharedBottomModel,
    SharedBottomPlusModel,
)
from rosen.models.msr.ssim import SSIMModel
from rosen.models.msr.star import AdditiveSTARModel, STARModel


MODEL_REGISTRY: Dict[str, Type[MSRModel]] = {
    "sharedbottom": SharedBottomModel,
    "sharedbottom_codebook": SharedBottomCodebookModel,
    "sharedbottom_less": SharedBottomLessModel,
    "sharedbottom_plus": SharedBottomPlusModel,
    "din": DINModel,
    "ads_din": ADSDINModel,
    "all_expert_mixer": AllDomainExpertMixerModel,
    "contexture_prototype_sharedbottom": ContexturePrototypeSharedBottomModel,
    "mmoe": MMoEModel,
    "mmoe_grad_balance": GradientBalancedMMoEModel,
    "ple": PLEModel,
    "ple_v1": PLEVersion1Model,
    "star": STARModel,
    "star_add": AdditiveSTARModel,
    "m3oe": M3oEModel,
    "m3oe_v1": M3oEVersion1Model,
    "m3oe_v2": M3oEVersion2Model,
    "m2m": M2MModel,
    "ppnet": PPNetModel,
    "epnet": EPNetModel,
    "pepnet": PEPNetModel,
    "crocodile": CrocodileModel,
    "pareto": ParetoModel,
    "hierrec": HierRec,
    "automtl": AutoMTLModel,
    "riple": RIPLEModel,
    "satrans": SATransModel,
    "hamur": HAMURModel,
    "hypergate": HyperGateModel,
    "sdsp_ple": SDSPPLEModel,
    "sdsp_mmoe": SDSPMMoEModel,
    "ssim": SSIMModel,
    "multiemb": MultiEmbModel,
}

def update_register(**kwargs) -> Dict[str, Type[MSRModel]]:
    for name, model_cls in kwargs.items():
        if not isinstance(name, str):
            raise TypeError(f"Model registry key must be str, got {type(name)}")
        if not isinstance(model_cls, type) or not issubclass(model_cls, MSRModel):
            raise TypeError(
                f"Model registry value for '{name}' must be an MSRModel subclass, got {model_cls!r}"
            )
        MODEL_REGISTRY[name.lower()] = model_cls
    return MODEL_REGISTRY


def build_model(
        schema_manager: SchemaManager,
        num_domains: int,
        model_cls: Union[str, Type[MSRModel]],
        **model_kwargs,
):

    if isinstance(model_cls, str):
        model_name = model_cls
        model_cls = MODEL_REGISTRY.get(model_name.lower())
    else:
        model_name = model_cls.__name__
    if model_cls is None:
        raise ValueError(f"Unknown model_name={model_name}. Available: {sorted(MODEL_REGISTRY.keys())}")
    return model_cls.from_manager(schema_manager, num_domains, **dict(model_kwargs or {}))


__all__ = [
    "MODEL_REGISTRY",
    "update_register",
    "build_model",
    "HierRec",
    "SharedBottomModel",
    "SharedBottomCodebookModel",
    "SharedBottomLessModel",
    "SharedBottomPlusModel",
    "DINModel",
    "ADSDINModel",
    "AllDomainExpertMixerModel",
    "ContexturePrototypeSharedBottomModel",
    "MMoEModel",
    "GradientBalancedMMoEModel",
    "PLEModel",
    "PLEVersion1Model",
    "STARModel",
    "AdditiveSTARModel",
    "M3oEModel",
    "M3oEVersion1Model",
    "M3oEVersion2Model",
    "RIPLEModel",
    "SATransModel",
    "SDSPMMoEModel",
    "SDSPPLEModel",
    "SSIMModel",
    "MultiEmbModel",
    "M2MModel",
    "PPNetModel",
    "EPNetModel",
    "PEPNetModel",
    "CrocodileModel",
    "ParetoModel",
    "AutoMTLModel",
    "HAMURModel",
    "HyperGateModel",
]
````

## File: src/rosen/models/msr/base.py
````python
import inspect
import warnings

import torch

from rosen.emb import SchemaManager
from rosen.models.base import BaseModel


class MSRModel(BaseModel):
    def __init__(self, manager: SchemaManager, num_domains: int, **kwargs):
        super(MSRModel, self).__init__(manager)
        self.num_domains = num_domains

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor):
        raise NotImplementedError

    @classmethod
    def from_manager(cls, manager: SchemaManager, num_domains: int, **kwargs) -> 'MSRModel':
        init_signature = inspect.signature(cls.__init__)
        parameters = init_signature.parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kwargs:
            return cls(manager, num_domains, **kwargs)

        valid_kwargs = {}
        ignored_keys = []
        for key, value in kwargs.items():
            if key in parameters and key not in {"self", "manager", "num_domains"}:
                valid_kwargs[key] = value
            else:
                ignored_keys.append(key)

        if ignored_keys:
            warnings.warn(
                f"{cls.__name__}.from_manager ignored unsupported kwargs: {sorted(ignored_keys)}",
                stacklevel=2,
            )
        return cls(manager, num_domains, **valid_kwargs)
````

## File: src/rosen/models/msr/sharedbottom.py
````python
from typing import Iterable, Optional

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead, SharedCodebookBottleneck
from rosen.models.utils.activation import activation_layer
from rosen.models.utils.common import build_mlp, to_dims


class SharedBottomBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_domains: int,
            hidden_dims=None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            activate_output: bool = False,
    ):
        super().__init__()
        hidden_dims = to_dims(hidden_dims, (input_dim, input_dim))
        self.output_dim = int(hidden_dims[-1])
        self.activate_output = bool(activate_output)
        self.bottom = build_mlp(
            input_dim,
            hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )
        self.output_activation = (
            activation_layer(activation, self.output_dim)
            if self.activate_output
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        del domain_ids
        return self.output_activation(self.bottom(x))


class SharedBottomModel(MSRModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = (256, 128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            activate_backbone_output: bool = False,
            tower_hidden_dims: Optional[Iterable[int]] = (128, 64),
            tower_dropout_rate: float = 0.0,
            include_domain_in_input: bool = True,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.include_domain_in_input = bool(include_domain_in_input)
        self.input_view = (
            self.omni_embedding.whole if self.include_domain_in_input
            else self.omni_embedding.whole_without_domain
        )
        self.input_dim = self.input_view.embedding_dim
        resolved_hidden_dims = (
            self._default_hidden_dims()
            if hidden_dims is None
            else tuple(hidden_dims)
        )

        self.backbone = SharedBottomBackbone(
            input_dim=self.input_dim,
            num_domains=num_domains,
            hidden_dims=resolved_hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            activate_output=activate_backbone_output,
        )

        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.backbone.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def _default_hidden_dims(self) -> tuple[int, ...]:
        input_dim = max(1, int(self.input_dim))
        return input_dim, input_dim

    def encode_features(self, interaction):
        x = self.input_view(interaction)
        return torch.flatten(x, start_dim=1), interaction[self.DOMAIN].long()

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x, domain_ids)
        return self.head(hidden, domain_ids)

    def predict(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.predict(interaction)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if self.lsp.active:
            self.lsp.add("task", loss)
        return loss


class SharedBottomLessModel(SharedBottomModel):
    def _default_hidden_dims(self) -> tuple[int, ...]:
        return (max(1, int(self.input_dim)),)

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Optional[Iterable[int]] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )


class SharedBottomPlusModel(SharedBottomModel):
    def _default_hidden_dims(self) -> tuple[int, ...]:
        input_dim = max(1, int(self.input_dim))
        return input_dim, input_dim, max(1, input_dim // 2), input_dim

    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = True,
            tower_hidden_dims: Optional[Iterable[int]] = None,
            tower_dropout_rate: float = 0.2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )


class SharedBottomCodebookModel(SharedBottomModel):
    def __init__(
            self,
            manager: SchemaManager,
            num_domains: int,
            hidden_dims: Iterable[int] = (256, 128, 64),
            dropout_rate: float = 0.0,
            activation: str = "relu",
            batch_norm: bool = False,
            tower_hidden_dims: Iterable[int] = (128, 64),
            tower_dropout_rate: float = 0.0,
            num_tokens: int = 256,
            codebook_dim: int = 8,
            codebook_decay: float = 0.99,
            codebook_input_proj_dim: Optional[int] = None,
            codebook_kmeans_init: bool = False,
            quantize_weight: float = 1e-2,
    ):
        super().__init__(
            manager=manager,
            num_domains=num_domains,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            tower_hidden_dims=tower_hidden_dims,
            tower_dropout_rate=tower_dropout_rate,
        )
        self.quantize_weight = float(quantize_weight)
        self.codebook = SharedCodebookBottleneck(
            input_dim=self.backbone.output_dim,
            projected_dim=codebook_input_proj_dim,
            num_tokens=num_tokens,
            codebook_dim=codebook_dim,
            decay=codebook_decay,
            kmeans_init=codebook_kmeans_init,
        )
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.codebook.output_dim,
            hidden_dims=tower_hidden_dims,
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
        )

    def _forward_dict(self, x: torch.Tensor, domain_ids: torch.Tensor):
        hidden = self.backbone(x, domain_ids)
        quantized, aux = self.codebook(hidden)
        logits = self.head(quantized, domain_ids)
        forward_dict = {
            "hidden": hidden,
            "quantized": quantized,
            "logits": logits,
            **aux,
        }
        return forward_dict

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self._forward_dict(x, domain_ids)["logits"]

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        x, domain_ids = self.encode_features(interaction)
        forward_dict = self._forward_dict(x, domain_ids)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(forward_dict["logits"], labels)
        quantize_loss = forward_dict["quantize_loss"]
        if self.lsp.active:
            self.lsp.add("task", bce_loss)
            self.lsp.add("vq.quantize", quantize_loss, weight=self.quantize_weight)
        return bce_loss + self.quantize_weight * quantize_loss
````

## File: src/rosen/models/utils/activation.py
````python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, emb_size, dim=2, epsilon=1e-8, device='cpu'):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)).to(device))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)).to(device))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    if isinstance(act_name, str):
        act_name = act_name.lower()
        act_builder = {
            'sigmoid': lambda: nn.Sigmoid(),
            'linear':  lambda: Identity(),
            'relu':    lambda: nn.ReLU(inplace=True),
            'prelu':   lambda: nn.PReLU(),
            'dice':    lambda: Dice(hidden_size, dice_dim) if hidden_size else ValueError("Dice requires hidden_size"),
            'silu':    lambda: nn.SiLU()
        }
        if act_name in act_builder:
            return act_builder[act_name]()  # 注意这里的 ()，这才是真正执行实例化的地方
        else:
            raise NotImplementedError(f"Activation {act_name} is not supported.")

    elif issubclass(act_name, nn.Module):
        return act_name()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    pass
````

## File: src/rosen/models/utils/common.py
````python
from collections.abc import Iterable
from torch import nn
from rosen.models.utils.general import MLP

def default_dims(input_dim):
    return [input_dim, input_dim // 2, 128]

def to_dims(value, default):
    if value is None:
        value = default

    if isinstance(value, int):
        dims = (int(value),)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        dims = tuple(int(v) for v in value)
    else:
        raise TypeError(f"Unsupported dims type: {type(value)}")

    if not dims:
        raise ValueError("dims can not be empty")
    return dims

def default_backbone_dims(input_dim: int, depth: int = 2) -> tuple[int, ...]:
    input_dim = max(1, int(input_dim))
    if depth <= 1:
        return (input_dim,)
    hidden = max(1, input_dim)
    return tuple(hidden for _ in range(depth))


def build_mlp(
        input_dim: int,
        hidden_dims,
        *,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False,
):
    dims = to_dims(hidden_dims, default_backbone_dims(input_dim))
    return MLP(input_dim, *dims, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm)


def build_gate(input_dim: int,
               hidden_dims,
               out_dim: int = 1,
               dropout_rate: float = 0.0,
               activation: str = "relu",
               batch_norm: bool = False,):
    """
    构建专家网络。
    :param output_dim: 如果有输出要求，默认emb // size 改成 传参
    :param in_dim: 输入特征维度
    :param depth_multiplier: 深度倍数。
                             1 对应单层 PLE 的深度 [D, D/2]
                             2 对应两层 PLE 的总深度 [D, D, D/2]
    """
    # 初始化输入维度
    dims = to_dims(hidden_dims, default_backbone_dims(input_dim))
    out_dim = to_dims(out_dim, 1)
    out_nums = out_dim[0]
    dims += out_dim
    mlp = MLP(input_dim, *dims, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm)
    if out_nums > 1 :
        return nn.Sequential(mlp, nn.Softmax(dim=-1))
    else:
        return nn.Sequential(mlp, nn.Sigmoid())

if __name__ == '__main__':
    gate = build_gate(128, (256, 64), 4)
    print(gate)
    import torch
    x = torch.randn(10, 128)
    x = gate(x)
    print(x.shape)
````

## File: src/rosen/models/utils/container.py
````python
from typing import List, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torch.nn.functional as F
from rosen.models.utils.general import MLP

def domain_select(tensor, domain_ids):
    batch_size = tensor.size(0)
    batch_indices = torch.arange(batch_size, device=tensor.device)
    tensor = tensor[batch_indices, domain_ids]
    return tensor

class MultiScenarioCloneBase(nn.Module, ABC):
    """
    多场景分身模型抽象基类
    继承此类的子类只需实现 `build_single_domain_net` 即可自动获得多场景能力。
    """
    def __init__(self, num_domains: int):
        super().__init__()
        self.num_domains = num_domains
        self.domain_networks = nn.ModuleList([
            self.build_single_domain_net() for _ in range(num_domains)
        ])

    @abstractmethod
    def build_single_domain_net(self) -> nn.Module:
        """
        子类必须实现此方法。
        在这里 return 一个标准的单场景网络结构 (例如一个简单的 MLP)。
        """
        pass

    def forward(self, x: torch.Tensor, select_domain_ids: torch.Tensor=None):
        """
        x: [batch_size, feature_dim] 共享层输出的特征
        domain_ids: [batch_size] 场景标识 (值为 0 到 num_domains - 1)
        """
        all_results = []
        for i in range(self.num_domains):
            out_i = self.domain_networks[i](x)
            all_results.append(out_i)
        logits = torch.stack(all_results, dim=1)
        if select_domain_ids is not None:
            if select_domain_ids is not None:
                logits = domain_select(logits, select_domain_ids)
        return logits


class MultiScenarioContainer(nn.Module):
    def __init__(self, num_domains: int, network_factory: Callable[[], nn.Module]):
        super().__init__()
        self.num_domains = num_domains
        self.domain_networks = nn.ModuleList([
            network_factory() for _ in range(num_domains)
        ])

    def forward(self, x: torch.Tensor, select_domain_ids: torch.Tensor = None):
        all_results = []
        for i in range(self.num_domains):
            out_i = self.domain_networks[i](x)
            all_results.append(out_i)

        logits = torch.stack(all_results, dim=1)
        if select_domain_ids is not None:
            if select_domain_ids is not None:
                logits = domain_select(logits, select_domain_ids)
        return logits

    def __getitem__(self, idx):
        return self.domain_networks[idx]


class MultiTaskContainer(nn.Module):
    """
    多任务网络容器。
    不同于多场景的“按需选择”，多任务需要对输入特征进行“全量分发与收集”。
    """
    def __init__(self, task_names: List[str], network_factory: Callable[[], nn.Module]):
        """
        :param task_names: 任务名称列表，例如 ['ctr', 'cvr']
        :param network_factory: 实例化单任务网络的工厂函数
        """
        super().__init__()
        self.task_names = task_names
        self.task_networks = nn.ModuleDict({
            task: network_factory() for task in task_names
        })

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: 底层提取出来的共享特征 [Batch, Feature_Dim]
        kwargs: 透传参数（例如可以把 domain_ids 透传给内部的多场景网络）
        """
        task_logits = {}
        for task_name, task_net in self.task_networks.items():
            task_logits[task_name] = task_net(x, **kwargs)
        return task_logits

    def __getitem__(self, task_name):
        return self.task_networks[task_name]



if __name__ == '__main__':
    class PLE(nn.Module):
        def __init__(self, num_domains):
            super().__init__()
            ## 第一层，共享兴趣由所有人共同设计，分兴趣很普通
            self.multi_experts_layer1 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: MLP(128, 64, 128)
            )
            # 2. 直接传入 lambda 函数来实例化多场景门控网络
            self.multi_gates_layer1 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: nn.Sequential(
                    MLP(128, 1),
                    nn.Sigmoid()
                )
            )
            self.shared_expert_layer1 = MLP(128, 64, 128)
            self.shared_gates_layer1 = nn.Sequential(
                MLP(128, num_domains + 1),
                nn.Softmax(dim=-1)
            )

            ## 第二层，共享兴趣再被输入
            self.multi_experts_layer2 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: MLP(128, 64, 128)
            )
            # 2. 直接传入 lambda 函数来实例化多场景门控网络
            self.multi_gates_layer2 = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: nn.Sequential(
                    MLP(128, 1),
                    nn.Sigmoid()
                )
            )
            self.shared_expert_layer2 = MLP(128, 64, 128)
            self.towers = MultiScenarioContainer(
                num_domains=num_domains,
                network_factory=lambda: nn.Sequential(
                    MLP(128, 1),
                    nn.Sigmoid()
                )
            )

        def forward(self, x, domain_ids):
            """
            :param x: B, D
            """
            all_feature = self.multi_experts_layer1(x)
            spe_feature = domain_select(all_feature, domain_ids)
            sha_feature = self.shared_expert_layer1(x)
            spe_gates = self.multi_gates_layer1(x, domain_ids)
            sha_gates = self.shared_gates_layer1(x).unsqueeze(-1) # B K+1 1

            spe_interest = spe_feature * spe_gates + sha_feature * (1 - spe_gates)
            weighted_spe_feature = sha_gates * torch.cat([all_feature, sha_feature.unsqueeze(1)], dim=1)
            sha_interest = torch.sum(weighted_spe_feature, dim=1)

            ## 第二层
            spe_state = self.multi_experts_layer2(spe_interest, domain_ids)
            sha_state = self.shared_expert_layer2(sha_interest)
            spe_gates = self.multi_gates_layer2(spe_interest, domain_ids)
            return self.towers(spe_state * spe_gates + sha_state * (1 - spe_gates), domain_ids).squeeze()


    # 设置超参
    BATCH_SIZE = 4
    FEATURE_DIM = 128
    NUM_DOMAINS = 3  # 假设有3个不同的场景 (0, 1, 2)

    # 初始化模型
    model = PLE(num_domains=NUM_DOMAINS)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    domain_ids = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    print("=== 输入信息 ===")
    print(f"输入 x 形状: {x.shape}")
    print(f"输入 domain_ids: {domain_ids}\n")

    # 执行前向传播
    final_out = model(x, domain_ids)

    print("=== 提取结果证明 ===")
    print(f"模型最终输出 形状: {final_out.shape}")

    # 验证最终形状是否符合预期 [Batch, Feature_Dim]
    assert final_out.shape == (BATCH_SIZE, FEATURE_DIM), "最终输出维度错误！"
    print("\n✅ Forward pass 成功，没有发生维度崩溃！")
````

## File: src/rosen/models/utils/general.py
````python
from typing import List, Iterable
import torch
from torch import nn

from rosen.models.utils.activation import activation_layer

class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', dropout_rate: float=0., use_bn=False, dice_dim=2):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

# class MLP(nn.Module):
#     def __init__(self, *dims, dropout_rate: float = 0.0, activation:str = 'relu', batch_norm=False):
#         super().__init__()
#         self.net = DNN(dims[0], dims[1:], activation, dropout_rate=dropout_rate, use_bn=batch_norm)
#
#     def forward(self, x):
#         return self.net(x)


class MLP(nn.Module):
    def __init__(self, *dims, dropout_rate: float = 0.0, activation: str = 'relu', batch_norm: bool = False):
        """
        Args:
            *dims: 动态传入各个层的维度。例如 MLP(256, 128, 64) 代表 256->128->64 的两层网络。
            dropout_rate: 丢弃率
            activation: 激活函数名称 ('relu', 'prelu', 'sigmoid', 'linear')
            batch_norm: 是否在隐层使用 BatchNorm1d
        """
        super().__init__()

        if len(dims) < 2:
            raise ValueError("MLP 至少需要输入维度和输出维度，例如 MLP(256, 128)")

        layers = []
        # 2. 遍历构建网络层
        for i in range(len(dims) - 1):
            is_last_layer = (i == len(dims) - 2)

            # (1) 线性映射
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # (2) 隐层处理（最后一层通常不需要加 BN、激活和 Dropout，直接输出 Logits）
            if not is_last_layer:
                # 标准范式：Linear -> BN -> Activation -> Dropout
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                layers.append(activation_layer(activation))

                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))

        # 使用 nn.Sequential 自动管理前向传播
        self.net = nn.Sequential(*layers)

        # 3. 科学的参数初始化（替换掉 DeepCTR 里那个极慢的 0.0001）
        self._init_weights(activation)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation.lower() in ['relu', 'prelu']:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn


class FeatureBifurcator(nn.Module):
    def __init__(self, num_features, feature_dim=-1, mean_dims=0, normalize_var=False, stack_dim=1):
        """
        通用特征分化组件：将任意特征分离为“静态可学习基准(Bias)”和“零均值动态波动(Fluctuation)” 一般来说只使用Fluctuation，并且监控它的大小可以判断特征的重要程度

        :param num_features: 特征的通道数（用于初始化可学习的静态 Bias）
        :param feature_dim: 特征所在的维度索引。对于 Linear 是 -1，对于 Conv2d(B,C,H,W) 通常是 1
        :param mean_dims: 在哪些维度上计算均值。
                          - Batch级别求均值: 0
                          - 图像空间求均值 (InstanceNorm风格): (2, 3)
                          - 特征内部求均值 (LayerNorm风格): feature_dim
        :param normalize_var: 是否除以标准差（True 则类似原版的 BN/LN，False 则只做纯粹的 0 均值化）
        :param stack_dim: 最终 stack 拼接的维度。默认在 dim=1 堆叠，生成 [..., 2, ...] 的形状
        """
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.mean_dims = mean_dims
        self.normalize_var = normalize_var
        self.stack_dim = stack_dim
        self.eps = 1e-5

        # 初始化静态基准 (可学习的 Bias)
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 1. 提取动态波动 (Fluctuation): 减去均值
        mean_val = x.mean(dim=self.mean_dims, keepdim=True)
        fluctuation = x - mean_val

        # 可选：是否进行方差归一化缩放
        if self.normalize_var:
            var_val = x.var(dim=self.mean_dims, keepdim=True, unbiased=False)
            fluctuation = fluctuation / torch.sqrt(var_val + self.eps)

        # 2. 形状对齐：将 1D bias 动态广播成与输入 x 完全相同的形状
        # 无论 x 是 [B, D] 还是 [B, C, H, W]，自动在 feature_dim 展开
        bias_shape = [1] * x.dim()
        bias_shape[self.feature_dim] = self.num_features
        bias_expanded = self.bias.view(*bias_shape).expand_as(x)

        return bias_expanded, fluctuation

class BifurcatedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, zero_mean_dim='batch'):
        """
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        :param zero_mean_dim: 'feature' (对每个样本的特征向量求0均值)
                           或 'batch' (对Batch内的特定特征位求0均值)
        """
        super().__init__()
        self.output_dim = output_dim
        self.zero_mean_dim = zero_mean_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # 定义 0 均值化的方法 (关闭 affine 避免模型偷偷把非零均值学回来)
        if self.zero_mean_dim == 'feature':
            self.zero_mean_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        elif self.zero_mean_dim == 'batch':
            self.zero_mean_norm = nn.BatchNorm1d(output_dim, affine=False)
        else:
            raise ValueError("zero_mean_dim must be 'feature' or 'batch'")
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size = x.size(0)
        raw_fluctuation = self.linear(x)
        fluctuation = self.zero_mean_norm(raw_fluctuation)  # shape: [Batch, OutputDim]
        bias_expanded = self.bias.unsqueeze(0).expand(batch_size, -1)
        return bias_expanded, fluctuation

class BifurcatedMLP(nn.Module):
    def __init__(self, *dims, dropout_rate: float = 0.0, activation: str = 'relu', batch_norm: bool = False):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP 至少需要输入维度和输出维度，例如 MLP(256, 128)")
        layers = []

        # 2. 遍历构建不包括最后一层的网络层
        for i in range(len(dims) - 1):
            is_last_layer = (i == len(dims) - 2)
            if not is_last_layer:
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(activation_layer(activation))
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))

        self.hidden_net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(dims[-2], dims[-1])
        self.bifurcator = FeatureBifurcator(dims[-1])
        self._init_weights(activation)

    @property
    def bias(self):
        return self.bifurcator.bias

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation.lower() in ['relu', 'prelu']:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.bifurcator(self.last_layer(self.hidden_net(x)))

class ModuleFactory:
    @staticmethod
    def build_expert(in_dim: int, hidout_dims=None, dropout_rate=0.1):
        """
        构建专家网络。
        :param output_dim: 如果有输出要求，默认emb // size 改成 传参
        :param in_dim: 输入特征维度
        :param depth_multiplier: 深度倍数。
                                 1 对应单层 PLE 的深度 [D, D/2]
                                 2 对应两层 PLE 的总深度 [D, D, D/2]
        """
        # 初始化输入维度
        hidout_dims = hidout_dims or in_dim // 2
        if not isinstance(hidout_dims, Iterable):
            hidout_dims = [hidout_dims]
        dims = [in_dim, *hidout_dims]
        return lambda: MLP(*dims, dropout_rate=dropout_rate, activation="relu")

    @staticmethod
    def build_gate(in_dim, num_output=None):
        if num_output is None or num_output == 1:
            num_output = 1
            return lambda: nn.Sequential(MLP(in_dim, in_dim // 2, num_output), nn.Sigmoid())
        else:
            return lambda: nn.Sequential(MLP(in_dim, in_dim // 2, num_output), nn.Softmax(-1))

    @staticmethod
    def build_tower(in_dim):
        return lambda: MLP(in_dim, in_dim // 2, 1, dropout_rate=0.2)


if __name__ == '__main__':
    N = 5000  # 样本量
    input_dim = 16  # 特征维度
    X = torch.randn(N, input_dim)

    # 我们强行设定一个极其刁钻的“全局偏置” (真实世界里可能是该场景的基础 CTR)
    TRUE_BIAS = 4.26

    # 真实用户的个性化偏好 (均值期望为 0，因为 X 是标准正态分布)
    # 假设前几个特征对结果有强影响
    TRUE_FLUCTUATION = X[:, 0] * 2.0 - X[:, 1] * 1.5 + X[:, 2] * 0.5

    # 最终的 Label = 全局偏置 + 个性化波动 + 少量高斯噪声
    Y = TRUE_BIAS + TRUE_FLUCTUATION + torch.randn(N) * 0.1
    Y = Y.unsqueeze(1)  # shape: [5000, 1]

    # ==========================================
    # 3. 训练模型，见证奇迹
    # ==========================================
    # 构建一个 MLP，输出维度必须是 1 (预测单分)
    model = BifurcatedMLP(16, 64, 32, 1)
    from torch import optim
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    print("\n=== 开始训练 ===")
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()

        # 前向传播，输出 shape: [Batch, 2, 1]
        out = model(X)

        # 把 Bias 和 Fluctuation 加起来算总预测值
        y_pred = out.sum(dim=1)  # shape: [Batch, 1]

        loss = criterion(y_pred, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/500, Loss: {loss.item():.4f}")

    # ==========================================
    # 4. 硬核解剖：验证解耦是否成功
    # ==========================================
    print("\n=== 验证解耦效果 ===")
    model.eval()
    with torch.no_grad():
        test_out = model(X)

        # 提取两部分
        learned_bias = test_out[:, 0, :]  # shape: [Batch, 1]
        learned_fluctuation = test_out[:, 1, :]  # shape: [Batch, 1]

        # 1. 验证 Bias 是否精准抓住了全局大盘水位
        # 参数 self.bias 和 BatchNorm 中记录的 running_mean 的总和才是物理上的真实 bias
        static_param_bias = model.bias.item()
        print()

        print(f"[验证 1] 设定的真实全局偏置 : {TRUE_BIAS}")
        print(f"[验证 1] 模型学到的总体 Bias: {model.bias.item():.4f}")

        # 2. 验证 Fluctuation 是否真的是 0 均值，且拟合了个性化波动
        mean_fluc = learned_fluctuation.mean().item()
        print(f"\n[验证 2] 预测调节值的均值 (应极度接近 0): {mean_fluc:.6f}")

        # 3. 验证 Fluctuation 的相关性
        # 如果解耦成功，学到的 fluctuation 应该和真值高度正相关
        fluc_flat = learned_fluctuation.squeeze()
        correlation = torch.corrcoef(torch.stack([fluc_flat, TRUE_FLUCTUATION]))[0, 1].item()
        print(f"[验证 3] 学到的波动与真实波动的皮尔逊相关系数: {correlation:.4f} (越接近 1.0 越好)")
````

## File: src/rosen/models/base.py
````python
import numpy as np
from torch import nn

from rosen.core.interaction import Interaction
from rosen.core.train.loss import LossPort
from rosen.emb import SchemaManager
from rosen.emb.emblayer import OmniEmbLayer
from rosen.utils.observatory.port import ObservationPort


class BaseModel(nn.Module):
    def __init__(self, manager: SchemaManager):
        super(BaseModel, self).__init__()
        self.manager: SchemaManager = manager
        self.omni_embedding: OmniEmbLayer = OmniEmbLayer(manager=manager)
        self.obp: ObservationPort = ObservationPort(namespace=type(self).__name__)
        self.lsp: LossPort = LossPort()

    def apply_xavier_initialization(self) -> None:
        """Optionally reset linear layers to Xavier normal with zero bias."""
        def _reset(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(_reset)

    def calculate_loss(self, interaction: Interaction):
        raise NotImplementedError

    def predict(self, interaction: Interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction: Interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + f": {params}"
        )
````

## File: src/rosen/utils/observatory/port.py
````python
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple, Union

import torch


@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: str = "generic"
    labels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationSpec:
    role: str = "tensor"
    axes: Tuple[AxisSpec, ...] = ()
    metrics: Tuple[str, ...] = ("basic", "spectral")
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class CapturedObservation:
    name: str
    value: Any
    namespace: str = ""

    @property
    def full_name(self) -> str:
        if not self.namespace:
            return self.name
        return f"{self.namespace}.{self.name}"


class ObservationPort:
    """
    Lightweight per-model trace buffer.

    The port is intentionally not a recorder: it does not compute metrics,
    print reports, or export artifacts. Models use it to mark interesting
    forward-pass tensors, and an external observatory can decide when to
    enable collection and how to interpret the captured trace.
    """

    def __init__(self, namespace: str = "", enabled: bool = False):
        self.namespace = str(namespace)
        self.enabled = bool(enabled)
        self._items: Dict[str, CapturedObservation] = {}
        self._specs: Dict[str, ObservationSpec] = {}
        self._cut_names: Set[str] = set()
        self._watch_names: Set[str] = set()
        self._active_cut: Optional[str] = None
        self._watch_enabled: bool = False
        self._watch_items: Dict[str, Any] = {}
        self._gradient_tap_names: Set[str] = set()
        self._gradient_tap_enabled: bool = False
        self._gradient_items: Dict[str, torch.Tensor] = {}
        self._gradient_handles: list[Any] = []

    @staticmethod
    def _normalize_axes(axes: Iterable[Union[str, AxisSpec]]) -> Tuple[AxisSpec, ...]:
        return tuple(axis if isinstance(axis, AxisSpec) else AxisSpec(name=str(axis), kind=str(axis)) for axis in axes)

    def register(
            self,
            name: str,
            spec: Optional[ObservationSpec] = None,
            *,
            role: str = "tensor",
            axes: Iterable[Union[str, AxisSpec]] = (),
            metrics: Iterable[str] = ("basic", "spectral"),
            metadata: Optional[Mapping[str, Any]] = None,
            overwrite: bool = False,
    ) -> ObservationSpec:
        key = str(name)
        if key in self._specs and not overwrite:
            raise ValueError(f"Observation {key!r} is already registered.")
        if spec is None:
            spec = ObservationSpec(
                role=str(role),
                axes=self._normalize_axes(axes),
                metrics=tuple(str(metric) for metric in metrics),
                metadata=dict(metadata or {}),
            )
        self._specs[key] = spec
        return spec

    def spec(self, name: str) -> Optional[ObservationSpec]:
        return self._specs.get(str(name))

    def clear(self) -> None:
        self._items.clear()

    def cut(self, name: str, value: Any) -> Any:
        """
        Mark a tensor as a one-at-a-time ablation point.

        In normal execution this is an identity function. When a matching cut
        name is active, tensor values are replaced by zeros at this exact point
        in the model's forward path.
        """
        key = str(name)
        self._cut_names.add(key)
        if self._active_cut != key:
            return value
        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        return value

    def watch(
            self,
            name: str,
            value: Any,
            *,
            detach: bool = True,
            clone: bool = False,
    ) -> Any:
        """
        Mark a downstream tensor to summarize during cut ablation search.
        """
        key = str(name)
        self._watch_names.add(key)
        if not self._watch_enabled:
            return value

        stored = value
        if isinstance(value, torch.Tensor):
            if detach:
                stored = value.detach()
            if clone:
                stored = stored.clone()
        self._watch_items[key] = stored
        return value

    def tap_gradient(
            self,
            name: str,
            value: Any,
            *,
            clone: bool = True,
    ) -> Any:
        """
        Mark a tensor as a backward-gradient tap.

        In normal execution this is an identity function. When gradient tapping
        is enabled, a backward hook stores the tensor's gradient under `name`.
        """
        key = str(name)
        self._gradient_tap_names.add(key)
        if not self._gradient_tap_enabled:
            return value
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            return value

        def store_gradient(grad: torch.Tensor) -> None:
            stored = grad.detach()
            if clone:
                stored = stored.clone()
            self._gradient_items[key] = stored

        self._gradient_handles.append(value.register_hook(store_gradient))
        return value

    def observe(
            self,
            name: str,
            value: Any,
            *,
            role: str = "tensor",
            axes: Iterable[Union[str, AxisSpec]] = (),
            metrics: Iterable[str] = ("basic", "rank"),
            metadata: Optional[Mapping[str, Any]] = None,
            watch: bool = False,
            gradient: bool = False,
            detach: bool = True,
            clone: bool = False,
            gradient_clone: bool = True,
    ) -> Any:
        """
        Declare one stable signal and participate in every active observation
        mode. Outside a collection/watch/gradient scope this remains a cheap
        identity operation.
        """
        key = str(name)
        if self.spec(key) is None:
            self.register(
                key,
                role=role,
                axes=axes,
                metrics=metrics,
                metadata=metadata,
            )
        self.capture(key, value, detach=detach, clone=clone)
        if watch:
            self.watch(key, value, detach=detach, clone=clone)
        if gradient:
            self.tap_gradient(key, value, clone=gradient_clone)
        return value

    @contextmanager
    def cutting(self, name: Optional[str]) -> Iterator["ObservationPort"]:
        previous = self._active_cut
        self._active_cut = None if name is None else str(name)
        try:
            yield self
        finally:
            self._active_cut = previous

    @contextmanager
    def watching(self, enabled: bool = True, clear: bool = True) -> Iterator["ObservationPort"]:
        previous = self._watch_enabled
        if clear:
            self._watch_items.clear()
        self._watch_enabled = bool(enabled)
        try:
            yield self
        finally:
            self._watch_enabled = previous

    @contextmanager
    def tapping_gradients(self, enabled: bool = True, clear: bool = True) -> Iterator["ObservationPort"]:
        previous = self._gradient_tap_enabled
        if clear:
            self.clear_gradients()
        self._gradient_tap_enabled = bool(enabled)
        try:
            yield self
        finally:
            self._gradient_tap_enabled = previous

    def cut_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._cut_names))

    def watch_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._watch_names))

    def gradient_tap_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._gradient_tap_names))

    def clear_cut_watch(self) -> None:
        self._cut_names.clear()
        self._watch_names.clear()
        self._watch_items.clear()
        self._active_cut = None

    def clear_gradients(self) -> None:
        self._gradient_items.clear()
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles.clear()

    def watch_snapshot(self, *, clear: bool = False) -> Dict[str, Any]:
        result = dict(self._watch_items)
        if clear:
            self._watch_items.clear()
        return result

    def gradient_snapshot(self, *, clear: bool = False) -> Dict[str, torch.Tensor]:
        result = dict(self._gradient_items)
        if clear:
            self.clear_gradients()
        return result

    def capture(
            self,
            name: str,
            value: Any,
            *,
            detach: bool = True,
            clone: bool = False,
    ) -> Any:
        if not self.enabled:
            return value

        stored = value
        if isinstance(value, torch.Tensor):
            if detach:
                stored = value.detach()
            if clone:
                stored = stored.clone()

        key = str(name)
        self._items[key] = CapturedObservation(
            name=key,
            value=stored,
            namespace=self.namespace,
        )
        return value

    def capture_many(
            self,
            values: Mapping[str, Any],
            *,
            detach: bool = True,
            clone: bool = False,
    ) -> Mapping[str, Any]:
        if not self.enabled:
            return values
        for name, value in values.items():
            self.capture(name, value, detach=detach, clone=clone)
        return values

    @contextmanager
    def collecting(self, enabled: bool = True, clear: bool = True) -> Iterator["ObservationPort"]:
        previous_enabled = self.enabled
        if clear:
            self.clear()
        self.enabled = bool(enabled)
        try:
            yield self
        finally:
            self.enabled = previous_enabled

    def snapshot(self, *, values_only: bool = True, clear: bool = False) -> Dict[str, Any]:
        if values_only:
            result = {name: item.value for name, item in self._items.items()}
        else:
            result = dict(self._items)
        if clear:
            self.clear()
        return result

    def summary(self, *, clear: bool = False) -> Dict[str, CapturedObservation]:
        """Return the observations captured for the current sampled step."""

        return self.snapshot(values_only=False, clear=clear)

    def clear_summary(self) -> None:
        self.clear()

    def __len__(self) -> int:
        return len(self._items)
````

## File: src/rosen/utils/optimize.py
````python
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Tuple, Union


DEFAULT_NO_DECAY_KEYWORDS = (
    "position_ids",
    # Free parameters with embedding-like semantics.  These are not always
    # owned by nn.Embedding, so module-type inspection alone is insufficient.
    "seed",
    "shift_begin",
    "shift_end",
    "boundary_flag",
    "_vectors",
)

def split_params_by_decay(
        named_params: Iterable[Tuple[str, nn.Parameter]],  # 修改这里
        weight_decay: float = 0.01,
        no_decay_keywords: Union[tuple, list] = ("embedding", "position_ids")
) -> List[Dict[str, Any]]:
    """
    基于张量维度和关键字自动分组优化器参数。
    """
    decay_params = []
    no_decay_params = []

    for name, param in named_params: # 修改这里
        if not param.requires_grad:
            continue

        if param.ndim < 2 or any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


def split_model_params_by_decay(
    model: nn.Module,
    weight_decay: float = 1.0e-6,
    no_decay_keywords: Union[tuple, list] = DEFAULT_NO_DECAY_KEYWORDS,
) -> List[Dict[str, Any]]:
    """Build optimizer groups while never decaying embedding-like parameters.

    Matrix/tensor weights receive ``weight_decay``. Biases, normalization
    parameters, scalar/vector parameters, parameters owned directly by an
    ``nn.Embedding`` module, and explicitly named embedding-like free
    parameters receive zero decay. A Linear nested under a container whose
    name happens to contain ``embedding`` is still decayed.

    Inspecting module ownership matters: an embedding table may be named
    ``lookup.weight`` and therefore cannot be identified reliably from the
    parameter name alone.
    """
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    lowered_keywords = tuple(str(keyword).lower() for keyword in no_decay_keywords)
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered_name = name.lower()
        no_decay = (
            parameter.ndim < 2
            or id(parameter) in embedding_parameter_ids
            or any(keyword in lowered_name for keyword in lowered_keywords)
        )
        (no_decay_params if no_decay else decay_params).append(parameter)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups

def create_complex_optimizer_groups(
        model: nn.Module,
        decay_dict: Dict[str, float],
        default_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    根据传入的 decay_dict 灵活配置不同网络层的前缀参数衰减。

    参数示例:
    decay_dict = {
        "embeddings": 0.0,       # 推荐系统 Embedding
        "moe_router": 0.05,      # 假设 MoE 门控网络需要更强的正则化
        "bias": 0.0,             # 偏置项
        "norm": 0.0              # 归一化层
    }
    """
    # 构建一个按 weight_decay 分组的字典
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 默认衰减值
        current_decay = default_decay

        # 遍历字典进行匹配（可以根据需求换成正则 re.search）
        for key, decay_value in decay_dict.items():
            if key in name.lower():
                current_decay = decay_value
                break  # 匹配到第一个优先级最高的就跳出

        # 归入对应的 decay 组
        if current_decay not in groups:
            groups[current_decay] = []
        groups[current_decay].append(param)

    # 转换为 Optimizer 需要的格式
    return [
        {"params": params, "weight_decay": wd}
        for wd, params in groups.items()
    ]
````

## File: src/rosen/utils/sequential.py
````python
from typing import Union, Tuple, Dict

import polars as pl
from typing import Dict, Optional, Any


def extract_history_sequences(
        lf: pl.LazyFrame,
        max_seq_len: int,
        user_col: str,
        time_col: str,
        feature_mapping: Dict[str, str],
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1
) -> pl.LazyFrame:
    """
    严格提取用户历史正向行为序列。避免负样本稀释，且绝对保证无数据泄漏（防穿越）。
    """
    if feature_mapping is None:
        feature_mapping = {"item_id": "item_seq"}

    # 1. 确保全局严格排序，并分配全局连续的物理行号作为“逻辑时钟”
    lf = lf.sort([user_col, time_col])
    lf = lf.with_row_index("_global_row_idx")

    # 2. 剥离出纯正样本的 DataFrame (继承了主表的 _global_row_idx)
    if label_col is not None:
        pos_lf = lf.filter(pl.col(label_col) == positive_label)
    else:
        pos_lf = lf

    # 在纯正样本中，为每个用户打上局部的连续序号 (0, 1, 2...)
    pos_lf = pos_lf.with_columns(
        pl.int_range(0, pl.len(), dtype=pl.UInt32).over(user_col).alias("_pos_idx")
    )

    # 3. 在纯正样本上进行严格的 N 长度开窗
    # 因为表中全都是正样本，所以开窗 max_seq_len 就能拿满 N 个正交互
    agg_exprs = [
        pl.col(src).alias(tgt_seq_name)
        for src, tgt_seq_name in feature_mapping.items()
    ]

    pos_rolling = pos_lf.rolling(
        index_column="_pos_idx",
        group_by=user_col,
        period=f"{max_seq_len}i"
    ).agg(agg_exprs)

    # 将生成的纯正序列和原有的“逻辑时钟”拼起来，且重命名避免冲突
    # pos_state 代表：在这个 _state_row_idx 时刻【之后】，用户的正向历史状态更新为了什么
    pos_state = pos_lf.select([user_col, "_global_row_idx", "_pos_idx"]).join(
        pos_rolling, on=[user_col, "_pos_idx"]
    ).rename({"_global_row_idx": "_state_row_idx"}).drop("_pos_idx")

    # 【关键修复】: 将 state_row_idx 转为有符号整数并严格排序
    pos_state = pos_state.with_columns(
        pl.col("_state_row_idx").cast(pl.Int64)
    ).sort("_state_row_idx")

    # 4. ASOF JOIN 拼回主表 (核心防泄漏机制)
    # 对于主表的每一行，我们要找的是严格发生在它【之前】的最新正向状态
    # 【关键修复】: 左表的 join_idx = 当前行号 - 1，必须先 cast 为 Int64 避免无符号 0 - 1 下溢
    lf = lf.with_columns(
        (pl.col("_global_row_idx").cast(pl.Int64) - 1).alias("_join_idx")
    )

    lf = lf.join_asof(
        pos_state,
        left_on="_join_idx",
        right_on="_state_row_idx",
        by=user_col,
        strategy="backward"  # 寻找 <= _join_idx 的最新状态
    )

    # 5. 清理与填补
    first_seq_col = list(feature_mapping.values())[0]
    drop_cols = ["_global_row_idx", "_join_idx", "_state_row_idx"]

    # 将匹配不到历史状态的 Null 填充为空列表 []
    fill_exprs = [
        pl.col(tgt).fill_null([]) for tgt in feature_mapping.values()
    ]

    lf = (
        lf.with_columns(fill_exprs)
        .with_columns(
            pl.col(first_seq_col).list.len().fill_null(0).cast(pl.UInt32).alias(seq_len_col)
        )
        .drop(drop_cols)
    )

    return lf

def extract_seq_len(
        lf: pl.LazyFrame,
        seq_col: str,
        seq_len_col: str = "seq_len"
) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col(seq_col)
        .list.len()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias(seq_len_col)
    )

def extract_history_items(
        lf: pl.LazyFrame,
        max_seq_len: int,
        user_col: str,
        time_col: str,
        item_col: str,
        seq_col: str = "items_seq",
        seq_len_col: str = "seq_len",
        label_col: Optional[str] = None,
        positive_label: Any = 1
) -> pl.LazyFrame:
    return extract_history_sequences(
        lf, max_seq_len, user_col, time_col,
        {item_col: seq_col}, seq_len_col,
        label_col, positive_label
    )




if __name__ == "__main__":
    def main():
        # 1. 构造模拟数据：两个用户，交易时间乱序（测试排序逻辑）
        data = {
            "user_id": [1, 2, 1, 1, 2, 1, 2],
            "timestamp": [100, 102, 101, 103, 200, 202, 201],
            "video_id": ["v_A", "v_C", "v_B", "v_D", "v_E", "v_G", "v_F"],
            "label": [0, 0, 1, 1, 1, 1, 1]
        }

        lf = pl.LazyFrame(data)

        # 2. 设置最大序列长度
        MAX_SEQ_LEN = 3

        print(f"--- 原始输入 (LazyFrame) ---")
        print(lf.collect())

        # 3. 调用你的提取函数
        # 逻辑预期：对于每个用户，当前行的 items_seq 应该是其【之前】的最多 3 个 video_id
        result_lf = extract_history_sequences(
            lf=lf,
            max_seq_len=MAX_SEQ_LEN,
            user_col="user_id",
            time_col="timestamp",
            label_col="label",
            feature_mapping={
                "video_id": "history_list"
            }
        )

        # 4. 执行并打印结果
        df_result = result_lf.collect()

        print(f"\n--- 处理后的结果 (max_seq_len={MAX_SEQ_LEN}) ---")
        # 按照用户和时间打印，方便观察序列增长
        print(df_result.sort(["user_id", "timestamp"]))

        # 5. 验证核心逻辑点
        print("\n--- 逻辑校验 ---")
        # 校验用户1的最后一行：时间103，前三行应为 [v_A, v_B, v_C]
        last_user1 = df_result.filter((pl.col("user_id") == 1) & (pl.col("timestamp") == 103))
        print(f"用户1(T=103) 的历史序列: {last_user1.get_column('history_list').to_list()}")

        # 校验用户2的第一行：历史序列应该是空的 []
        first_user2 = df_result.filter((pl.col("user_id") == 2) & (pl.col("timestamp") == 200))
        print(f"用户2(T=200) 的历史序列: {first_user2.get_column('history_list').to_list()}")
    main()
````

## File: src/rosen/utils/shuffle.py
````python
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, Literal, Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_input_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidates = [Path(paths)]
    else:
        candidates = [Path(path) for path in paths]

    resolved: list[Path] = []
    for path in candidates:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.parquet")))
        elif path.exists():
            resolved.append(path)

    if not resolved:
        raise FileNotFoundError("No input parquet files found for shuffle checkpoint.")
    return resolved


def _parquet_columns(paths: list[Path]) -> list[str]:
    return pq.ParquetFile(paths[0]).schema_arrow.names


def _filter_existing_fields(paths: list[Path], fields: Optional[Iterable[str]]) -> Optional[list[str]]:
    if fields is None:
        return None
    existing = set(_parquet_columns(paths))
    return [field for field in fields if field in existing]


def _empty_table(paths: list[Path], fields: Optional[list[str]]) -> pa.Table:
    schema = pq.ParquetFile(paths[0]).schema_arrow
    if fields is not None:
        schema = pa.schema([field for field in schema if field.name in fields])
    arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _append_parquet_batches(
    input_paths: list[Path],
    out_path: Path,
    *,
    batch_size: int,
    compression: str,
    rand_col: str,
    shuffle_rows: bool,
    seed: int,
) -> None:
    writer = None
    wrote_rows = False
    try:
        chunk_idx = 0
        for path in input_paths:
            parquet_file = pq.ParquetFile(path)
            for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
                df = pl.from_arrow(batch)
                if df.height == 0:
                    continue
                if shuffle_rows:
                    row_seed = (int(seed) + chunk_idx * 1000003 + batch_idx * 9176) & 0xFFFFFFFF
                    df = df.sample(fraction=1.0, shuffle=True, seed=row_seed)
                if rand_col in df.columns:
                    df = df.drop(rand_col)
                table = df.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
                writer.write_table(table)
                wrote_rows = True
            chunk_idx += 1
    finally:
        if writer is not None:
            writer.close()

    if not wrote_rows:
        table = _empty_table(input_paths, None)
        pq.write_table(table, out_path, compression=compression)


class _ParquetWriterGroup:
    def __init__(self, compression: str):
        self.compression = compression
        self._writers: dict[int, pq.ParquetWriter] = {}

    def write(self, key: int, path: Path, df: pl.DataFrame) -> int:
        if df.height == 0:
            return 0
        table = df.to_arrow()
        if key not in self._writers:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writers[key] = pq.ParquetWriter(str(path), table.schema, compression=self.compression)
        self._writers[key].write_table(table)
        return int(df.height)

    def close(self) -> None:
        first_error = None
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                first_error = first_error or exc
        self._writers.clear()
        if first_error is not None:
            raise first_error


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _hash_range_target_expr(rand_col: str, target_count: int) -> pl.Expr:
    if target_count == 1:
        return pl.lit(0, dtype=pl.UInt32)
    if _is_power_of_two(target_count):
        shift = 64 - (target_count.bit_length() - 1)
        step = 1 << shift
        return (pl.col(rand_col) // step).cast(pl.UInt32)
    return (
        ((pl.col(rand_col).cast(pl.Float64) / float(1 << 64)) * target_count)
        .floor()
        .clip(0, target_count - 1)
        .cast(pl.UInt32)
    )


def _write_target_sort_shuffled_parquet(
    input_paths: list[Path],
    out_path: Path,
    *,
    selected_fields: Optional[list[str]],
    filter_expr: Optional[pl.Expr],
    seed: int,
    target_count: int,
    batch_size: int,
    compression: str,
    temp_dir: str | Path | None,
) -> None:
    """External random-key shuffle.

    Rows are first range-partitioned by the high bits of a stable row hash, then
    each target is sorted by the full hash. Concatenating targets in target id
    order is equivalent to sorting the full dataset by the random key, without
    materializing the full table in memory.
    """

    rand_col = "__bb_shuffle_id"
    target_col = "__bb_shuffle_target"
    tmp_cols = ["__bb_file_idx", "__bb_row_idx", rand_col, target_col]
    existing_cols = set(_parquet_columns(input_paths))
    conflict_cols = sorted(existing_cols.intersection(tmp_cols))
    if conflict_cols:
        raise ValueError(f"Input parquet contains reserved shuffle columns: {conflict_cols}")

    tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    staging_root = tmp_dir / "stage"
    final_root = tmp_dir / "final"
    staging_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    staging_paths = [
        staging_root / f"part-{target_id:05d}.parquet"
        for target_id in range(target_count)
    ]
    read_columns = None if filter_expr is not None else selected_fields
    writers = _ParquetWriterGroup(compression=compression)

    try:
        for file_idx, path in enumerate(input_paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if filter_expr is not None:
                    df = df.filter(filter_expr)
                    if df.height == 0:
                        row_offset += batch.num_rows
                        continue
                    if selected_fields is not None:
                        df = df.select(selected_fields)
                elif selected_fields is not None:
                    df = df.select(selected_fields)

                if df.height > 0:
                    df = (
                        df
                        .with_row_index("__bb_row_idx", offset=row_offset)
                        .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                        .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
                        .with_columns(_hash_range_target_expr(rand_col, target_count).alias(target_col))
                    )

                    for key, target_df in df.partition_by(
                        target_col,
                        maintain_order=False,
                        include_key=True,
                        as_dict=True,
                    ).items():
                        target_id = int(key[0] if isinstance(key, tuple) else key)
                        writers.write(target_id, staging_paths[target_id], target_df)

                row_offset += batch.num_rows
            print(f"    -> shuffle targetize {file_idx + 1}/{len(input_paths)}")
    finally:
        writers.close()

    try:
        target_paths: list[Path] = []
        for target_id, staging_path in enumerate(staging_paths):
            if not staging_path.exists():
                continue
            final_path = final_root / f"part-{target_id:05d}.parquet"
            (
                pl.scan_parquet(staging_path)
                .sort(rand_col)
                .drop(tmp_cols)
                .sink_parquet(final_path, compression=compression)
            )
            target_paths.append(final_path)

        if out_path.exists():
            out_path.unlink()
        if target_paths:
            _append_parquet_batches(
                target_paths,
                out_path,
                batch_size=batch_size,
                compression=compression,
                rand_col=rand_col,
                shuffle_rows=False,
                seed=seed,
            )
        else:
            pq.write_table(_empty_table(input_paths, selected_fields), out_path, compression=compression)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_shuffled_parquet(
    input_paths: str | Path | Iterable[str | Path],
    out_path: str | Path,
    *,
    fields: Optional[Iterable[str]] = None,
    filter_expr: Optional[pl.Expr] = None,
    seed: int = 2026,
    bucket_count: int = 64,
    batch_size: int = 65536,
    compression: str = "zstd",
    redo: bool = True,
    exact: bool = False,
    mode: Literal["target_sort", "bucket_chunk"] = "target_sort",
    temp_dir: str | Path | None = None,
) -> Path:
    """Write a physically shuffled parquet checkpoint.

    The default target_sort mode performs an external random-key sort by
    range-partitioning rows into power-of-two targets, sorting each target by
    the stable row hash, and concatenating targets in key-range order.
    Set mode="bucket_chunk" to use the older low-memory chunk shuffle.
    Set exact=True only when a single in-memory/logical full sort is required.
    """

    paths = _resolve_input_paths(input_paths)
    out_path = Path(out_path)
    if out_path.exists() and not redo:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected_fields = _filter_existing_fields(paths, fields)
    rand_col = "__bb_rand"

    if exact:
        lf = pl.scan_parquet([str(path) for path in paths])
        if filter_expr is not None:
            lf = lf.filter(filter_expr)
        if selected_fields is not None:
            lf = lf.select(selected_fields)
        (
            lf
            .with_row_index("__bb_row_idx")
            .with_columns(pl.struct(["__bb_row_idx"]).hash(seed=seed).alias(rand_col))
            .sort(rand_col)
            .drop(["__bb_row_idx", rand_col])
            .sink_parquet(out_path, compression=compression)
        )
        return out_path

    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if mode not in {"target_sort", "bucket_chunk"}:
        raise ValueError("mode must be either 'target_sort' or 'bucket_chunk'.")

    if mode == "target_sort":
        _write_target_sort_shuffled_parquet(
            paths,
            out_path,
            selected_fields=selected_fields,
            filter_expr=filter_expr,
            seed=seed,
            target_count=bucket_count,
            batch_size=batch_size,
            compression=compression,
            temp_dir=temp_dir,
        )
        return out_path

    # TODO: This bucket/chunk path is not equivalent to a true global shuffle.
    # If upstream files are sorted by user/time and downstream training reads
    # output shards sequentially, large local order blocks can remain. Add a
    # round-robin/random shard reader or make exact hash-sort the default for
    # no-sequence datasets where the memory cost is acceptable.
    tmp_dir = Path(temp_dir) if temp_dir is not None else out_path.parent / f"_{out_path.stem}_shuffle_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    bucket_root = tmp_dir / "buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    for bucket_id in range(bucket_count):
        (bucket_root / f"b{bucket_id:03d}").mkdir(parents=True, exist_ok=True)

    file_counts = {bucket_id: 0 for bucket_id in range(bucket_count)}
    read_columns = None if filter_expr is not None else selected_fields

    try:
        for file_idx, path in enumerate(paths):
            row_offset = 0
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
                df = pl.from_arrow(batch)
                if filter_expr is not None:
                    df = df.filter(filter_expr)
                    if df.height == 0:
                        row_offset += batch.num_rows
                        continue
                    if selected_fields is not None:
                        df = df.select(selected_fields)

                df = (
                    df
                    .with_row_index("__bb_row_idx", offset=row_offset)
                    .with_columns(pl.lit(file_idx, dtype=pl.UInt32).alias("__bb_file_idx"))
                    .with_columns(pl.struct(["__bb_file_idx", "__bb_row_idx"]).hash(seed=seed).alias(rand_col))
                    .with_columns((pl.col(rand_col) % bucket_count).cast(pl.UInt32).alias("__bb_bucket"))
                )

                for key, bucket_df in df.partition_by(
                    "__bb_bucket",
                    maintain_order=False,
                    include_key=True,
                    as_dict=True,
                ).items():
                    bucket_id = int(key[0] if isinstance(key, tuple) else key)
                    chunk_path = (
                        bucket_root
                        / f"b{bucket_id:03d}"
                        / f"c{file_counts[bucket_id]:06d}.parquet"
                    )
                    (
                        bucket_df
                        .drop(["__bb_bucket", "__bb_file_idx", "__bb_row_idx"])
                        .write_parquet(chunk_path, compression=compression)
                    )
                    file_counts[bucket_id] += 1

                row_offset += batch.num_rows
            print(f"    -> shuffle bucketize {file_idx + 1}/{len(paths)}")

        rng = random.Random(seed)
        bucket_order = list(range(bucket_count))
        rng.shuffle(bucket_order)

        shuffled_chunks: list[Path] = []
        for bucket_id in bucket_order:
            bucket_files = sorted((bucket_root / f"b{bucket_id:03d}").glob("*.parquet"))
            rng.shuffle(bucket_files)
            shuffled_chunks.extend(bucket_files)

        if out_path.exists():
            out_path.unlink()
        if shuffled_chunks:
            _append_parquet_batches(
                shuffled_chunks,
                out_path,
                batch_size=batch_size,
                compression=compression,
                rand_col=rand_col,
                shuffle_rows=True,
                seed=seed,
            )
        else:
            pq.write_table(_empty_table(paths, selected_fields), out_path, compression=compression)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path


def shuffle_parquet_files(*args, **kwargs) -> Path:
    return write_shuffled_parquet(*args, **kwargs)
````

## File: src/rosen/utils/sort.py
````python
import os
import shutil

import duckdb
from typing import Union, List
from pathlib import Path


def duckdb_sort_parquet(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        sort_by: Union[str, List[str]],
        descending: Union[bool, List[bool]] = False,
        memory_limit: str = "84GB",  # 物理防爆盾：强制最高内存限制
        temp_dir: str = "./workspace/tmp"  # 磁盘防爆盾：防止把系统 C 盘写满
):
    input_path = str(input_path)
    output_path = str(output_path)

    # 确保安全的临时目录存在
    os.makedirs(temp_dir, exist_ok=True)

    # 统一转换 sort_by 和 descending 为列表，对齐 Polars 体验
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_by)
    elif len(descending) != len(sort_by):
        raise ValueError("descending 的长度必须与 sort_by 一致")

    # 构建 SQL 的 ORDER BY 语句
    order_clauses = []
    for col, desc in zip(sort_by, descending):
        direction = "DESC" if desc else "ASC"
        order_clauses.append(f'"{col}" {direction}')
    order_by_str = ", ".join(order_clauses)

    print(f"\n[DuckDB] 🚄 启动核外大表物理排序...")
    print(f"    -> 输入: {input_path}")
    print(f"    -> 输出: {output_path}")
    print(f"    -> 规则: ORDER BY {order_by_str}")
    print(f"    -> 资源限制: 内存不超过 {memory_limit}, 临时缓冲: {temp_dir}")

    # 建立 DuckDB 连接并注入资源防爆指令
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{temp_dir}'")

    # 执行查询并流式落盘，开启高压缩比
    query = f"""
        COPY (
            SELECT * FROM read_parquet('{input_path}')
            ORDER BY {order_by_str}
        ) TO '{output_path}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
    """
    con.execute(query)
    con.close()

def sort_parquet_inplace(
        file_path: Union[str, Path],
        sort_by: Union[str, List[str]],
        descending: Union[bool, List[bool]] = False,
        memory_limit: str = "64GB",
        temp_dir = "./tmp"
):
    """
    使用 DuckDB 对 Parquet 文件进行极低内存的“原地排序”。
    机制：将排序结果输出到 ./tmp 目录下，完成后直接覆盖原文件，绝对安全。
    """
    file_path = Path(file_path).absolute()
    if not file_path.exists():
        raise FileNotFoundError(f"找不到目标文件: {file_path}")

    # 创建安全的临时目录
    temp_dir_path = Path(temp_dir).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    temp_output_path = temp_dir_path / f"sorted_{file_path.name}"

    # 统一转换 sort_by 和 descending 为列表
    if isinstance(sort_by, str):
        sort_by = [sort_by]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_by)
    elif len(descending) != len(sort_by):
        raise ValueError("descending 的长度必须与 sort_by 一致")

    # 构建 SQL 的 ORDER BY 语句
    order_clauses = [
        f'"{col}" {"DESC" if desc else "ASC"}'
        for col, desc in zip(sort_by, descending)
    ]
    order_by_str = ", ".join(order_clauses)

    print(f"\n[DuckDB] 🚄 启动大表原地物理排序 (In-place Sort)...")
    print(f"    -> 目标文件: {file_path}")
    print(f"    -> 排序规则: ORDER BY {order_by_str}")
    print(f"    -> 资源限制: 内存 <= {memory_limit}, 临时引擎目录: {temp_dir_path}")

    try:
        # 建立 DuckDB 连接并注入资源限制
        con = duckdb.connect()
        # 注意：DuckDB 接收路径最好用 POSIX 格式 (特别是 Windows 下)，避免转义坑
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        con.execute(f"PRAGMA temp_directory='{temp_dir_path.as_posix()}'")

        # 执行查询并流式落盘到 temp_output_path
        query = f"""
            COPY (
                SELECT * FROM read_parquet('{file_path.as_posix()}')
                ORDER BY {order_by_str}
            ) TO '{temp_output_path.as_posix()}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000, COMPRESSION 'ZSTD');
        """
        con.execute(query)
        con.close()

        # ✨ 核心：安全地替换原文件
        # 使用 shutil.move 可以跨磁盘移动，并且覆盖原文件
        shutil.move(str(temp_output_path), str(file_path))
        print(f"[DuckDB] ✅ 排序完成！已成功覆写原文件: {file_path.name}\n")

    except Exception as e:
        print(f"\n[DuckDB] ❌ 排序过程中发生错误: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise e
````

## File: src/rosen/__init__.py
````python
"""rosen 顶层包。"""
````

## File: pyproject.toml
````toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "rosen"
version = "0.1.0"
description = "Explicit PyTorch framework for multi-scenario recommendation experiments"
readme = "README.md"
requires-python = ">=3.10,<3.12"
dependencies = [
    "torch>=2.0,<3", "numpy>=1.24,<2", "polars>=1.36.1,<2",
    "pyarrow>=21,<22", "pandas>=2,<3", "scipy>=1.11,<2",
    "scikit-learn>=1.6,<2", "duckdb>=1.5.4,<2", "psutil>=5.9,<8",
    "matplotlib>=3.7,<4", "pillow>=10,<12", "six>=1.16,<2",
]

[project.optional-dependencies]
vision = ["torchvision>=0.15,<1"]

[dependency-groups]
dev = ["pytest>=8,<9"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["rosen", "rosen.*"]
namespaces = false

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests", "workspace/kr_time_budget/tests"]
````

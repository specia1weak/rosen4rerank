"""Resolve labels once, then dispatch one EvalBatch contract to every evaluator."""
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from weakref import ref

import polars as pl
import torch

from rosen.core.prediction import Prediction, as_prediction
from rosen.core.interaction import Interaction
from .evaluator import BaseEvaluator, PointwiseEvaluator, RetrievalEvaluator, SlateValueEvaluator
from .resolver import BatchResolver
from .scorer import EvalContext


class EvaluatorFilter(ABC):
    """Pure row selector; registrations sharing one instance reuse its result."""
    @abstractmethod
    def __call__(self, inter):
        raise NotImplementedError


class AllowAllFilter(EvaluatorFilter):
    def __call__(self, inter):
        return True


class DomainFilter(EvaluatorFilter):
    def __init__(self, field_name: str, domain_id: int):
        self.field_name, self.domain_id = field_name, domain_id

    def __call__(self, inter):
        return inter[self.field_name] == self.domain_id


@dataclass(frozen=True)
class _EvalTask:
    evaluator: BaseEvaluator
    filter_fn: Callable[[Optional[Interaction]], bool | torch.Tensor]
    target: Optional[str]
    uid_field: Optional[str]
    resolver: BatchResolver


def _default_view_for(evaluator):
    if isinstance(evaluator, SlateValueEvaluator):
        return "slate"
    if isinstance(evaluator, (PointwiseEvaluator, RetrievalEvaluator)):
        return "score_only"
    return "ranking"


class EvaluatorManager:
    def __init__(self, manager=None, log_path=None, title="untitled", *, uid_field=None):
        self.schema_manager = manager
        self.uid_field = uid_field
        self.registry: dict[str, _EvalTask] = {}
        self.log_path, self.title = log_path, title
        self._is_first_summary = True
        self._allow_all = AllowAllFilter()
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def bind(self, manager):
        """Trainer binding must not silently replace an explicitly supplied schema."""
        if self.schema_manager is not None and self.schema_manager is not manager:
            raise ValueError("EvaluatorManager is already bound to a different SchemaManager.")
        self.schema_manager = manager

    def register(self, name, evaluator: BaseEvaluator, filter_fn=None, *, target=None, uid_field=None, resolver=None):
        if not isinstance(evaluator, BaseEvaluator):
            raise TypeError("Evaluators must inherit BaseEvaluator and implement collect(EvalBatch).")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError("target must be a non-empty label name.")
        if resolver is not None and not isinstance(resolver, BatchResolver):
            raise TypeError("resolver must inherit BatchResolver")

        default_view = _default_view_for(evaluator)
        self.registry[name] = _EvalTask(
            evaluator=evaluator,
            filter_fn=filter_fn if filter_fn is not None else self._allow_all,
            target=target,
            uid_field=uid_field,
            resolver=resolver if resolver is not None else BatchResolver(view=default_view),
        )

    def _filter_rows(self, filter_fn: Callable, inter, pred: Prediction):
        """Apply row-level filter and return sliced (inter, pred), or (None, None) if empty."""
        row_mask = filter_fn(inter)
        if isinstance(row_mask, bool):
            return (inter, pred) if row_mask and pred.batch_size > 0 else (None, None)

        if not isinstance(row_mask, torch.Tensor) or row_mask.dtype != torch.bool or row_mask.shape != (pred.batch_size,):
            raise ValueError("Evaluator filters must return bool or a boolean Tensor[B].")
        if not row_mask.any():
            return None, None

        selected_inter = inter[row_mask] if inter is not None else None
        selected_pred = pred[row_mask]
        return (selected_inter, selected_pred) if selected_pred.batch_size > 0 else (None, None)

    def _resolve_target(self, pred: Prediction, requested: Optional[str], resolver: BatchResolver) -> Optional[str]:
        """Infer target label name and validate against Prediction and SchemaManager."""
        target = requested if requested is not None else pred.target
        labels = getattr(self.schema_manager, "label_fields", ())
        if target is None and len(labels) == 1:
            target = labels[0]

        if not resolver.needs_target(pred):
            return target

        if target is None:
            if len(labels) > 1:
                raise ValueError("Ambiguous evaluation target: set register(target=...) or Prediction.target.")
            if (isinstance(pred.scores, Mapping) or isinstance(pred.order, Mapping)
                    or isinstance(pred.chosen_scores, Mapping)):
                raise ValueError("Mapping predictions require an explicit target; keys are never guessed.")

        if requested is not None and pred.target is not None and requested != pred.target:
            if resolver.uses_prediction_target(pred):
                raise ValueError(f"Prediction target {pred.target!r} conflicts with registered target {requested!r}.")

        if resolver.uses_schema_labels and labels and target is not None and target not in labels:
            raise KeyError(f"Target {target!r} is not declared in SchemaManager.label_fields.")

        return target

    def _resolve_mask(self, inter, target: Optional[str], resolver: BatchResolver) -> Optional[torch.Tensor]:
        if inter is None:
            return None
        get_spec = getattr(self.schema_manager, "get_label_spec", None)
        labels = getattr(self.schema_manager, "label_fields", ())
        spec = get_spec(target) if get_spec is not None and target in labels else None
        mask_field = resolver.mask_field or (spec.mask_field if spec is not None else None)
        return inter[mask_field] if mask_field is not None else None

    def _resolve_users(self, inter, task_uid_field: Optional[str]) -> Optional[torch.Tensor]:
        if inter is None:
            return None
        explicit_uid = task_uid_field or self.uid_field
        if explicit_uid is not None:
            return inter[explicit_uid]
        schema_uid = getattr(self.schema_manager, "uid_field", None)
        return inter[schema_uid] if schema_uid is not None and schema_uid in inter else None

    def collect(self, inter, prediction):
        pred = as_prediction(prediction)
        if inter is not None and len(inter) != pred.batch_size:
            raise ValueError("Interaction and Prediction must share the batch dimension.")

        filtered = {}
        cpu_tensors = {}

        def to_cpu(tensor):
            if tensor is None:
                return None
            key = id(tensor)
            cached = cpu_tensors.get(key)
            if cached is None or cached[0]() is not tensor:
                # Check identity without retaining temporary judge tensors on the GPU.
                cpu_tensors[key] = ref(tensor), tensor.detach().cpu()
            return cpu_tensors[key][1]

        for task in self.registry.values():
            filter_key = id(task.filter_fn)
            if filter_key not in filtered:
                filtered[filter_key] = self._filter_rows(task.filter_fn, inter, pred)
            selected_inter, selected_pred = filtered[filter_key]
            if selected_pred is None:
                continue

            target = self._resolve_target(selected_pred, task.target, task.resolver)
            mask = self._resolve_mask(selected_inter, target, task.resolver)
            users = self._resolve_users(selected_inter, task.uid_field)

            context = EvalContext(selected_pred, selected_inter, target)
            batch = task.resolver.resolve(context, mask=mask, users=users, to_cpu=to_cpu)
            task.evaluator.collect(batch)

    def summary(self, epoch=None, step=None, *, stage=None):
        results = {name: dict(task.evaluator.summary()) for name, task in self.registry.items()}
        if self.log_path:
            flat_metrics = {
                f"{name}.{metric}": value
                for name, values in results.items()
                for metric, value in values.items()
            }
            self._write_polars_log(flat_metrics, epoch, step, stage=stage)
        return results

    def _write_polars_log(self, metrics_dict, epoch, step, *, stage=None):
        """内部方法：专门负责将指标转成 Polars 字符串并追加到文件"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pl.DataFrame({
            "Metric": list(metrics_dict.keys()),
            "Value": list(metrics_dict.values()),
        })

        with pl.Config(tbl_rows=100, tbl_hide_dataframe_shape=True):
            table_str = str(df)

        with open(self.log_path, "a", encoding="utf-8") as f:
            if self._is_first_summary:
                f.write(f"\n========== 🚀 [{self.title}] 开始记录 | {current_time} ==========\n")
                self._is_first_summary = False

            header_parts = [f"\n[{current_time}]"]
            if stage is not None:
                header_parts.append(f"Stage: {stage}")
            if epoch is not None:
                header_parts.append(f"Epoch: {epoch:02d}")
            if step is not None:
                header_parts.append(f"Step: {step}")

            f.write(" | ".join(header_parts) + "\n")
            f.write(table_str + "\n")

    def clear(self):
        for task in self.registry.values():
            task.evaluator.clear()

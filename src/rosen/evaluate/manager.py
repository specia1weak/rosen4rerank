"""Resolve targets once, then dispatch one EvalBatch contract to every evaluator."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import polars as pl
import torch

from rosen.core.prediction import Prediction, as_prediction
from .batch import EvalBatch
from .evaluator import BaseEvaluator


class EvaluatorFilter(ABC):
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


class EvaluatorManager:
    def __init__(self, manager=None, log_path=None, title="untitled", *, uid_field=None):
        self.schema_manager = manager
        self.uid_field = uid_field
        self.registry = {}
        self.log_path, self.title = log_path, title
        self._is_first_summary = True
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def bind(self, manager):
        """Trainer binding must not silently replace an explicitly supplied schema."""
        if self.schema_manager is not None and self.schema_manager is not manager:
            raise ValueError("EvaluatorManager is already bound to a different SchemaManager.")
        self.schema_manager = manager

    def register(self, name, evaluator: BaseEvaluator, filter_fn=None, *, target=None, uid_field=None):
        if not isinstance(evaluator, BaseEvaluator):
            raise TypeError("Evaluators must inherit BaseEvaluator; wrap old objects with LegacyEvaluatorAdapter.")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise ValueError("target must be a non-empty label name.")
        self.registry[name] = dict(evaluator=evaluator, filter_fn=filter_fn or AllowAllFilter(),
                                   target=target, uid_field=uid_field)

    def _target(self, pred, requested):
        target = requested if requested is not None else pred.target
        labels = getattr(self.schema_manager, "label_fields", ())
        if target is None:
            if len(labels) == 1:
                target = labels[0]
            elif len(labels) > 1:
                raise ValueError("Ambiguous evaluation target: set register(target=...) or Prediction.target.")
            elif isinstance(pred.scores, Mapping) or isinstance(pred.order, Mapping):
                raise ValueError("Mapping predictions require an explicit target; keys are never guessed.")
        if requested is not None and pred.target is not None and requested != pred.target:
            if isinstance(pred.scores, torch.Tensor) or isinstance(pred.order, torch.Tensor) or pred.full_scores is not None:
                raise ValueError(f"Prediction target {pred.target!r} conflicts with registered target {requested!r}.")
        if labels and target is not None and target not in labels:
            raise KeyError(f"Target {target!r} is not declared in SchemaManager.label_fields.")
        return target

    def collect(self, inter, prediction):
        pred = as_prediction(prediction)
        if inter is not None and len(inter) != pred.batch_size:
            raise ValueError("Interaction and Prediction must share the batch dimension.")
        for config in self.registry.values():
            row_mask = config["filter_fn"](inter)
            if isinstance(row_mask, bool):
                if not row_mask:
                    continue
                selected_inter, selected_pred = inter, pred
            else:
                if not isinstance(row_mask, torch.Tensor) or row_mask.dtype != torch.bool or row_mask.shape != (pred.batch_size,):
                    raise ValueError("Evaluator filters must return bool or a boolean Tensor[B].")
                if not row_mask.any():
                    continue
                selected_inter = inter[row_mask] if inter is not None else None
                selected_pred = pred[row_mask]
            if selected_pred.batch_size == 0:
                continue
            target = self._target(selected_pred, config["target"])
            def slot(value):
                return value[target] if isinstance(value, Mapping) else value
            scores, order = slot(selected_pred.scores), slot(selected_pred.order)
            targets = selected_inter[target] if selected_inter is not None and target is not None else None
            mask = None
            get_spec = getattr(self.schema_manager, "get_label_spec", None)
            spec = get_spec(target) if get_spec is not None and target is not None else None
            if spec is not None and spec.mask_field and selected_inter is not None:
                mask = selected_inter[spec.mask_field]
            uid_field = config["uid_field"] or self.uid_field
            if uid_field is not None:
                users = selected_inter[uid_field] if selected_inter is not None else None
            else:
                uid_field = getattr(self.schema_manager, "uid_field", None)
                users = (selected_inter[uid_field] if selected_inter is not None and uid_field is not None
                         and uid_field in selected_inter else None)
            config["evaluator"].collect(EvalBatch(pred=selected_pred, inter=selected_inter, scores=scores,
                                                 order=order, targets=targets, mask=mask, users=users))

    def summary(self, epoch=None, step=None):
        results = {name: dict(config["evaluator"].summary()) for name, config in self.registry.items()}
        if self.log_path:
            self._write_polars_log({f"{name}.{metric}": value for name, values in results.items()
                                    for metric, value in values.items()}, epoch, step)
        return results

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

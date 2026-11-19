"""KR 逐刷继续预测训练、最终 Test 评估及开场预算命中率。

从原始 CSV 重新生成数据并训练（不需要任何历史 workspace）：
uv run python -m experiments.kuairand.run_nxt_continue --source_dir /path/to/KuaiRand-1K/data --experiment_name raw_weak_epoch1 --device cpu
已有预处理数据时，省略 source_dir 并用 data_dir 指定目录。
uv run python -m experiments.kuairand.run_nxt_continue --experiment_name structured_epoch1 --structured True
uv run python -m experiments.kuairand.run_nxt_continue --experiment_name single_epoch1 --num_components 1
uv run python -m experiments.kuairand.run_nxt_continue --experiment_name fusion_epoch1 --model_type fusion

同 seed 的普通/结构化 K=2 对照保持共有参数初值与训练口径相同。
模型由 valid AUC 选 checkpoint；Test 只在选择结束后评估。
预算评估取每场第一刷的参数，真实长度来自终屏；全局中心只从 Train 拟合。
默认对齐旧0.7基线：当前刷四反馈+pace、UID=0、弱双辅助合计<=0.1*joint，主Loss回传两塔。
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np
import polars as pl
import torch

from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainerWithLife
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import DomainFilter, EvaluatorManager
from rosen.experiment import change_root_workdir
from rosen.experiment.param import ConfigBase, ParamManager, seed_everything
from rosen.utils.optimize import split_model_params_by_decay

from experiments.kuairand.nxt_continue_v1 import (
    DEFAULT_DATA_DIR, FusionExitModel, TimeBudgetExitModel, WeibullHead, load_data, prepare_data,
)


@dataclass
class ExitExperimentConfig(ConfigBase):
    dataset_name: str = "kr_nxt_continue_v1"
    data_dir: str = str(DEFAULT_DATA_DIR)
    source_dir: str = ""
    run_root: str = "workspace/kr_time_budget/runs/nxt_continue_v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs: int = 1
    batch_size: int = 4096
    num_threads: int = 4
    train_log_interval: int = 100
    model_type: Literal["ours", "fusion"] = "ours"
    num_components: int = 2
    structured: bool = False
    coupled_bathtub: bool = True
    add_pace: bool = True
    zero_uid: bool = True
    left_weight: float = 0.1
    right_weight: float = 0.1
    detach_budget_in_joint: bool = False
    max_aux_ratio: float = 0.1
    budget_lr: float = 5e-3
    base_lr: float = 1e-3
    weight_decay: float = 1e-6


def build_optimizer(model, cfg: ExitExperimentConfig):
    budget_ids = ({id(p) for m in model.budget_modules for p in m.parameters()}
                  if isinstance(model, TimeBudgetExitModel) else set())
    groups = []
    for group in split_model_params_by_decay(model, weight_decay=cfg.weight_decay):
        for is_budget, lr in ((True, cfg.budget_lr), (False, cfg.base_lr)):
            params = [p for p in group["params"] if (id(p) in budget_ids) == is_budget]
            if params:
                groups.append({"params": params, "weight_decay": group["weight_decay"], "lr": lr})
    return torch.optim.AdamW(groups)


@torch.no_grad()
def discrete_quantiles(head: WeibullHead, raw: torch.Tensor, *, component: int | None = None) -> torch.Tensor:
    """返回离散 L 的 Q25/Q50/Q75；整数二分，无 200 屏截断或插值偏差。"""
    scale, shape, log_pi = head._parse(raw.repeat_interleave(3, dim=0))
    threshold = raw.new_tensor([0.75, 0.50, 0.25], dtype=torch.float64).log().repeat(len(raw))

    def log_survival(length):
        hazard, _ = head._effective_hazard_and_increment(scale, shape, length, torch.zeros_like(length))
        return -hazard[:, component] if component is not None else torch.logsumexp(log_pi - hazard, dim=-1)

    low = torch.zeros(len(threshold), dtype=torch.int64, device=raw.device)
    high = torch.ones_like(low)
    while True:
        expand = log_survival(high) > threshold
        if not expand.any():
            break
        if (high[expand] >= 2**40).any():
            raise FloatingPointError("Budget quantile exceeds 2**40 screens")
        high = torch.where(expand, high * 2, high)
    while (low < high).any():
        middle = (low + high) // 2
        go_right = log_survival(middle) > threshold
        low = torch.where(go_right, middle + 1, low)
        high = torch.where(go_right, high, middle)
    return low.reshape(len(raw), 3)


def _hit(lengths: np.ndarray, center: np.ndarray) -> np.ndarray:
    return ((lengths >= np.minimum(center / 1.6, center - 2.0))
            & (lengths <= np.maximum(center * 1.6, center + 2.0)))


@torch.no_grad()
def evaluate_budget_hit_rate(model, loader, device, out_dir: Path, *, train_baseline: dict) -> dict:
    """每场一票、开场参数；两个中心均为真实分量中位数，使用同一命中窗口。"""
    model.eval()
    starts, terminals = [], []
    for batch in loader:
        batch = batch.to(device)
        first, terminal = batch["req_idx_raw"] == 1, batch["y_continue"] == 0
        if terminal.any():
            terminals.append(pl.DataFrame({
                "session_id": batch["session_id"][terminal].cpu().numpy(),
                "true_L": batch["req_idx_raw"][terminal].cpu().numpy(),
            }))
        if not first.any():
            continue
        opening = batch[first]
        raw = model.step_head.raw(model.budget_encoder(model.budget_view(opening)))
        params = model.step_head.survival(raw, torch.zeros_like(raw[:, 0]))
        quantiles = discrete_quantiles(model.step_head, raw).cpu().numpy()
        center0 = discrete_quantiles(model.step_head, raw, component=0)[:, 1].cpu().numpy()
        center1 = discrete_quantiles(model.step_head, raw, component=model.num_components - 1)[:, 1].cpu().numpy()
        scale, shape, prior = (v.cpu().numpy() for v in (params.scale, params.shape, params.prior))
        starts.append(pl.DataFrame({
            "session_id": opening["session_id"].cpu().numpy(),
            "user_id": opening["user_id_raw"].cpu().numpy(),
            "request_id": opening["request_id"].cpu().numpy(),
            "pred_r25": quantiles[:, 0], "pred_r50": quantiles[:, 1], "pred_r75": quantiles[:, 2],
            "component_median0": center0, "component_median1": center1,
            "scale0": scale[:, 0], "shape0": shape[:, 0], "scale1": scale[:, -1], "shape1": shape[:, -1], "pi0": prior[:, 0],
        }))
    if not starts or not terminals:
        raise ValueError("Budget evaluation requires complete sessions")
    opening, ending = pl.concat(starts), pl.concat(terminals)
    frame = opening.join(ending, on="session_id", validate="1:1").sort("session_id")
    if frame.height != opening.height or frame.height != ending.height:
        raise ValueError("Opening and terminal session IDs must match exactly")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_dir / "test_budget_centers.parquet")
    length = frame["true_L"].to_numpy()
    center = frame["pred_r50"].to_numpy()
    global_center = np.full(len(frame), train_baseline["median"])
    summary = {
        "n_sessions": frame.height,
        "train_global_median": train_baseline["median"],
        "global_baseline_hit_rate_1.6x": float(_hit(length, global_center).mean()),
        "ours_single_center_hit_rate_1.6x": float(_hit(length, center).mean()),
        "ours_two_center_hit_rate_1.6x": float((
            _hit(length, frame["component_median0"].to_numpy()) | _hit(length, frame["component_median1"].to_numpy())
        ).mean()),
        "ours_iqr_50pct_band_coverage": float(((length >= frame["pred_r25"].to_numpy()) & (length <= frame["pred_r75"].to_numpy())).mean()),
        "ours_iqr_mean_width": float((frame["pred_r75"] - frame["pred_r25"]).mean()),
        "shape1_gt_1_ratio": float((frame["shape1"] > 1).mean()),
    }
    protocol = {
        "prediction_time": "first request, before session length is known",
        "clock": "discrete screen count L; each test session has equal weight",
        "quantiles": "smallest integer r with S(r) <= 1-q; no fixed upper grid",
        "hit_window": "[min(c/1.6,c-2), max(1.6*c,c+2)]; same for model and train-only global baseline",
        "two_centers": "component medians; coupled component 0 uses exp(-H0-H1), component 1 uses exp(-H1)",
        "coverage": "inclusive discrete Q25..Q75; atoms can make coverage exceed 50%",
        "interpretation": "two-center HR has more coverage capacity than single-center HR; not a capacity-matched GMM comparison",
        "shape_constraint": "structured shape1>1 is imposed, not evidence of a learned second mode",
    }
    (out_dir / "budget_summary.json").write_text(json.dumps({"metrics": summary, "protocol": protocol}, indent=2) + "\n")
    print("BUDGET", json.dumps(summary), flush=True)
    return summary


def _state_hash(model, *, shared_only=False):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if shared_only and name.startswith("step_head."):
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_experiment(cfg: ExitExperimentConfig) -> dict:
    if cfg.extras:
        raise ValueError(f"Unknown experiment parameters: {sorted(cfg.extras)}")
    if cfg.model_type not in ("ours", "fusion"):
        raise ValueError("model_type must be ours or fusion")
    if cfg.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)
    run_dir = Path(cfg.run_root) / cfg.experiment_name
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.ckpt_dir = str(run_dir / "checkpoints")
    print("运行日志：", run_dir / "logs/train.log", flush=True)
    (run_dir / "logs").mkdir()
    with (run_dir / "logs/train.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print(cfg, flush=True)
        start_time = time.monotonic()
        data_dir = Path(cfg.data_dir)
        if cfg.source_dir:
            print("从原始 CSV 生成数据：", cfg.source_dir, "->", data_dir, flush=True)
            prepare_data(cfg.source_dir, data_dir, seed=cfg.seed)
        manager, loaders = load_data(data_dir, batch_size=cfg.batch_size, zero_uid=cfg.zero_uid)
        manifest_path = data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        baseline = pl.scan_parquet(data_dir / "train.parquet").filter(pl.col("y_continue") == 0).select(
            pl.col("req_idx_raw").median().alias("median")
        ).collect().row(0, named=True)
        req_per_sess = manifest["splits"]["train"]["requests_per_session"]
        if cfg.model_type == "fusion":
            model = FusionExitModel(manager, add_pace=cfg.add_pace)
        else:
            model = TimeBudgetExitModel(
                manager, num_components=cfg.num_components, step_only=True, add_pace=cfg.add_pace,
                left_weight=cfg.left_weight, right_weight=cfg.right_weight, requests_per_session=req_per_sess,
                structured=cfg.structured, coupled_bathtub=cfg.coupled_bathtub,
                detach_budget_in_joint=cfg.detach_budget_in_joint,
                max_aux_ratio=cfg.max_aux_ratio,
            )
        initial_hash = _state_hash(model)
        shared_hash = _state_hash(model, shared_only=True)
        evaluator = EvaluatorManager(log_path=run_dir / "metrics.csv", title=cfg.experiment_name, uid_field="user_id_raw")
        evaluator.register("overall", PointwiseEvaluator("auc", "logloss", "gauc"))
        domains = pl.scan_parquet(data_dir / "train.parquet").select(pl.col("tab").unique().sort()).collect()["tab"].to_list()
        for domain in domains:
            evaluator.register(f"tab_domain{domain}", PointwiseEvaluator("auc", "logloss"), filter_fn=DomainFilter("tab", domain))
        optimizer = build_optimizer(model, cfg)
        trainer = BaseTrainerWithLife(
            model, optimizer, manager,
            TrainerDataLoaders(train=loaders["train"], valid=loaders["valid"], test=loaders["test"]),
            TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper()), cfg,
        )
        trainer.run()
        if trainer.best_checkpoint_path is None:
            raise RuntimeError("No checkpoint selected by validation")
        trained_steps = trainer.global_step
        selected = trainer.load_checkpoint(trainer.best_checkpoint_path)
        test_metrics = trainer.evaluate_test()
        budget = (evaluate_budget_hit_rate(model, loaders["test"], cfg.device, run_dir / "audit", train_baseline=baseline)
                  if isinstance(model, TimeBudgetExitModel) else None)
        result = {
            "config": asdict(cfg), "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "data": manifest["splits"], "initial_state_sha256": initial_hash,
            "initial_shared_state_sha256": shared_hash,
            "selected_checkpoint": str(trainer.best_checkpoint_path), "selected_epoch": selected["epoch"],
            "selected_step": selected["global_step"], "trained_optimizer_steps": trained_steps,
            "valid": selected["metrics"], "test": test_metrics, "budget": budget,
            "weight_decay_policy": "split_model_params_by_decay; embeddings, bias, normalization and free vectors excluded",
            "elapsed_seconds": time.monotonic() - start_time,
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT", json.dumps(result), flush=True)
    print("实验完成：", run_dir / "result.json", flush=True)
    return result


if __name__ == "__main__":
    change_root_workdir()
    run_experiment(ParamManager(ExitExperimentConfig).build(experiment_name="raw_weak_epoch1"))

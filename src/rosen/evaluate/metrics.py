import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import kendalltau

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
        return log_loss(y_true, y_pred, labels=[0, 1])

class GAUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, users, **kwargs):
        labels, scores, users = np.asarray(y_true), np.asarray(y_pred), np.asarray(users)
        if labels.ndim != 1 or labels.shape != scores.shape or labels.shape != users.shape:
            raise ValueError("GAUC requires matching vectors of labels, scores and users")
        if not len(labels):
            return 0.0
        # Group users, then sort scores; average tied ranks within each user.
        order = np.lexsort((scores, users))
        labels, scores, users = labels[order], scores[order], users[order]
        starts = np.r_[True, users[1:] != users[:-1]]
        groups = starts.cumsum() - 1
        offsets = np.flatnonzero(starts)
        sizes = np.diff(np.r_[offsets, len(labels)])
        ranks = np.arange(len(labels)) - offsets[groups] + 1.0
        ties = np.r_[True, starts[1:] | (scores[1:] != scores[:-1])]
        tie_offsets = np.flatnonzero(ties)
        tie_sizes = np.diff(np.r_[tie_offsets, len(labels)])
        ranks = np.repeat(np.add.reduceat(ranks, tie_offsets) / tie_sizes, tie_sizes)
        positives = np.bincount(groups, weights=labels, minlength=len(sizes))
        negatives = sizes - positives
        rank_sums = np.bincount(groups, weights=ranks * labels, minlength=len(sizes))
        legal = (positives > 0) & (negatives > 0)
        if not legal.any():
            return 0.0
        aucs = (rank_sums[legal] - positives[legal] * (positives[legal] + 1) / 2) / (positives[legal] * negatives[legal])
        return float(np.average(aucs, weights=sizes[legal]))

# ==================== TopK 指标簇 ====================
class HR(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, **kwargs):
        # hits_k shape: [Batch_size, K]
        return hits_k.any(dim=-1).float().sum().item()

class NDCG(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, k, **kwargs):
        # 针对 Leave-One-Out 的极速 NDCG 计算
        device = hits_k.device
        ranks = torch.arange(1, k + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)
        return (hits_k * dcg_weights).sum().item()


def _validate_list_inputs(relevance, topk_indices, mask=None, k=None):
    if relevance.ndim != 2 or not relevance.shape[1]:
        raise ValueError("List NDCG requires relevance [B,N] with N > 0")
    if (topk_indices.ndim != 2 or topk_indices.dtype != torch.long
            or topk_indices.shape[0] != relevance.shape[0]
            or not 1 <= topk_indices.shape[1] <= relevance.shape[1]):
        raise ValueError("topk_indices must be LongTensor[B,K], 1 <= K <= N")
    indices = topk_indices.to(relevance.device)
    sorted_indices = indices.sort(-1).values
    if ((sorted_indices[:, 0] < 0).any() or (sorted_indices[:, -1] >= relevance.shape[1]).any()
            or (sorted_indices[:, 1:] <= sorted_indices[:, :-1]).any()):
        raise ValueError("topk_indices must be a partial permutation: unique indices in [0,N)")
    k = indices.shape[1] if k is None else k
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("NDCG k must be positive")
    if mask is not None and mask.shape != relevance.shape:
        raise ValueError("NDCG mask must match relevance")
    valid = mask > 0 if mask is not None else torch.ones_like(relevance, dtype=torch.bool)
    return indices, valid, k


class ListNDCG(BaseMetric):
    """NDCG from unified selected indices, with IDCG over the full candidate pool."""
    @classmethod
    def calculate(cls, relevance, topk_indices, mask=None, *, k=None):
        indices, valid, k = _validate_list_inputs(relevance, topk_indices, mask, k)
        gains = torch.where(valid, 2. ** relevance.double() - 1., 0.)
        discount = 1. / torch.log2(torch.arange(min(k, relevance.shape[1]), device=gains.device).double() + 2)
        size = min(k, indices.shape[1])
        dcg = (gains.gather(1, indices[:, :size]) * discount[:size]).sum(-1)
        idcg = (gains.sort(dim=-1, descending=True).values[:, :len(discount)] * discount).sum(-1)
        return torch.where(idcg > 0, dcg / idcg.clamp_min(torch.finfo(torch.float64).tiny), 0.)


class CatalogMRR(BaseMetric):
    @classmethod
    def calculate(cls, hits_k, **kwargs):
        ranks = torch.arange(1, hits_k.shape[1] + 1, device=hits_k.device).double()
        return (hits_k / ranks).sum().item()


class ListRecall(BaseMetric):
    @classmethod
    def calculate(cls, relevance, topk_indices, mask=None, *, k=None):
        indices, valid, k = _validate_list_inputs(relevance, topk_indices, mask, k)
        positive = (relevance > 0) & valid
        total = positive.sum(-1)
        hits = positive.gather(1, indices[:, :k]).sum(-1).double()
        return torch.where(total > 0, hits / total.clamp_min(1), torch.nan)


class ListHitRate(BaseMetric):
    @classmethod
    def calculate(cls, relevance, topk_indices, mask=None, *, k=None):
        indices, valid, k = _validate_list_inputs(relevance, topk_indices, mask, k)
        positive = (relevance > 0) & valid
        hits = positive.gather(1, indices[:, :k]).any(-1).double()
        return torch.where(positive.any(-1), hits, torch.nan)


class ListPrecision(BaseMetric):
    """Positive fraction among valid selected positions (not a fixed-k denominator)."""
    @classmethod
    def calculate(cls, relevance, topk_indices, mask=None, *, k=None):
        indices, valid, k = _validate_list_inputs(relevance, topk_indices, mask, k)
        selected = indices[:, :k]
        count = valid.gather(1, selected).sum(-1)
        hits = ((relevance > 0) & valid).gather(1, selected).sum(-1).double()
        return torch.where(count > 0, hits / count.clamp_min(1), torch.nan)


class ListMRR(BaseMetric):
    @classmethod
    def calculate(cls, relevance, topk_indices, mask=None, *, k=None):
        indices, valid, k = _validate_list_inputs(relevance, topk_indices, mask, k)
        positive = (relevance > 0) & valid
        selected = positive.gather(1, indices[:, :k])
        first = selected.long().argmax(-1).double() + 1
        rr = torch.where(selected.any(-1), first.reciprocal(), 0.)
        return torch.where(positive.any(-1), rr, torch.nan)


class ListMAP(BaseMetric):
    @classmethod
    def calculate(cls, relevance, topk_indices, mask=None, *, k=None):
        indices, valid, k = _validate_list_inputs(relevance, topk_indices, mask, k)
        positive = (relevance > 0) & valid
        selected = positive.gather(1, indices[:, :k]).double()
        ranks = torch.arange(1, selected.shape[1] + 1, device=selected.device).double()
        ap = ((selected.cumsum(-1) / ranks) * selected).sum(-1)
        count = positive.sum(-1).clamp(max=k)
        return torch.where(count > 0, ap / count.clamp_min(1), torch.nan)


class InSlateAUC(BaseMetric):
    """One AUC per row; undefined single-class rows are NaN.

    Average tied ranks after a per-row sort: O(B K log K), O(B K) memory.
    """
    @classmethod
    def calculate(cls, relevance, scores, mask=None):
        if relevance.ndim != 2 or scores.shape != relevance.shape or not scores.shape[1]:
            raise ValueError("in_slate_auc requires matching non-empty [B,K] inputs")
        if mask is not None and mask.shape != scores.shape:
            raise ValueError("mask must match slot scores")
        valid = mask > 0 if mask is not None else torch.ones_like(scores, dtype=torch.bool)
        if not torch.isin(relevance[valid], relevance.new_tensor([0, 1])).all():
            raise ValueError("in_slate_auc requires binary labels")
        if not torch.isfinite(scores[valid]).all():
            raise ValueError("Valid scores must be finite")
        values, order = scores.double().masked_fill(~valid, torch.inf).sort(dim=-1, stable=True)
        valid = valid.gather(1, order)
        positives = (relevance.gather(1, order) == 1) & valid
        starts = torch.cat((torch.ones_like(valid[:, :1]), values[:, 1:] != values[:, :-1]), -1)
        groups = starts.long().cumsum(-1) - 1
        positions = torch.arange(1, scores.shape[1] + 1, dtype=torch.float64, device=scores.device).expand_as(values)
        counts = torch.zeros_like(values).scatter_add(1, groups, valid.double())
        rank_sums = torch.zeros_like(values).scatter_add(1, groups, positions * valid)
        ranks = (rank_sums / counts.clamp_min(1)).gather(1, groups)
        p = positives.sum(-1)
        n = valid.sum(-1) - p
        wins = (ranks * positives).sum(-1) - p.double() * (p + 1) / 2
        return torch.where(p * n > 0, wins / (p * n).clamp_min(1), torch.nan)


def _regression_arrays(y_true, y_pred):
    def array(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)
    target, prediction = array(y_true), array(y_pred)
    if target.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("Value metrics require matching vectors [N]")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Value metrics require finite inputs")
    return target, prediction


class PCOC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        target, prediction = _regression_arrays(y_true, y_pred)
        denominator = target.sum()
        return float(prediction.sum() / denominator) if denominator != 0 else float("nan")


class RegressionAUC(BaseMetric):
    """P(pred_i > pred_j | value_i > value_j), with half credit for score ties.

    Kendall's sort/inversion-count algorithm is O(N log N), not a pair matrix.
    If D is the number of comparable value pairs and P is the number of
    non-tied prediction pairs, tau_b=(concordant-discordant)/sqrt(D*P).
    Thus R-AUC=1/2 + tau_b*sqrt(P/D)/2, including half credit for score ties.
    """
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        target, prediction = _regression_arrays(y_true, y_pred)
        pairs = len(target) * (len(target) - 1) // 2
        def untied(values):
            counts = np.unique(values, return_counts=True)[1]
            return pairs - int((counts * (counts - 1) // 2).sum())
        comparable = untied(target)
        if not comparable:
            return float("nan")
        prediction_pairs = untied(prediction)
        if not prediction_pairs:
            return .5
        tau = kendalltau(target, prediction, method="asymptotic").statistic
        return float(np.clip(.5 + .5 * tau * np.sqrt(prediction_pairs / comparable), 0, 1))


class SlateAUC(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        target, prediction = _regression_arrays(y_true, y_pred)
        binary = target > 0
        return float(roc_auc_score(binary, prediction)) if np.unique(binary).size == 2 else float("nan")


class RMSE(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        target, prediction = _regression_arrays(y_true, y_pred)
        return float(np.sqrt(np.mean((prediction - target) ** 2))) if len(target) else float("nan")


class MAE(BaseMetric):
    @classmethod
    def calculate(cls, y_true, y_pred, **kwargs):
        target, prediction = _regression_arrays(y_true, y_pred)
        return float(np.mean(np.abs(prediction - target))) if len(target) else float("nan")


class NCIS(BaseMetric):
    """Normalized capped score-ratio weighting; propensity semantics belong to scorers."""
    @classmethod
    def calculate(cls, y_true, y_pred, y_ref, cap=10.0, **kwargs):
        target, prediction = _regression_arrays(y_true, y_pred)
        _, reference = _regression_arrays(target, y_ref)
        if not np.isfinite(cap) or cap <= 0 or (prediction < 0).any() or (reference < 0).any():
            raise ValueError("ncis requires nonnegative scores and a finite positive cap")
        weight = np.minimum(prediction / np.maximum(reference, 1e-6), cap)
        total = weight.sum()
        return float(np.dot(weight, target) / total) if total > 0 else float("nan")

# 注册表：将字符串映射到对应的计算类
METRIC_REGISTRY = {
    "auc": AUC, "logloss": LogLoss, "gauc": GAUC,
    "hr": HR,
    "catalog_recall": HR,
    "catalog_mrr": CatalogMRR, "ndcg": NDCG, "list_ndcg": ListNDCG,
    "list_recall": ListRecall, "list_hit": ListHitRate, "list_precision": ListPrecision,
    "list_mrr": ListMRR, "list_map": ListMAP, "list_auc": InSlateAUC,
    "in_slate_auc": InSlateAUC, "pcoc": PCOC, "r_auc": RegressionAUC,
    "slate_auc": SlateAUC, "rmse": RMSE, "mae": MAE, "ncis": NCIS,
}

"""Fixed model constants and exhaustive-permutation coarse ranking."""

import torch
from torch import nn

from ..base import apply_permutation, build_full_permutations


def _constant(value, shape, default):
    tensor = default if value is None else torch.as_tensor(value, dtype=torch.float32).clone()
    if tensor.shape != shape or not torch.isfinite(tensor).all():
        raise ValueError(f"Expected finite constant with shape {tuple(shape)}, got {tuple(tensor.shape)}.")
    return tensor.detach()


class PPMLSimHashRetriever(nn.Module):
    def __init__(self, poi_num, page_num, emb_dim, *, position_encoding=None,
                 hash_vector=None, distance_weight=None, seed=2026):
        super().__init__()
        permutations = build_full_permutations(poi_num)
        if page_num <= 0 or emb_dim <= 0:
            raise ValueError("page_num and emb_dim must be positive.")
        # Local RNG: buffer creation never shifts shared trainable initialization.
        rng = torch.Generator().manual_seed(seed)
        self.register_buffer("full_permutations", permutations)
        self.register_buffer("position_encoding", _constant(
            position_encoding, (poi_num, emb_dim), torch.rand(poi_num, emb_dim, generator=rng)))
        self.register_buffer("hash_vector", _constant(
            hash_vector, (page_num, emb_dim), torch.rand(page_num, emb_dim, generator=rng)))
        self.register_buffer("distance_weight", _constant(
            distance_weight, (1, page_num), (1 / (torch.arange(page_num) // 2 + 2).float())[None]))

    def pool(self, pages):
        """PPML averages position-weighted embeddings over POIs and attributes."""
        return (pages * self.position_encoding[:, None, :]).mean(dim=(-3, -2))

    def forward(self, slate, history, page_mask=None):
        if history.shape[1] != self.hash_vector.shape[0]:
            raise ValueError("History page count does not match SimHash configuration.")
        # [B,K!,K,D] is enough: average attributes before gathering permutations.
        per_poi = slate.mean(-2)
        candidates = apply_permutation(per_poi, self.full_permutations)
        perm_signature = (candidates * self.position_encoding).mean(-2)
        hist_signature = self.pool(history)
        page_hash = torch.sign(hist_signature * self.hash_vector)
        perm_hash = torch.sign(perm_signature[:, :, None] * self.hash_vector)
        distance = (perm_hash - page_hash[:, None]).abs().sum(-1)
        weights = self.distance_weight
        if page_mask is not None:
            weights = weights * (page_mask == 1)
        return (distance * weights[:, None]).sum(-1)

    @staticmethod
    def top_k(distances, k):
        if not 1 <= k <= distances.shape[-1]:
            raise ValueError("Retrieval k must lie in [1, number of permutations].")
        return distances.argsort(dim=-1, stable=True)[:, :k]

    @staticmethod
    def last_k(distances, k):
        if not 1 <= k <= distances.shape[-1]:
            raise ValueError("Retrieval k must lie in [1, number of permutations].")
        return distances.argsort(dim=-1, descending=True, stable=True)[:, :k]


class BeamSearchRetriever(nn.Module):
    """Reference heuristic: rank exhaustive priors, not an iterative beam decoder."""

    def __init__(self, poi_num, expose_rate=None):
        super().__init__()
        default = (1.0 - 0.1 * torch.arange(poi_num)).clamp_min(0)[None]
        self.register_buffer("expose_rate", _constant(expose_rate, (1, poi_num), default))

    def forward(self, permutation_priors, top_k):
        if permutation_priors.ndim != 3 or permutation_priors.shape[-1] != self.expose_rate.shape[-1]:
            raise ValueError("Beam prior scores must be [B,K!,K].")
        if not 1 <= top_k <= permutation_priors.shape[1]:
            raise ValueError("top_k exceeds the available permutations.")
        return (permutation_priors * self.expose_rate).sum(-1).argsort(dim=-1, descending=True, stable=True)[:, :top_k]

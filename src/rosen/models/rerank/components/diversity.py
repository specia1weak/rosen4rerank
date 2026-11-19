"""Per-sample candidate statistics with explicit definitions (no fixed batch size)."""
import torch
from torch.nn import functional as F

from .generators import lc_exact_score


def candidate_statistics(embeddings, orders):
    flat = embeddings.flatten(2)
    pairs = torch.triu_indices(flat.shape[1], flat.shape[1], 1, device=flat.device)
    result = {"candidate_lc_index": lc_exact_score(orders)}
    if pairs.shape[1]:
        a, b = flat[:, pairs[0]], flat[:, pairs[1]]
        l2 = (a - b).norm(dim=-1)
        cosine = 1 - F.cosine_similarity(a, b, dim=-1)
        result.update(candidate_mean_l2=l2.mean(-1), candidate_max_l2=l2.amax(-1),
                      candidate_mean_cosine_distance=cosine.mean(-1), candidate_max_cosine_distance=cosine.amax(-1))
    else:
        result.update({name: flat.new_zeros(flat.shape[0]) for name in (
            "candidate_mean_l2", "candidate_max_l2", "candidate_mean_cosine_distance", "candidate_max_cosine_distance")})
    return result

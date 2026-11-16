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

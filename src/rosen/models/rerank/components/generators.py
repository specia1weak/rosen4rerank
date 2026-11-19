"""Independent encoders/decoders sharing caller-owned attention and pointer heads."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, widths, activation):
    layers = []
    for width in widths:
        if width <= 0:
            raise ValueError("MLP widths must be positive.")
        layer = nn.Linear(input_dim, width)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, activation()))
        input_dim = width
    return nn.Sequential(*layers), input_dim


class ContextEncoder(nn.Module):
    def __init__(self, num_attrs, emb_dim, mlp_a=(60, 32, 20)):
        super().__init__()
        self.mlp_a, width = mlp(num_attrs * emb_dim, mlp_a, nn.SiLU)
        self.output_dim = 3 * emb_dim + 1 + width

    def forward(self, cate, history, dense, page_mask, oam, intra_poi, target_attention):
        hist_pages, _ = oam(history)
        return self.encode_slate(cate, dense, hist_pages, page_mask, oam, intra_poi, target_attention)

    def encode_slate(self, cate, dense, hist_pages, page_mask, oam, intra_poi, target_attention):
        page, per_poi = oam(cate)
        interest = target_attention(page, hist_pages, page_mask)
        k = cate.shape[-3]
        return torch.cat((page.unsqueeze(-2).expand(*page.shape[:-1], k, -1),
                          interest.unsqueeze(-2).expand(*interest.shape[:-1], k, -1), intra_poi(per_poi),
                          dense, self.mlp_a(cate.flatten(-2))), dim=-1)


class PointwiseGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, selected_prior):
        logits = self.linear(selected_prior).squeeze(-1)
        return logits, logits.argsort(dim=-1, descending=True, stable=True)[:, None]


class ListwiseGenerator(nn.Module):
    """Also serves as the supervised OCPM, with an independently owned encoder."""

    def __init__(self, num_attrs, emb_dim, mlp_a=(60, 32, 20), mlp_b=(128, 32)):
        super().__init__()
        self.encoder = ContextEncoder(num_attrs, emb_dim, mlp_a)
        self.mlp_b, width = mlp(self.encoder.output_dim, mlp_b, nn.ReLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cate, history_pages, dense, page_mask, oam, intra_poi, target_attention):
        features = self.encoder.encode_slate(cate, dense, history_pages, page_mask, oam, intra_poi, target_attention)
        logits = self.output(self.mlp_b(features)).squeeze(-1)
        return logits, logits.argsort(dim=-1, descending=True, stable=True).unsqueeze(-2)


class PointerHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.layers, width = mlp(hidden_dim * 2, (64, 16), nn.ReLU)
        self.output = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, candidates, state):
        features = torch.cat((candidates, state[:, None].expand_as(candidates)), -1)
        return self.output(self.layers(features)).squeeze(-1)


@dataclass
class AutoregressiveOutput:
    indices: torch.Tensor       # [B,1,K]; training argmax records need not be permutations
    probabilities: torch.Tensor  # [B,K,K], step then item
    max_scores: torch.Tensor
    loss: torch.Tensor


class AutoregressiveGenerator(nn.Module):
    def __init__(self, num_attrs, emb_dim, mlp_a=(60, 32, 20), hidden_dim=64):
        super().__init__()
        self.encoder = ContextEncoder(num_attrs, emb_dim, mlp_a)
        self.project = nn.Linear(self.encoder.output_dim, hidden_dim)
        nn.init.xavier_uniform_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, cate, history, dense, page_mask, oam, intra_poi, target_attention,
                pointer_head, *, labels=None, label_mask=None, coverage=None, teacher_forcing=None):
        features = self.project(self.encoder(cate, history, dense, page_mask, oam, intra_poi, target_attention))
        b, k, _ = features.shape
        teacher_forcing = self.training if teacher_forcing is None else teacher_forcing
        if teacher_forcing and labels is None:
            raise ValueError("Teacher forcing requires labels.")
        state = features.mean(1).detach()
        next_input = torch.zeros_like(state)
        available = torch.ones(b, k, dtype=torch.bool, device=features.device)
        probabilities, indices, maxima = [], [], []
        loss = features.new_zeros(())
        rows = torch.arange(b, device=features.device)
        for step in range(k):  # Sequential decoding is inherently autoregressive.
            state = self.gru(next_input, state)
            logits = pointer_head(features, state).masked_fill(~available, -torch.inf)
            probs = logits.softmax(-1)
            chosen = probs.argmax(-1)
            indices.append(chosen)
            probabilities.append(probs)
            maxima.append(probs.amax(-1))
            selected = torch.full_like(chosen, step) if teacher_forcing else chosen
            if teacher_forcing:
                reward = labels[:, step]
                if coverage is not None:
                    reward = reward - coverage[:, step]
                if label_mask is not None:
                    reward = torch.where(label_mask[:, step] > 0, reward, 0) * label_mask[:, step]
                epsilon = 1e-6 if coverage is not None else 1e-9
                loss = loss - (reward * probs[:, step].clamp(epsilon, 1).log()).mean()
            available = available.scatter(1, selected[:, None], False)
            next_input = features[rows, selected].detach()
        return AutoregressiveOutput(torch.stack(indices, 1)[:, None], torch.stack(probabilities, 1),
                                    torch.stack(maxima, 1), loss)


# The two decoders have separate parameters; sharing is restricted to caller-owned heads/attention.
GenerativeGenerator = AutoregressiveGenerator
SubmodularityGenerator = AutoregressiveGenerator


def cosine_coverage(cate, other_slates):
    dot = (cate[:, None] * other_slates).sum((-1, -2))
    norm = cate[:, None].square().sum((-1, -2)).sqrt() * other_slates.square().sum((-1, -2)).sqrt()
    return (dot / norm.clamp_min(1e-4)).mean(1).detach()


def lc_exact_score(permutations):
    """Reference index-comparison statistic, not an inverse-permutation distance.

    Shape [B,G,K]. This deliberately preserves the original TF definition even
    for repeated training argmax indices. Discrete indices have no gradient.
    """
    relations = permutations[..., :, None] < permutations[..., None, :]
    pairs = torch.triu_indices(permutations.shape[1], permutations.shape[1], 1, device=permutations.device)
    return (relations[:, pairs[0]] != relations[:, pairs[1]]).float().sum((-1, -2, -3))


def lc_soft_score(probabilities, reference_permutations, tau=1.0):
    """Sum squared precedence differences over every pair, using one soft order.

    This is an experimental objective, not a relaxation numerically equivalent
    to lc_exact_score's comparisons of raw permutation indices.
    """
    if tau <= 0:
        raise ValueError("lc_soft temperature must be positive.")
    k = probabilities.shape[-1]
    steps = torch.arange(k, dtype=probabilities.dtype, device=probabilities.device)
    expected = (probabilities * steps[None, :, None]).sum(1)
    soft = torch.sigmoid((expected[:, None, :] - expected[:, :, None]) / tau)
    # Convert each reference order to positions per item. Training orders may
    # repeat; one-hot membership retains the explicitly recorded behavior.
    ref_positions = (F.one_hot(reference_permutations, k).to(probabilities.dtype) * steps[None, None, :, None]).sum(-2)
    reference = (ref_positions[..., None, :] > ref_positions[..., :, None]).to(probabilities.dtype).detach()
    diagonal = ~torch.eye(k, dtype=torch.bool, device=probabilities.device)
    soft_pairs = ((soft[:, None] - reference).square() * diagonal).sum((-1, -2, -3))
    # Reference-reference pairs are constant w.r.t. the learned order.
    # Only the soft-to-reference term belongs in the optimization objective.
    return soft_pairs

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from rosen.emb import SchemaManager
from rosen.models.msr.base import MSRModel
from rosen.models.msr.components import DomainTowerHead
from rosen.models.unified.rankmixer import Embedding2Tokenization, RankMixerBlock
from rosen.models.utils.common import to_dims


class DomainGatedTokenReadout(nn.Module):
    def __init__(self, num_domains: int, num_tokens: int):
        super().__init__()
        self.num_domains = int(num_domains)
        self.num_tokens = int(num_tokens)
        self.logits = nn.Embedding(self.num_domains, self.num_tokens)
        nn.init.zeros_(self.logits.weight)

    def weights(self, domain_ids: torch.Tensor) -> torch.Tensor:
        domain_ids = domain_ids.long().view(-1).clamp(min=0, max=self.num_domains - 1)
        return torch.softmax(self.logits(domain_ids), dim=-1)

    def forward(self, tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        weight = self.weights(domain_ids).unsqueeze(-1)
        return (tokens * weight).sum(dim=1)


class HeadedCTRModel(MSRModel):
    """Non-sequential multi-view CTR encoder.

    This is the non-sequence counterpart of RankMixer-style token processing:
    dense current embeddings are tokenized into latent heads, mixed by fixed
    HeadMix blocks, processed by per-token FFNs, then read by a lightweight
    readout before the domain towers.
    """

    def __init__(
        self,
        manager: SchemaManager,
        num_domains: int,
        token_dim: int = 64,
        num_layers: int = 3,
        expansion_factor: int = 2,
        num_tokens: int = 8,
        readout: str = "mean",
        activation: str = "relu",
        dropout_rate: float = 0.0,
        tower_hidden_dims: Iterable[int] = (128, 64),
        tower_dropout_rate: float = 0.0,
        tower_batch_norm: bool = False,
    ):
        super().__init__(manager, num_domains)
        self.DOMAIN = self.manager.domain_field
        self.LABEL = self.manager.label_field
        self.input_view = self.omni_embedding.whole
        self.input_dim = int(self.input_view.embedding_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.readout = str(readout).lower()
        if self.readout not in {"mean", "flatten", "domain_gate"}:
            raise ValueError("readout must be one of: mean, flatten, domain_gate.")

        self.tokenizer = Embedding2Tokenization(
            input_dim=self.input_dim,
            token_dim=self.token_dim,
            num_tokens=self.num_tokens,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.mixer = RankMixerBlock(
            input_dim=self.token_dim,
            num_tokens=self.num_tokens,
            num_layers=num_layers,
            expand=expansion_factor,
            dropout_rate=dropout_rate,
        )
        self.domain_readout = (
            DomainGatedTokenReadout(num_domains=num_domains, num_tokens=self.num_tokens)
            if self.readout == "domain_gate"
            else None
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.feature_dim = self.token_dim * self.num_tokens if self.readout == "flatten" else self.token_dim
        self.head = DomainTowerHead(
            num_domains=num_domains,
            input_dim=self.feature_dim,
            hidden_dims=to_dims(tower_hidden_dims, default=(128, 64)),
            dropout_rate=tower_dropout_rate,
            activation=activation,
            batch_norm=tower_batch_norm,
        )

    def encode_tokens(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.flatten(self.input_view(interaction), start_dim=1)
        domain_ids = interaction[self.DOMAIN].long().view(-1)
        tokens = self.tokenizer(x)
        tokens = self.mixer(tokens)
        return tokens, domain_ids

    def readout_tokens(self, tokens: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if self.readout == "mean":
            features = tokens.mean(dim=1)
        elif self.readout == "flatten":
            features = tokens.flatten(start_dim=1)
        elif self.readout == "domain_gate":
            if self.domain_readout is None:
                raise RuntimeError("domain readout is not initialized.")
            features = self.domain_readout(tokens, domain_ids)
        else:
            raise RuntimeError(f"Unsupported readout={self.readout!r}.")
        return self.dropout(features)

    def encode_features(self, interaction) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, domain_ids = self.encode_tokens(interaction)
        return self.readout_tokens(tokens, domain_ids), domain_ids

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.head(x, domain_ids)

    def logits(self, interaction):
        x, domain_ids = self.encode_features(interaction)
        return self.forward(x, domain_ids)

    def calculate_loss(self, interaction):
        labels = interaction[self.LABEL].float()
        logits = self.logits(interaction)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

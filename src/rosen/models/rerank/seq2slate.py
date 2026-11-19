"""Seq2Slate reference variant: greedy GRU-pointer candidate with auxiliary heads."""
from .nar import NARRerankModel


class Seq2SlateRerankModel(NARRerankModel):
    candidate_index = 2

"""NAR reference variant: three auxiliary generators, listwise-only candidate."""
from .mge import MultiGERerankModel
from .presets import resolve_rerank_config


class NARRerankModel(MultiGERerankModel):
    candidate_index = 1

    def __init__(self, manager, **kwargs):
        if kwargs.pop("submodularity_mode", "none") != "none":
            raise ValueError("NAR/Seq2Slate use three generators; submodularity_mode must be none.")
        super().__init__(manager, **resolve_rerank_config("nar", **kwargs))

    def _generate(self, *args, **kwargs):
        orders, aux = super()._generate(*args, **kwargs)
        selection = slice(self.candidate_index, self.candidate_index + 1)
        aux["candidate_dense"] = aux["candidate_dense"][:, selection]
        return orders[:, selection], aux

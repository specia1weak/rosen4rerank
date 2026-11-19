"""One source of variant defaults; explicit caller arguments always win."""

_GENERATOR_DEFAULTS = {"candidate_dense_mode": "generator", "loss_reduction": "nonzero_mean"}
RERANK_PRESETS = {
    "mge3": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "mge_cosine": {**_GENERATOR_DEFAULTS, "submodularity_mode": "cosine"},
    "lc_mge": {**_GENERATOR_DEFAULTS, "submodularity_mode": "lc"},
    "nar": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "seq2slate": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "mge_metric": {**_GENERATOR_DEFAULTS, "submodularity_mode": "none"},
    "mge_cosine_metric": {**_GENERATOR_DEFAULTS, "submodularity_mode": "cosine"},
}


def resolve_rerank_config(name, **overrides):
    """Apply a named preset without rewriting any explicit override."""
    return {**RERANK_PRESETS.get(name.lower(), {}), **overrides}

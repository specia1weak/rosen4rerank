"""Registered MSR models and HAMUR branches for prediction contract checks."""
import torch
from dataclasses import replace

from rosen.models.msr import MODEL_REGISTRY, build_model
from rosen.emb.schema import SeqGroupConfig, SequenceSetting
from tests.support.model_fixtures import DummyCls


CASES = {name: (name, {}) for name in MODEL_REGISTRY}
CASES['hierrec'] = ('hierrec', {'embed_dim': DummyCls.EMB_DIM})
for name in ('sdsp_ple', 'sdsp_mmoe'):
    CASES[name] = (name, {'scaled_batch': (2, 2, 2)})
for backbone, depth in (('mlp', 2), ('mlp', 7), ('dcn', 2), ('widedeep', 2)):
    for adapter in (False, True):
        kwargs = dict(backbone=backbone, depth=depth, use_adapter=adapter, k=4,
                      adapter_bottleneck_dim=4, hyper_dims=(8,))
        if backbone == 'mlp':
            kwargs['fcn_dims'] = (16,) * depth
        else:
            kwargs['mlp_dims'] = (16, 8)
        if backbone == 'widedeep':
            kwargs.update(wide_fields=('user_id',), deep_fields=('video_id', 'domain'))
        CASES[f'hamur_{backbone}{depth}_{adapter}'] = ('hamur', kwargs)


def make_case(name):
    torch.manual_seed(2026)
    model_name, kwargs = CASES[name]
    manager = DummyCls.MANAGER
    if model_name in ('din', 'ads_din'):
        history = SequenceSetting('hist_video', manager.get_setting('video_id'),
                                  SeqGroupConfig('history', 'history_len', 3))
        manager = replace(manager, settings=(*manager.settings, history))
    model = build_model(manager, DummyCls.NUM_DOMAINS, model_name, **kwargs)
    interaction = DummyCls.make_interaction(
        batch_size=6, domain=torch.tensor([0, 0, 1, 1, 2, 2]), label=torch.tensor([0., 1., 0., 1., 0., 1.]),
    )
    if model_name in ('din', 'ads_din'):
        interaction['domain'] = interaction['domain'] + 1
        interaction['hist_video'] = torch.tensor([[1, 2, 0]] * 6)
        interaction['history_len'] = torch.tensor([2] * 6)
    return model, interaction

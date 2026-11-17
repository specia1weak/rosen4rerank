"""Concrete unified model families for the shared logits/probability contract."""
from dataclasses import replace
import importlib
import inspect

import torch

from rosen.emb.schema import SeqGroupConfig, SequenceSetting
from rosen.models.msr.base import MSRModel
from tests.support.model_fixtures import DummyCls

MODULES = ('headed_ctr', 'hyformer', 'rankmixer', 'tokenmixer', 'mixformer', 'sasrec', 'unirank_seq')
CASES = {}
for module_name in MODULES:
    module = importlib.import_module(f'rosen.models.unified.{module_name}')
    for name, cls in vars(module).items():
        if (inspect.isclass(cls) and cls.__module__ == module.__name__
                and issubclass(cls, MSRModel) and name != 'UniRankSequenceBaseModel'):
            CASES[name] = cls


def make_case(name):
    torch.manual_seed(2026)
    group = SeqGroupConfig('user_history', 'history_len', 4)
    manager = replace(DummyCls.MANAGER, settings=(*DummyCls.MANAGER.settings,
        SequenceSetting('video_id_seq', DummyCls.MANAGER.get_setting('video_id'), group),
        SequenceSetting('domain_tab_seq', DummyCls.MANAGER.get_setting('domain'), group)))
    cls = CASES[name]
    params = inspect.signature(cls.__init__).parameters
    # Small valid dimensions; architecture options retain their declared defaults.
    options = dict(token_dim=16, num_layers=1, num_heads=2, num_ns_token=4,
                   num_tokens=4, tower_hidden_dims=(16, 8), bottom_hidden_dims=(16, 8),
                   tower_dropout_rate=0.0, item_id_emb_dim=8, query_size=2, real_seq_len=2, event_tab_seq_field='domain_tab_seq')
    if name == 'TokenMixerModel':
        options['token_dim'] = 20  # five tokens; token mixing requires a divisible width
    kwargs = {k: v for k, v in options.items() if k in params}
    model = cls(manager=manager, num_domains=DummyCls.NUM_DOMAINS, **kwargs)
    inter = DummyCls.make_interaction(batch_size=6, domain=torch.tensor([1, 1, 2, 2, 3, 3]),
                                     label=torch.tensor([0., 1., 0., 1., 0., 1.]))
    if name == 'HeadedCTRModel':
        inter['domain'] -= 1  # this non-sequential family routes zero-based IDs directly
    inter['video_id_seq'] = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]] * 2)
    inter['domain_tab_seq'] = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0], [3, 1, 2, 3]] * 2)
    inter['history_len'] = torch.tensor([3, 2, 4] * 2)
    return model, inter

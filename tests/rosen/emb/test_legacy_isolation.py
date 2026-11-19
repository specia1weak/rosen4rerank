import importlib

import pytest
import torch

from rosen.core.enum_type import FeatureSource
from rosen.emb.schema import SparseEmbSetting


@pytest.mark.parametrize('name', ['BoleEmbLayer', 'SideEmb', 'UserSideEmb', 'ItemSideEmb',
                                 'InterSideEmb', 'CustomSideEmb', 'ProfileEncoder',
                                 'UserProfileEncoder', 'ItemProfileEncoder', 'SeqEmbedder'])
def test_old_import_names_resolve_to_isolated_legacy_classes(name):
    modern = importlib.import_module('rosen.emb.emblayer')
    legacy = importlib.import_module('rosen.emb.legacy')
    assert getattr(modern, name) is getattr(legacy, name)
    assert getattr(legacy, name).__module__ == 'rosen.emb.legacy'


def test_legacy_side_embedding_preserves_state_keys_and_outputs():
    from rosen.emb.emblayer import SideEmb as OldSideEmb
    from rosen.emb.legacy import SideEmb
    atom = SparseEmbSetting('item', FeatureSource.ITEM, embedding_dim=4)
    atom._build_vocab_indices(['a', 'b'])
    model = OldSideEmb([atom])
    restored = SideEmb([atom])
    assert list(model.state_dict()) == ['embedding.emb_modules.item.embedding.weight']
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = {'item': torch.tensor([1, 2])}
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)


def test_modern_source_defines_only_the_four_current_classes():
    import ast
    import inspect
    module = importlib.import_module('rosen.emb.emblayer')
    classes = [node.name for node in ast.parse(inspect.getsource(module)).body if isinstance(node, ast.ClassDef)]
    assert classes == ['RecEmbedding', 'EmbView', 'SeqGroupView', 'OmniEmbLayer']

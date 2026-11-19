import torch

from rosen.core.enum_type import FeatureSource
from rosen.emb import SchemaManager
from rosen.emb.emblayer import EmbView, OmniEmbLayer
from rosen.emb.schema import SparseEmbSetting


def test_missing_domain_and_explicit_empty_filters_stay_empty(tmp_path):
    atom = SparseEmbSetting('item', FeatureSource.ITEM, embedding_dim=4)
    atom._build_vocab_indices(['1', '2'])
    model = OmniEmbLayer(manager=SchemaManager([atom], tmp_path))
    batch = {'item': torch.tensor([1, 2])}
    for view in (model.domain, model.domain_id, EmbView(model, include_fields=()), EmbView(model, target_sources=())):
        assert view.selected_settings == () and view.embedding_dim == 0
        assert view(batch) is None
    assert model.whole.embedding_dim == 4
    assert EmbView(model, target_sources=FeatureSource.ITEM).embedding_dim == 4
    assert model.whole_without_domain.embedding_dim == 4


def test_explicit_domain_only_selects_that_field(tmp_path):
    settings = [SparseEmbSetting(name, source, embedding_dim=4) for name, source in (
        ('item', FeatureSource.ITEM), ('domain', FeatureSource.INTERACTION))]
    for atom in settings:
        atom._build_vocab_indices(['1', '2'])
    model = OmniEmbLayer(manager=SchemaManager(settings, tmp_path, domain_fields='domain'))
    assert [s.field_name for s in model.domain.selected_settings] == ['domain']
    assert [s.field_name for s in model.whole_without_domain.selected_settings] == ['item']

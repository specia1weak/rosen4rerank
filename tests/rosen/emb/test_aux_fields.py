import json

import polars as pl
import pytest
import torch

from rosen.core.enum_type import FeatureSource
from rosen.data.dataset import ParquetStreamDataset, RawParquetStreamDataset
from rosen.data.padding import FeatureContext, TensorFormatter
from rosen.emb import AuxField, SchemaManager, SlateLabel
from rosen.emb.emblayer import OmniEmbLayer
from rosen.emb.schema import SparseEmbSetting


@pytest.mark.parametrize('dataset_class', [ParquetStreamDataset, RawParquetStreamDataset])
def test_aux_inputs_are_formatted_without_extras_or_embeddings(tmp_path, dataset_class):
    atom = SparseEmbSetting('item', FeatureSource.ITEM, embedding_dim=4)
    atom._build_vocab_indices(['1', '2'])
    specs = [AuxField('prior', 3, pad_val=-1.), AuxField('pages', 2, 'int64', 'left'),
             AuxField('weight'), AuxField('request', dtype='int64')]
    manager = SchemaManager([atom], tmp_path / 'schema', label_fields=SlateLabel('label', 3, 'mask'), aux_fields=specs)
    manager.save_schema()
    assert len(json.loads(manager.meta_filepath.read_text())) == 1
    model = OmniEmbLayer(manager=manager)
    assert [s.field_name for s in model.whole.selected_settings] == ['item']
    frame = pl.DataFrame({'item': [1, 2], 'label': [[1.], [0., 1.]], 'mask': [[1.], [1., 1.]],
                          'prior': [[.4], [.2, .3]], 'pages': [[1], [1, 0]],
                          'weight': [1.5, 2.], 'request': [11, 12]})
    path = tmp_path / 'data.parquet'
    frame.write_parquet(path)
    batch, = list(dataset_class(path, manager, batch_size=2, shuffle=False, drop_last=False))
    assert set(batch.columns) == set(manager.fields())
    torch.testing.assert_close(batch['prior'], torch.tensor([[.4, -1., -1.], [.2, .3, -1.]]))
    assert batch['pages'].tolist() == [[0, 1], [1, 0]] and batch['pages'].dtype == torch.long
    assert batch['weight'].dtype == torch.float32 and batch['request'].dtype == torch.long
    reloaded = SchemaManager([SparseEmbSetting('item', FeatureSource.ITEM, embedding_dim=4)],
                             tmp_path / 'schema', label_fields=SlateLabel('label', 3, 'mask'), aux_fields=specs)
    reloaded.load_schema()
    assert reloaded.aux_fields == manager.aux_fields
    assert reloaded.get_aux_spec('pages') == specs[1]


def test_aux_fit_transform_preserves_unembedded_input(tmp_path):
    frame = pl.DataFrame({'item': ['a', 'b'], 'prior': [[.1, .2], [.3]], 'label': [1., 0.]})
    manager = SchemaManager([SparseEmbSetting('item', FeatureSource.ITEM, embedding_dim=4)],
                             tmp_path, label_fields='label', aux_fields=AuxField('prior', 2))
    manager.fit(frame.lazy())
    transformed = manager.transform(frame.lazy()).collect()
    assert transformed['prior'].to_list() == frame['prior'].to_list()
    assert manager.get_setting('prior') is None


@pytest.mark.parametrize('factory', [FeatureContext.from_manager, FeatureContext.from_raw_manager])
def test_label_free_context_keeps_typed_aux_and_shared_masks(tmp_path, factory):
    manager = SchemaManager([], tmp_path, label_fields=[
        SlateLabel('click', 3, 'valid', padding_side='left'),
        SlateLabel('long_view', 3, 'valid', padding_side='left'), 'retention',
    ], aux_fields=[AuxField('prior', 3, pad_val=-1.), AuxField('request', dtype='int64')])
    context = factory(manager, include_labels=False)
    assert context.output_col_names == ('valid', 'prior', 'request')
    assert set(context.read_col_names) == set(context.output_col_names)
    assert factory(manager).output_col_names == tuple(manager.fields())
    frame = pl.DataFrame({'valid': [[1., 1.]], 'prior': [[.4]], 'request': [2**53 + 1]})
    batch = TensorFormatter(context).format({name: frame[name].to_numpy() for name in context.read_col_names})
    assert batch['valid'].tolist() == [[0., 1., 1.]]
    torch.testing.assert_close(batch['prior'], torch.tensor([[.4, -1., -1.]]))
    assert batch['request'].dtype == torch.int64 and batch['request'].item() == 2**53 + 1
    # Excluding labels also wins over explicit extra columns/formatters.
    override = factory(manager, extra_col_names=['click'],
                       extra_col_formatters={'retention': TensorFormatter(context).col_formatters['prior']},
                       include_labels=False)
    assert override.read_col_names == context.read_col_names
    assert override.output_col_names == context.output_col_names


@pytest.mark.parametrize('kwargs', [{'max_len': 0}, {'dtype': 'float16'}, {'padding_side': 'up'},
                                   {'pad_val': float('nan')}, {'dtype': 'int64', 'pad_val': .5}])
def test_aux_declaration_validation(kwargs):
    with pytest.raises(ValueError):
        AuxField('prior', **kwargs)


def test_aux_conflicts_do_not_silently_override_supervision_or_features(tmp_path):
    with pytest.raises(ValueError, match='conflicts'):
        SchemaManager([], tmp_path, label_fields='label', aux_fields=AuxField('label'))
    with pytest.raises(ValueError, match='Conflicting'):
        SchemaManager([], tmp_path, aux_fields=[AuxField('prior', 2), AuxField('prior', 3)])
    with pytest.raises(TypeError, match='AuxField'):
        SchemaManager([], tmp_path, aux_fields=['prior'])

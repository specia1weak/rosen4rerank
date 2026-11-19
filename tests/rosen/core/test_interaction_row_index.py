import numpy as np
import pytest
import torch

from rosen.core.interaction import Interaction


@pytest.mark.parametrize('index', [torch.tensor([True, False, True]), torch.tensor([2, 0]),
                                  np.array([True, False, True]), np.array([2, 0], dtype=np.uint32)])
def test_tensor_and_numpy_selection_use_native_indexing_without_python_lists(monkeypatch, index):
    batch = Interaction({'id': torch.tensor([4, 7, 9]), 'features': torch.arange(6).reshape(3, 2)})
    expected = {key: batch[key][index.astype(np.int64) if isinstance(index, np.ndarray) and index.dtype == np.uint32 else index]
                for key in batch}
    monkeypatch.setattr(torch.Tensor, 'tolist', lambda *args: (_ for _ in ()).throw(AssertionError('tolist called')))
    result = batch[index]
    for key in batch:
        torch.testing.assert_close(result[key], expected[key])

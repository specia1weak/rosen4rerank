import numpy as np
import pytest
import torch

from rosen.core.formatters import (
    FallbackFormatter,
    IntFormatter,
    PaddedFloatSequenceFormatter,
    PaddedIntSequenceFormatter,
    PaddedNestedSequenceFormatter,
    VectorDenseFormatter,
)


@pytest.mark.parametrize("formatter_cls, dtype", [
    (PaddedIntSequenceFormatter, torch.int64),
    (PaddedFloatSequenceFormatter, torch.float32),
])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mode, truncated", [("head", [1, 2, 3]), ("tail", [2, 3, 4])])
def test_sequence_truncation_is_independent_of_padding(formatter_cls, dtype, side, mode, truncated):
    output = formatter_cls(3, side, -1, mode).format([
        [1, 2, 3, 4], (5,), np.array([6, 7]), [], None, np.nan, np.array(None),
    ])
    expected = [truncated, [5, -1, -1], [6, 7, -1], *[[-1, -1, -1]] * 4]
    if side == "left":
        expected[1:3] = [[-1, -1, 5], [-1, 6, 7]]
    assert output.dtype == dtype
    assert output.tolist() == expected


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("mode", ["head", "tail"])
@pytest.mark.parametrize("tag_mode", ["head", "tail"])
def test_nested_sequences_truncate_outer_and_inner_axes_independently(side, mode, tag_mode):
    output = PaddedNestedSequenceFormatter(2, 2, side, -1, mode, tag_mode).format([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10]], [None, []], None,
    ])
    outer = [[1, 2, 3], [4, 5, 6]] if mode == "head" else [[4, 5, 6], [7, 8, 9]]
    expected = [tags[:2] if tag_mode == "head" else tags[-2:] for tags in outer]
    assert output[0].tolist() == expected
    assert output[1].tolist() == ([[10, -1], [-1, -1]] if side == "right"
                                  else [[-1, -1], [10, -1]])
    assert output[2:].tolist() == [[[-1, -1], [-1, -1]]] * 2
    assert output.dtype == torch.int64


@pytest.mark.parametrize("formatter", [
    PaddedIntSequenceFormatter(2, padding_side="up"),
    PaddedFloatSequenceFormatter(2, truncate_mode="middle"),
    PaddedNestedSequenceFormatter(2, 2, padding_side="up"),
    PaddedNestedSequenceFormatter(2, 2, truncate_mode="middle"),
    PaddedNestedSequenceFormatter(2, 2, tag_truncate_mode="middle"),
])
def test_invalid_padding_args_are_rejected_even_with_empty_batches(formatter):
    with pytest.raises(ValueError, match="Unknown"):
        formatter.format([])


def test_zero_width_and_empty_batches():
    assert PaddedIntSequenceFormatter(0).format([[1], None]).shape == (2, 0)
    assert PaddedIntSequenceFormatter(3).format([]).shape == (0, 3)
    assert PaddedNestedSequenceFormatter(0, 2).format([[[1]]]).shape == (1, 0, 2)
    assert PaddedNestedSequenceFormatter(2, 0, tag_truncate_mode="tail").format([[[1]]]).shape == (1, 2, 0)


def test_scalar_and_vector_inputs():
    values = np.array([2**32 - 1], dtype=np.uint32)
    for formatter in (IntFormatter(), FallbackFormatter()):
        assert formatter.format(values).tolist() == [2**32 - 1]
        assert formatter.format([2**53 + 1]).tolist() == [2**53 + 1]
    assert FallbackFormatter().format(np.array([1.5], dtype=np.float64)).dtype == torch.float32
    assert VectorDenseFormatter(2).format([None, np.array(None), [], [1], [1, 2, 3]]).tolist() == [
        [0, 0], [0, 0], [0, 0], [1, 0], [1, 2],
    ]
    with pytest.raises(ValueError, match="row 0"):
        VectorDenseFormatter(2, zero_fill=False).format([np.array(None)])

import torch

from rosen.utils.observatory import field_zeroing_intervention


def test_field_zeroing_intervention_computes_field_and_domain_deltas():
    x = torch.tensor(
        [
            [1.0, 2.0, 10.0, 20.0],
            [3.0, 4.0, 30.0, 40.0],
            [5.0, 6.0, 50.0, 60.0],
            [7.0, 8.0, 70.0, 80.0],
        ]
    )
    field_slices = {
        "left": slice(0, 2),
        "right": slice(2, 4),
    }
    domain_ids = torch.tensor([0, 0, 1, 1])
    module = torch.nn.Dropout(p=0.5)
    module.train()

    def forward_fn(masked_x):
        return {
            "sum": masked_x.sum(dim=1),
            "hidden": torch.stack(
                [
                    masked_x[:, 0] + masked_x[:, 2],
                    masked_x[:, 1] - masked_x[:, 3],
                ],
                dim=1,
            ),
        }

    result = field_zeroing_intervention(
        x,
        field_slices,
        forward_fn,
        targets=("sum", "hidden"),
        domain_ids=domain_ids,
        num_domains=2,
        module=module,
    )

    assert module.training is True
    assert result.field_names == ("left", "right")
    assert result.target_names == ("sum", "hidden")
    assert result.domain_labels == ("d0", "d1")
    assert torch.allclose(result.abs_delta["sum"], torch.tensor([9.0, 90.0]))
    assert torch.allclose(result.signed_delta["sum"], torch.tensor([9.0, 90.0]))
    assert torch.allclose(
        result.by_domain_abs_delta["sum"],
        torch.tensor(
            [
                [5.0, 13.0],
                [50.0, 130.0],
            ]
        ),
    )
    assert result.target_matrix("abs_delta").shape == (2, 2)
    assert result.domain_matrix("sum").shape == (2, 2)
    assert result.top_fields("sum", k=1) == (("right", 90.0),)

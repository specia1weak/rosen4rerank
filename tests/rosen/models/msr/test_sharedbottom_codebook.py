import torch
from torch import nn

from rosen.models.msr import build_model
from rosen.models.msr.sharedbottom import SharedBottomBackbone
from tests.support.model_fixtures import DummyCls


def test_sharedbottom_backbone_preserves_signed_output_by_default():
    backbone = SharedBottomBackbone(
        input_dim=2,
        num_domains=1,
        hidden_dims=(2,),
        activation="relu",
        batch_norm=False,
    )
    linear = backbone.bottom.net[0]
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()

    output = backbone(torch.tensor([[-1.0, 2.0]]), torch.tensor([0]))

    assert torch.equal(output, torch.tensor([[-1.0, 2.0]]))


def test_sharedbottom_backbone_can_enable_output_activation():
    backbone = SharedBottomBackbone(
        input_dim=2,
        num_domains=1,
        hidden_dims=(2,),
        activation="relu",
        batch_norm=False,
        activate_output=True,
    )
    linear = backbone.bottom.net[0]
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()

    output = backbone(torch.tensor([[-1.0, 2.0]]), torch.tensor([0]))

    assert torch.equal(output, torch.tensor([[0.0, 2.0]]))


def test_sharedbottom_backbone_can_disable_output_activation():
    backbone = SharedBottomBackbone(
        input_dim=2,
        num_domains=1,
        hidden_dims=(2,),
        activation="relu",
        batch_norm=False,
        activate_output=False,
    )
    linear = backbone.bottom.net[0]
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()

    output = backbone(torch.tensor([[-1.0, 2.0]]), torch.tensor([0]))

    assert torch.equal(output, torch.tensor([[-1.0, 2.0]]))


def test_sharedbottom_less_and_plus_build_resolved_defaults_once():
    less = build_model(DummyCls.MANAGER, DummyCls.NUM_DOMAINS, "sharedbottom_less")
    plus = build_model(DummyCls.MANAGER, DummyCls.NUM_DOMAINS, "sharedbottom_plus")

    input_dim = less.input_dim
    less_dims = [
        module.out_features
        for module in less.backbone.bottom.modules()
        if isinstance(module, nn.Linear)
    ]
    plus_dims = [
        module.out_features
        for module in plus.backbone.bottom.modules()
        if isinstance(module, nn.Linear)
    ]

    assert less_dims == [input_dim]
    assert plus_dims == [input_dim, input_dim, max(1, input_dim // 2), input_dim]
    assert all(
        any(isinstance(module, nn.BatchNorm1d) for module in tower.modules())
        for tower in less.head.head.domain_networks
    )


def test_sharedbottom_codebook_smoke():
    model = build_model(DummyCls.MANAGER, DummyCls.NUM_DOMAINS, "sharedbottom_codebook")
    interaction = DummyCls.make_interaction(batch_size=4)

    logits = model.predict(interaction)
    loss = model.calculate_loss(interaction)

    assert logits.shape == (4,)
    assert torch.isfinite(loss)
    assert loss.ndim == 0

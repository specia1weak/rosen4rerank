import torch

from rosen.models.msr import MODEL_REGISTRY, build_model
from rosen.models.msr.satrans import SATransModel
from tests.support.model_fixtures import DummyCls


def test_satrans_registered():
    assert MODEL_REGISTRY["satrans"] is SATransModel


def test_satrans_shared_linear_smoke():
    model = build_model(
        DummyCls.MANAGER,
        DummyCls.NUM_DOMAINS,
        "satrans",
        domain_att_layer_num=1,
        att_head_num=2,
        meta_hidden_units=(16, 8),
    )
    interaction = DummyCls.make_interaction(batch_size=4)

    logits = model.logits(interaction)
    loss = model.calculate_loss(interaction)

    assert logits.shape == (4,)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    assert model.field_names == (DummyCls.USER_ID, DummyCls.ITEM_ID, DummyCls.DOMAIN)


def test_satrans_domain_tower_smoke():
    model = build_model(
        DummyCls.MANAGER,
        DummyCls.NUM_DOMAINS,
        "satrans",
        domain_att_layer_num=1,
        att_head_num=2,
        meta_hidden_units=(16, 8),
        head_type="domain_tower",
        tower_hidden_dims=(16,),
    )
    interaction = DummyCls.make_interaction(batch_size=5)

    logits = model.logits(interaction)

    assert logits.shape == (5,)

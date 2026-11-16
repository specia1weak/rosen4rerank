from rosen.emb.emblayer import EmbView, OmniEmbLayer
from tests.support.model_fixtures import DummyCls


def test_emb_view_exposes_selected_field_segments_and_slices():
    omni = OmniEmbLayer(manager=DummyCls.MANAGER)

    assert tuple(setting.field_name for setting in omni.whole.selected_settings) == (
        DummyCls.USER_ID,
        DummyCls.ITEM_ID,
        DummyCls.DOMAIN,
    )
    assert omni.whole.field_segments == (
        (DummyCls.USER_ID, 0, DummyCls.EMB_DIM),
        (DummyCls.ITEM_ID, DummyCls.EMB_DIM, DummyCls.EMB_DIM * 2),
        (DummyCls.DOMAIN, DummyCls.EMB_DIM * 2, DummyCls.EMB_DIM * 3),
    )
    assert omni.whole.field_slices == {
        DummyCls.USER_ID: slice(0, DummyCls.EMB_DIM),
        DummyCls.ITEM_ID: slice(DummyCls.EMB_DIM, DummyCls.EMB_DIM * 2),
        DummyCls.DOMAIN: slice(DummyCls.EMB_DIM * 2, DummyCls.EMB_DIM * 3),
    }

    without_domain = EmbView(omni, exclude_fields=(DummyCls.DOMAIN,))

    assert tuple(without_domain.field_slices) == (DummyCls.USER_ID, DummyCls.ITEM_ID)
    assert without_domain.embedding_dim == DummyCls.EMB_DIM * 2

import torch

from rosen.core.train.hooks import ObservationSourceProtocol
from rosen.models.base import BaseModel
from tests.support.model_fixtures import DummyCls
from rosen.utils.observatory import (
    AxisSpec,
    CapturedObservation,
    ObservationPort,
    ObservationSpec,
    TensorObservatory,
    record_observation_port,
)


class _ToyModel(BaseModel):
    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError


class _ObservedToyModel(_ToyModel, ObservationSourceProtocol):
    def should_collect_observations(self, ctx, stage: str) -> bool:
        return stage == "train"

    def collect_observations(self, ctx, stage: str) -> None:
        del ctx, stage


def test_observation_port_noop_and_collecting_snapshot():
    port = ObservationPort(namespace="demo")
    x = torch.ones(2, 3, requires_grad=True)

    returned = port.capture("x", x)

    assert returned is x
    assert len(port) == 0

    with port.collecting(enabled=True, clear=True):
        y = x * 2.0
        returned = port.capture("y", y)

    assert returned is y
    assert y.requires_grad
    assert len(port) == 1

    captured = port.snapshot(values_only=False)["y"]
    assert isinstance(captured, CapturedObservation)
    assert captured.full_name == "demo.y"
    assert captured.value.requires_grad is False
    assert torch.equal(captured.value, y.detach())

    values = port.snapshot()
    assert torch.equal(values["y"], y.detach())

    cleared = port.snapshot(clear=True)
    assert "y" in cleared
    assert len(port) == 0


def test_observation_port_registers_static_specs():
    port = ObservationPort(namespace="demo")
    spec = port.register(
        "hidden",
        role="hidden",
        axes=("batch", AxisSpec("hidden", kind="hidden")),
        metrics=("basic", "spectral"),
    )
    port.register("domain_ids", role="domain", axes=("batch",), metrics=())

    assert isinstance(spec, ObservationSpec)
    assert spec.role == "hidden"
    assert tuple(axis.name for axis in spec.axes) == ("batch", "hidden")
    assert port.spec("domain_ids").metrics == ()

    with port.collecting(enabled=True, clear=True):
        port.capture_many({
            "hidden": torch.ones(2, 3),
            "domain_ids": torch.tensor([0, 1]),
            "temporary": torch.zeros(1),
        })

    assert set(port.snapshot()) == {"hidden", "domain_ids", "temporary"}

    try:
        port.register("hidden")
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate observation registration should fail")


def test_observation_port_cut_and_watch_are_noop_until_activated():
    port = ObservationPort(namespace="demo")
    x = torch.ones(2, 3)

    returned = port.cut("field_a", x)
    assert returned is x
    assert port.cut_names() == ("field_a",)

    with port.cutting("field_a"):
        cut_value = port.cut("field_a", x)
        other_value = port.cut("field_b", x)

    assert torch.equal(cut_value, torch.zeros_like(x))
    assert other_value is x
    assert port.cut_names() == ("field_a", "field_b")

    watched = port.watch("hidden", x)
    assert watched is x
    assert port.watch_names() == ("hidden",)
    assert port.watch_snapshot() == {}

    with port.watching(enabled=True, clear=True):
        port.watch("hidden", x * 2.0)

    trace = port.watch_snapshot(clear=True)
    assert torch.equal(trace["hidden"], x * 2.0)
    assert port.watch_snapshot() == {}


def test_observation_port_gradient_taps_are_noop_until_activated():
    port = ObservationPort(namespace="demo")
    x = torch.ones(2, 3, requires_grad=True)

    y = x * 2.0
    returned = port.tap_gradient("hidden", y)
    assert returned is y
    assert port.gradient_tap_names() == ("hidden",)

    y.sum().backward()

    assert port.gradient_snapshot() == {}


def test_observation_port_taps_gradients_when_enabled():
    port = ObservationPort(namespace="demo")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    with port.tapping_gradients(enabled=True, clear=True):
        y = x * 2.0
        returned = port.tap_gradient("hidden", y)
        assert returned is y
        (y.square().sum()).backward()

    grads = port.gradient_snapshot(clear=True)

    assert set(grads) == {"hidden"}
    assert torch.equal(grads["hidden"], 2.0 * y.detach())
    assert port.gradient_snapshot() == {}


def test_observation_port_observe_can_watch_and_tap_gradient():
    port = ObservationPort(namespace="demo")
    x = torch.ones(2, 3, requires_grad=True)

    with port.watching(enabled=True, clear=True), port.tapping_gradients(enabled=True, clear=True):
        y = x * 3.0
        returned = port.observe("hidden", y, watch=True, gradient=True)
        assert returned is y
        y.sum().backward()

    trace = port.watch_snapshot(clear=True)
    grads = port.gradient_snapshot(clear=True)

    assert set(trace) == {"hidden"}
    assert set(grads) == {"hidden"}
    assert torch.equal(trace["hidden"], y.detach())
    assert torch.equal(grads["hidden"], torch.ones_like(y))
    assert trace["hidden"].requires_grad is False
    assert port.watch_snapshot() == {}
    assert port.gradient_snapshot() == {}


def test_record_observation_port_uses_registered_metrics():
    port = ObservationPort(namespace="toy")
    recorder = TensorObservatory(window_size=2)
    port.register("hidden", role="hidden", axes=("batch", "hidden"), metrics=("basic",))
    port.register("domain_ids", role="domain", axes=("batch",), metrics=())

    with port.collecting(enabled=True, clear=True):
        port.capture_many({
            "hidden": torch.ones(3, 4),
            "domain_ids": torch.tensor([0, 1, 2]),
            "unregistered": torch.zeros(3, 1),
        })

    trace = record_observation_port(port, recorder, step=7)

    assert set(trace) == {"hidden", "domain_ids", "unregistered"}
    assert "toy.hidden" in recorder.features
    assert "toy.domain_ids" not in recorder.features
    assert "toy.unregistered" not in recorder.features
    assert recorder.features["toy.hidden"][-1]["step"] == 7
    assert len(port) == 0


def test_base_model_has_obp_without_becoming_observation_source():
    model = _ToyModel(DummyCls.MANAGER)

    assert isinstance(model.obp, ObservationPort)
    assert not isinstance(model, ObservationSourceProtocol)
    assert all("obp" not in key for key in model.state_dict())
    assert all(name != "obp" for name, _ in model.named_modules())


def test_explicit_observation_source_protocol_marks_collectable_model():
    model = _ObservedToyModel(DummyCls.MANAGER)

    assert isinstance(model.obp, ObservationPort)
    assert isinstance(model, ObservationSourceProtocol)

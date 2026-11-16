from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from rosen.utils.observatory.config import TensorMonitorOptions
from rosen.utils.observatory.port import ObservationPort


def record_observation_port(
        port: ObservationPort,
        recorder: Any,
        *,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
        clear: bool = True,
) -> Dict[str, Any]:
    """
    Drain an ObservationPort into a TensorObservatory-like recorder.

    Models own the forward trace. The port owns static observation specs. The
    recorder owns metric computation. This connector is the small boundary
    between those pieces.
    """
    trace = port.snapshot(clear=clear)
    if recorder is None or not hasattr(recorder, "record"):
        return trace

    namespace = port.namespace if prefix is None else str(prefix)
    for name, value in trace.items():
        spec = port.spec(name)
        if spec is None or not spec.metrics:
            continue
        if not isinstance(value, torch.Tensor):
            continue

        record_name = f"{namespace}.{name}" if namespace else str(name)
        options = TensorMonitorOptions(metrics=tuple(spec.metrics))
        recorder.record(record_name, value, options=options, step=step)

    return trace

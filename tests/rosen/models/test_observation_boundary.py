from __future__ import annotations

import ast
from pathlib import Path


MODELS_ROOT = Path(__file__).parents[3] / "src" / "rosen" / "models"


def test_core_models_do_not_write_to_the_observation_port():
    violations: list[str] = []
    for path in sorted(MODELS_ROOT.rglob("*.py")):
        if path.name == "base.py" and path.parent == MODELS_ROOT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if (
                node.attr == "obp"
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                relative = path.relative_to(MODELS_ROOT)
                violations.append(f"{relative}:{node.lineno}")

    assert violations == [], (
        "rosen.models may expose BaseModel.obp, but core model implementations "
        "must not read from or write to it; add semantic observations in custom_models: "
        + ", ".join(violations)
    )

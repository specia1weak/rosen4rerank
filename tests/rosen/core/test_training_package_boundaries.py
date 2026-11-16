"""Architecture guards for the train/trainers dependency direction."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import rosen.core.train as train
import rosen.core.trainers as trainers


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PACKAGE = REPO_ROOT / "src" / "rosen" / "core" / "train"
ROSEN_PACKAGE = REPO_ROOT / "src" / "rosen"


def test_train_foundation_does_not_import_trainer_implementations():
    violations = []
    for path in sorted(TRAIN_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "rosen.core.trainers" or module.startswith("rosen.core.trainers."):
                    violations.append(f"{path.name}:{node.lineno} imports {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rosen.core.trainers" or alias.name.startswith("rosen.core.trainers."):
                        violations.append(f"{path.name}:{node.lineno} imports {alias.name}")

    assert violations == []


def test_public_packages_do_not_wrap_each_other_layers():
    assert not hasattr(train, "BaseTrainer")
    assert not hasattr(train, "EMATrainer")
    assert not hasattr(trainers, "EMAConfig")
    assert not hasattr(trainers, "EMAController")


def test_published_rosen_package_contains_no_test_code():
    test_files = {
        *ROSEN_PACKAGE.rglob("test*.py"),
        *ROSEN_PACKAGE.rglob("*_test.py"),
    }

    assert sorted(str(path.relative_to(REPO_ROOT)) for path in test_files) == []


def test_training_foundation_has_no_lifecycle_object_or_component_slot():
    assert not (TRAIN_PACKAGE / "lifecycle.py").exists()
    assert not hasattr(train, "TrainingLifecycle")
    assert "lifecycle" not in {item.name for item in fields(train.TrainerComponents)}

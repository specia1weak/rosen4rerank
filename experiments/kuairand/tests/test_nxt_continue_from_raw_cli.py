"""The public command must run with raw CSV and tracked code only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments.kuairand.nxt_continue_v1 import load_data
from experiments.kuairand.tests.test_nxt_continue_data import raw_source


def test_raw_cli_runs_without_historical_workspace(raw_source, tmp_path):
    repo = Path(__file__).resolve().parents[3]
    clean = tmp_path / "clean_checkout"
    clean.mkdir()
    for name in ("experiments", "src"):
        shutil.copytree(repo / name, clean / name, ignore=shutil.ignore_patterns("__pycache__"))
    assert not (clean / "workspace").exists()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(clean / "src"), str(clean))),
           "ROSEN_ROOT_DIR": str(clean), "POLARS_MAX_THREADS": "2"}
    command = [sys.executable, "-m", "experiments.kuairand.run_nxt_continue",
               "--source_dir", str(raw_source), "--experiment_name", "fresh",
               "--run_root", "workspace/runs", "--device", "cpu",
               "--batch_size", "4", "--num_threads", "1"]
    completed = subprocess.run(command, cwd=clean, env=env, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    data = clean / "workspace/kr_time_budget/data/nxt_continue_v1_raw"
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["build"]["kind"] == "raw_csv"
    assert {r["name"] for r in manifest["build"]["inputs"]} == {
        "log_fixture.csv", "video_features_basic_1k.csv", "user_features_1k.csv",
    }
    assert all(len(r["sha256"]) == 64 for r in manifest["build"]["inputs"])
    result = json.loads((clean / "workspace/runs/fresh/result.json").read_text())
    assert result["trained_optimizer_steps"] == 2
    assert result["data"]["train"]["rows"] == 8
    assert result["data"]["valid"]["rows"] == result["data"]["test"]["rows"] == 4
    assert 0 <= result["test"]["overall"]["auc"] <= 1
    assert "READ log_fixture.csv" in (data / "logs/prepare.log").read_text()
    assert not (data.parent / "v6_current_feedback").exists()
    assert not (data.parent / "nxt_continue_v1_aligned").exists()


def test_missing_data_points_to_raw_command_without_creating_schema(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="--source_dir"):
        load_data(missing)
    assert not missing.exists()

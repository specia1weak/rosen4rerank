import os
import subprocess
import sys


def test_installed_package_uses_calling_project(tmp_path):
    env = dict(os.environ)
    env.pop("ROSEN_ROOT_DIR", None)
    result = subprocess.check_output(
        [sys.executable, "-c", "from rosen.experiment import ROOT_DIR; print(ROOT_DIR)"],
        cwd=tmp_path, env=env, text=True,
    )
    assert result.strip() == str(tmp_path.resolve())


def test_explicit_experiment_root(tmp_path):
    env = dict(os.environ, ROSEN_ROOT_DIR=str(tmp_path / "experiment"))
    result = subprocess.check_output(
        [sys.executable, "-c", "from rosen.experiment import ROOT_DIR, WORKSPACE; print(ROOT_DIR); print(WORKSPACE)"],
        cwd=tmp_path, env=env, text=True,
    )
    assert result.splitlines() == [str(tmp_path / "experiment"), str(tmp_path / "experiment/workspace")]

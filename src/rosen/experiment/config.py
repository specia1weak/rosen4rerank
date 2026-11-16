"""Experiment paths belong to the calling project, not the installed package."""
import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("ROSEN_ROOT_DIR", Path.cwd())).expanduser().resolve()
WORKSPACE = ROOT_DIR / "workspace"

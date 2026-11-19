"""Rebuild four Repomix reading bundles and verify every embedded source file.

Run from any directory: uv run python docs/rosen/repomix/rebuild.py
Requires the local repomix CLI. Does not import or execute experiment scripts.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIGS = (
    "repomix.config.json",
    "repomix.multitask.config.json",
    "repomix.rerank.config.json",
    "repomix.original.config.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def embedded_files(document):
    """Read file entries without interpreting headings inside fenced source."""
    lines = document.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        header = re.fullmatch(r"## File: (.+)\n?", lines[index])
        index += 1
        if header is None:
            continue
        if index == len(lines):
            raise ValueError(f"Missing source fence: {header[1]}")
        opening = re.fullmatch(r"(`{3,}|~{3,})[^\n]*\n?", lines[index])
        if opening is None:
            raise ValueError(f"Invalid source fence: {header[1]}")
        index += 1
        start = index
        while index < len(lines) and lines[index].rstrip("\r\n") != opening[1]:
            index += 1
        if index == len(lines):
            raise ValueError(f"Unclosed source fence: {header[1]}")
        yield header[1], "".join(lines[start:index])
        index += 1


def main():
    executable = shutil.which("repomix")
    if executable is None:
        raise SystemExit("repomix CLI is required; it was not found on PATH")
    version = subprocess.check_output([executable, "--version"], text=True).strip()
    manifest = {"repomix_version": version, "source_comparison": "full text, ignoring file-edge whitespace only",
                "bundles": {}}
    for name in CONFIGS:
        config_path = HERE / name
        config = json.loads(config_path.read_text())
        subprocess.run([executable, "--config", str(config_path)], cwd=ROOT, check=True)
        output_path = ROOT / config["output"]["filePath"]
        packed = output_path.read_bytes()
        files = {}
        for relative, contents in embedded_files(packed.decode("utf-8")):
            if relative in files:
                raise ValueError(f"Duplicate file in {output_path.name}: {relative}")
            source = (ROOT / relative).read_bytes()
            if contents.strip() != source.decode("utf-8").strip():
                raise ValueError(f"Source content changed in {output_path.name}: {relative}")
            files[relative] = {"bytes": len(source), "sha256": digest(source)}
        expected = {str(path.relative_to(ROOT)) for pattern in config["include"]
                    for path in ROOT.glob(pattern) if path.is_file()}
        if set(files) != expected:
            raise ValueError(f"Source selection mismatch in {output_path.name}: "
                             f"missing={sorted(expected - set(files))}, extra={sorted(set(files) - expected)}")
        if not files:
            raise ValueError(f"Empty bundle: {output_path.name}")
        manifest["bundles"][output_path.name] = {
            "config": str(config_path.relative_to(ROOT)), "bytes": len(packed),
            "sha256": digest(packed), "file_count": len(files), "files": files,
        }
        print(f"VERIFIED {output_path.name}: {len(files)} full files, {len(packed):,} bytes", flush=True)
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("WROTE docs/rosen/repomix/manifest.json", flush=True)


if __name__ == "__main__":
    main()

"""Reproducible run folders with a manifest, source snapshot and file inventory."""

from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import uuid

from .learning import write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def initialize_run(output, settings):
    output = Path(output)
    root = Path(__file__).resolve().parent.parent
    snapshot = output / "source"
    snapshot.mkdir()
    for source in [*sorted((root / "cart_pendulum").glob("*.py")),
                   root / "pyproject.toml", root / "requirements-tested.txt"]:
        if source.is_file():
            target = snapshot / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    versions = {}
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "openai"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    write_json(output / "run.json", {"schema_version": 1, "run_id": uuid.uuid4().hex,
        "status": "running", "started_at": timestamp(), "settings": settings,
        "python": platform.python_version(), "platform": platform.system(),
        "machine": platform.machine(), "packages": versions, "git_revision": revision,
        "tracked_changes": dirty, "source_snapshot": "source",
        "note": "Source snapshot records the actual implementation, including uncommitted changes. No keys or environment variables are captured."})


def finalize_run(output, status, reason):
    output = Path(output)
    path = output / "run.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    manifest.update(status=status, finished_at=timestamp(), stop_reason=reason)
    inventory = []
    for file in sorted(output.rglob("*")):
        if file.is_file() and file.name not in ("run.json", "files.json") and not file.name.endswith(".tmp"):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            inventory.append({"path": str(file.relative_to(output)), "bytes": file.stat().st_size,
                              "sha256": digest.hexdigest()})
    write_json(output / "files.json", inventory)
    write_json(path, manifest)


def recorded_run(function):
    @wraps(function)
    def wrapped(output, *args, **kwargs):
        # An existing folder belongs to an earlier run; never rewrite its status.
        existed = Path(output).exists()
        try:
            result = function(output, *args, **kwargs)
        except BaseException as error:
            if not existed:
                finalize_run(output, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", type(error).__name__)
            raise
        reason = result["stop_reason"]
        status = ("failed" if reason.startswith("error:") else
                  "completed" if reason in ("trial_limit", "block_limit") else "stopped")
        finalize_run(output, status, reason)
        return result
    return wrapped

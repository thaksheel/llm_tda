"""Shared gradient and checkpoint score cache locations."""

import hashlib
import json
import os
from pathlib import Path


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _publish(directory, descriptor):
    pending = directory / "gradient_cache.json.tmp"
    pending.write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    pending.replace(directory / "gradient_cache.json")


def directory_in_use(directory):
    """Check Linux process descriptors to avoid concurrent cache writers."""
    proc = Path("/proc")
    if not proc.is_dir():
        return False  # Windows also rejects renames when open handles prohibit them.
    prefix = str(directory.resolve()) + os.sep
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    if os.readlink(descriptor).startswith(prefix):
                        return True
                except OSError:
                    continue
        except OSError:
            continue
    return False


def _within(root, path):
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Cache path must stay under {root}: {path}")


def gradient_directory(output_dir, identities):
    """Locate or create a cache for matching gradient inputs."""
    root = Path(output_dir)
    destination = root / "gradients" / cache_key(identities)
    _within(root, destination)
    if destination.exists():
        if directory_in_use(destination):
            raise RuntimeError(f"Shared gradients are in use by another process: {destination}")
        descriptor = destination / "gradient_cache.json"
        saved = _read(descriptor) if descriptor.is_file() else {}
        if saved and saved.get("identities") != identities:
            raise ValueError("Shared gradient cache identity mismatch")
    else:
        destination.mkdir(parents=True, exist_ok=True)
    _publish(destination, {"identities": identities})
    return destination


def score_directory(output_dir, identity):
    destination = Path(output_dir) / "tracin" / "checkpoint_scores" / cache_key(identity)
    _within(Path(output_dir), destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination

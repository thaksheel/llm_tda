"""Sample-level RapidIn: per-parameter L2 normalization and signed block sums.

Compression follows huawei-lin/RapidIn's RapidGrad.py: shared seed-42 signs,
factor-based row/column shuffling, and zero padding to a multiple of 2**24.
Raw gradients and compressed features retain float32 precision in this pipeline.
"""

import json
import math
from pathlib import Path

from .gradient_cache import cache_key


COMPRESSION = {"dimensions": 2**16, "seed": 42, "shuffles": 20,
               "padding_multiple": 2**24, "dtype": "float32",
               "implementation": "reference-factor-shuffle-signed-sum-v1"}


def feature_identity(gradients, device):
    return {"version": "rapidin-sample-v1", "gradients": gradients,
            "compression": dict(COMPRESSION), "device": str(device),
            "normalization": "per-parameter-L2-eps-1e-12-before-padding",
            "post_projection_normalization": "none"}


def parameter_spans(parameter_groups, dimensions):
    """Recover parameter boundaries from the raw cache's exact flattening order."""
    spans, names, offset = [], set(), 0
    for group in parameter_groups:
        for parameter in group:
            name, shape = parameter["name"], parameter["shape"]
            if name in names or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError("RapidIn needs unique parameter names and valid shapes")
            names.add(name)
            stop = offset + math.prod(shape)
            spans.append((offset, stop))
            offset = stop
    if not spans or offset != dimensions:
        raise ValueError("RapidIn parameter mapping does not match raw gradient dimensions")
    return spans


def compression_plan(dimensions, config):
    """Reproduce reference RNG draw order without changing NumPy's global RNG."""
    import numpy as np

    width, multiple = config["dimensions"], config["padding_multiple"]
    if dimensions <= 0 or width <= 0 or multiple <= 0 or config["shuffles"] < 0:
        raise ValueError("RapidIn dimensions/padding must be positive and shuffles nonnegative")
    padded = math.ceil(dimensions / multiple) * multiple
    if padded % width or padded < width:
        raise ValueError("RapidIn padded dimension must be divisible by its feature dimension")
    rng = np.random.RandomState(config["seed"])
    signs = rng.randint(0, 2, (padded,), dtype=np.int8)
    signs[signs == 0] = -1
    # The reference draws reshaping dimensions from products of shuffled prime factors.
    remaining, factors, divisor = padded, [], 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        factors.append(remaining)
    transforms = []
    for _ in range(config["shuffles"] * 2):
        count = rng.randint(len(factors) // 4, len(factors) // 2 + 1)
        rng.shuffle(factors)
        size = int(np.prod(factors[:count], dtype=np.int64))
        transforms.append((size, rng.permutation(size)))
    return padded, signs, transforms


class RapidProjector:
    """Compose the shared shuffles once, then apply the same sketch to every row."""

    def __init__(self, dimensions, parameter_groups, config, device):
        import torch

        self.spans = parameter_spans(parameter_groups, dimensions)
        self.dimensions, self.width = dimensions, config["dimensions"]
        self.padded, signs, transforms = compression_plan(dimensions, config)
        self.signs = torch.from_numpy(signs).to(device=device, dtype=torch.float32)
        indices = torch.arange(self.padded, device=device)
        for index, (size, permutation) in enumerate(transforms):
            permutation = torch.from_numpy(permutation).to(device)
            if index % 2 == 0:
                indices = indices.reshape(size, -1).index_select(0, permutation).reshape(-1)
            else:
                indices = indices.reshape(-1, size).index_select(1, permutation).reshape(-1)
        self.indices = indices

    def project(self, gradients):
        import torch.nn.functional as F

        if gradients.ndim != 2 or gradients.shape[1] != self.dimensions:
            raise ValueError("RapidIn input dimensions differ from the compression plan")
        normalized = gradients.clone()
        for start, stop in self.spans:
            normalized[:, start:stop] = F.normalize(gradients[:, start:stop], dim=1, eps=1e-12)
        values = F.pad(normalized, (0, self.padded - self.dimensions))
        values = values.index_select(1, self.indices) * self.signs
        return values.reshape(len(values), self.width, self.padded // self.width).sum(dim=2)


def _write_json(path, value):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def compressed_features(output_dir, matrices, identity, parameter_groups, device,
                        *, projector_factory=RapidProjector, progress_factory=None):
    """Reuse or resume compressed features; never modify the raw gradient cache."""
    import shutil
    import numpy as np
    import torch
    if progress_factory is None:
        from tqdm import tqdm
        progress_factory = tqdm

    directory = Path(output_dir) / "rapid_in" / "features" / cache_key(identity)
    directory.mkdir(parents=True, exist_ok=True)
    config = identity["compression"]
    width, projector, results = config["dimensions"], None, []
    parameter_spans(parameter_groups, matrices[0].shape[1])
    for kind, raw in zip(("source", "query"), matrices, strict=True):
        if raw.ndim != 2 or raw.shape[1] != matrices[0].shape[1]:
            raise ValueError("RapidIn source and query gradient dimensions must match")
        path = directory / f"{kind}_features.npy"
        progress_path = directory / f"{kind}_progress.json"
        shape, matrix, completed = (len(raw), width), None, 0
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            completed = progress["completed_rows"]
            if (progress["identity"] != identity or type(completed) is not int
                    or not 0 <= completed <= len(raw)):
                raise ValueError("RapidIn feature inputs or saved row progress changed")
            matrix = np.load(path, mmap_mode="r" if completed == len(raw) else "r+", allow_pickle=False)
            if (matrix.shape != shape or matrix.dtype != np.float32
                    or matrix.offset + matrix.nbytes != path.stat().st_size):
                raise ValueError("Invalid RapidIn feature file")
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            matrix, completed = None, 0
        if matrix is None:
            required = math.prod(shape) * 4 + 128
            existing = path.stat().st_size if path.is_file() else 0
            if max(0, required - existing) + 64 * 2**20 > shutil.disk_usage(directory).free:
                raise OSError("Insufficient disk space for RapidIn compressed features")
            progress_path.unlink(missing_ok=True)
            matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            _write_json(progress_path, {"identity": identity, "completed_rows": 0})
        if completed == len(raw):
            print(f"Reusing RapidIn {kind} features: {completed}/{len(raw)} rows", flush=True)
            results.append(matrix)
            continue
        if projector is None:
            projector = projector_factory(raw.shape[1], parameter_groups, config, device)
        with progress_factory(total=len(raw), initial=completed, desc=f"RapidIn {kind} compression") as progress:
            for start in range(completed, len(raw), 8):
                stop = min(start + 8, len(raw))
                batch = torch.from_numpy(np.array(raw[start:stop], copy=True)).to(device)
                features = projector.project(batch)
                if not torch.isfinite(features).all():
                    raise ValueError("Nonfinite RapidIn compressed features")
                matrix[start:stop] = features.cpu().numpy()
                matrix.flush()
                _write_json(progress_path, {"identity": identity, "completed_rows": stop})
                progress.update(stop - start)
        results.append(matrix)
    _write_json(directory / "feature_metadata.json", identity)
    return directory, results

"""LESS source Adam directions and query gradients, projected and normalized.

Follows princeton-nlp/LESS collect_grad_reps.py and matching.py: source-only
Adam moments without bias correction, epsilon inside sqrt, seed-0 Rademacher
projection, then row normalization. Calculations and saved features use float32.
"""

import json
import math
from pathlib import Path

from .gradient_cache import cache_key


PROJECTION = {"dimensions": 8192, "seed": 0, "block_size": 128,
              "type": "rademacher", "dtype": "float32", "model_id": 0,
              "implementation": "bundled-trak-basic-v1"}


def feature_identity(gradients, optimizer_sha256, device):
    return {"version": "less-reference-adam-cosine-v1", "gradients": gradients,
            "optimizer_sha256": optimizer_sha256, "projection": dict(PROJECTION),
            "device": str(device), "source_transform": "saved-adam-moments-no-bias-correction",
            "query_transform": "raw-gradient", "epsilon": "inside-sqrt",
            "normalization": "project-then-row-l2-eps-1e-12"}


def load_moments(checkpoint, parameter_groups, device):
    """Align Adam moments with the saved raw-gradient optimizer parameter order."""
    import torch

    state = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=True)
    if len(state["param_groups"]) != len(parameter_groups):
        raise ValueError("LESS optimizer groups do not match the gradient parameter mapping")
    first, second, groups = [], [], []
    offset = 0
    for saved, mapping in zip(state["param_groups"], parameter_groups, strict=True):
        if saved["params"] != [p["optimizer_id"] for p in mapping]:
            raise ValueError("LESS optimizer parameter order does not match raw gradients")
        betas, eps = saved.get("betas"), saved.get("eps")
        if (betas is None or len(betas) != 2 or any(not 0 <= b < 1 for b in betas)
                or eps is None or not math.isfinite(eps) or eps <= 0
                or saved.get("amsgrad", False) or saved.get("maximize", False)):
            raise ValueError("LESS requires ordinary Adam/AdamW moments, betas, and positive epsilon")
        start = offset
        for parameter in mapping:
            moments = state["state"][parameter["optimizer_id"]]
            for key, target in (("exp_avg", first), ("exp_avg_sq", second)):
                value = moments.get(key)
                if (value is None or list(value.shape) != parameter["shape"]
                        or not torch.isfinite(value).all() or (key == "exp_avg_sq" and (value < 0).any())):
                    raise ValueError(f"LESS requires valid {key} for {parameter['name']}")
                target.append(value.reshape(-1).float())
            offset += math.prod(parameter["shape"])
        if offset > start:
            groups.append((start, offset, float(betas[0]), float(betas[1]), float(eps)))
    if not first:
        raise ValueError("LESS requires nonempty optimizer moments")
    return torch.cat(first).to(device), torch.cat(second).to(device), groups


def adam_direction(gradients, moments):
    """Independent hypothetical Adam update per source; never mutate saved moments."""
    first, second, groups = moments
    result = gradients.clone()
    for start, stop, beta1, beta2, eps in groups:
        g = gradients[:, start:stop]
        result[:, start:stop] = (beta1 * first[start:stop] + (1 - beta1) * g) / (
            beta2 * second[start:stop] + (1 - beta2) * g.square() + eps
        ).sqrt()
    return result


def _write_json(path, value):
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def projected_features(output_dir, matrices, identity, checkpoint, parameter_groups, device,
                       *, projector_factory=None, progress_factory=None):
    """Resume batched projection without copying or modifying raw gradient files."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    if progress_factory is None:
        from tqdm import tqdm
        progress_factory = tqdm
    directory = Path(output_dir) / "less" / "features" / cache_key(identity)
    directory.mkdir(parents=True, exist_ok=True)
    projection = identity["projection"]
    width = projection["dimensions"]
    projector, moments, results = None, None, []
    try:
        for kind, raw in zip(("source", "query"), matrices, strict=True):
            path = directory / f"{kind}_features.npy"
            progress_path = directory / f"{kind}_progress.json"
            shape = (len(raw), width)
            matrix, completed = None, 0
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                completed = progress["completed_rows"]
                if (progress["identity"] != identity or type(completed) is not int
                        or not 0 <= completed <= len(raw)):
                    raise ValueError("LESS feature inputs or row progress changed")
                matrix = np.load(path, mmap_mode="r" if completed == len(raw) else "r+", allow_pickle=False)
                if (matrix.shape != shape or matrix.dtype != np.float32
                        or matrix.offset + matrix.nbytes != path.stat().st_size):
                    raise ValueError("Invalid LESS feature file")
            except (OSError, ValueError, KeyError, TypeError):
                matrix = None
                completed = 0
            if matrix is None:
                progress_path.unlink(missing_ok=True)
                matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
                _write_json(progress_path, {"identity": identity, "completed_rows": 0})
            if completed == len(raw):
                print(f"Reusing LESS {kind} features: {completed}/{len(raw)} rows", flush=True)
                results.append(matrix)
                continue
            if projector is None:
                if projector_factory is None:
                    from modules import BasicProjector, ProjectionType
                    projector_factory = lambda **kwargs: BasicProjector(
                        proj_type=ProjectionType.rademacher, **kwargs)
                projector = projector_factory(grad_dim=raw.shape[1], proj_dim=width,
                    seed=projection["seed"], block_size=projection["block_size"],
                    device=torch.device(device), dtype=torch.float32)
            if kind == "source" and moments is None:
                moments = load_moments(checkpoint, parameter_groups, device)
                if len(moments[0]) != raw.shape[1]:
                    raise ValueError("LESS optimizer moment dimensions differ from raw gradients")
            with progress_factory(total=len(raw), initial=completed, desc=f"LESS {kind} projection") as progress:
                for start in range(completed, len(raw), 16):
                    stop = min(start + 16, len(raw))
                    batch = torch.from_numpy(np.array(raw[start:stop], copy=True)).to(device)
                    if kind == "source":
                        batch = adam_direction(batch, moments)
                    batch = projector.project(batch, model_id=projection["model_id"])
                    if not torch.isfinite(batch).all():
                        raise ValueError("Nonfinite LESS projected features")
                    batch = F.normalize(batch.float(), dim=1, eps=1e-12)
                    matrix[start:stop] = batch.cpu().numpy()
                    matrix.flush()
                    _write_json(progress_path, {"identity": identity, "completed_rows": stop})
                    progress.update(stop - start)
            results.append(matrix)
        _write_json(directory / "feature_metadata.json", identity)
        return directory, results
    finally:
        if projector is not None:
            projector.free_memory()

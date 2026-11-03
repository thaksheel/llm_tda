"""TrackStar adapted to saved LoRA parameters and AdamW second moments."""

import json
import math
from pathlib import Path
import re

from .gradient_cache import cache_key
from .less import load_moments, _write_json


SETTINGS = {"layer_blocks": 8, "projection_side": 64, "projection_seed": 0,
            "projection": "two-sided-gaussian", "eval_mix": 0.9,
            "second_moment": "saved-adamw-exp-avg-sq-sqrt-plus-eps",
            "loss": "response-token-sum", "dtype": "float32",
            "solver": "block-inverse-square-root-pseudoinverse",
            "tolerance": "max(rows,dimension)*float32-eps-on-covariance",
            "normalization": "global-l2-after-whitening", "batch_size": 16}


def feature_identity(gradients, optimizer_sha256, device):
    return {"version": "trackstar-lora-adamw-v1", "gradients": gradients,
            "optimizer_sha256": optimizer_sha256, "settings": dict(SETTINGS), "device": str(device)}


def parameter_blocks(parameter_groups, maximum=8):
    """Pool consecutive transformer layers; separate attention from MLP matrices."""
    parameters, layers, offset = [], set(), 0
    names = set()
    for group in parameter_groups:
        for p in group:
            name, shape = p["name"], p["shape"]
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if (not match or len(shape) != 2 or min(shape) <= 0 or name in names
                    or not (".lora_A." in name or ".lora_B." in name)):
                raise ValueError(f"TrackStar requires unique mapped transformer LoRA A/B matrices: {name}")
            family = "attention" if ".self_attn." in name else "mlp" if ".mlp." in name else None
            if family is None:
                raise ValueError(f"Unsupported TrackStar transformer module: {name}")
            transpose = ".lora_A." in name
            rows, columns = (shape[1], shape[0]) if transpose else shape
            parameters.append(dict(name=name, shape=shape, start=offset, stop=offset + math.prod(shape),
                                   layer=int(match[1]), family=family, transpose=transpose,
                                   rows=rows, columns=columns))
            layers.add(int(match[1]))
            names.add(name)
            offset += math.prod(shape)
    if not layers or maximum <= 0:
        raise ValueError("TrackStar requires nonempty layers and positive layer_blocks")
    ordered = sorted(layers)
    count = min(maximum, len(ordered))
    groups = {}
    for p in parameters:
        number = min(count - 1, ordered.index(p["layer"]) * count // len(ordered))
        key = f"block-{number:02d}-{p['family']}"
        groups.setdefault(key, []).append(p)
    result = []
    for key, entries in sorted(groups.items()):
        if len({p["columns"] for p in entries}) != 1:
            raise ValueError("TrackStar needs a common LoRA rank within each pooled block")
        result.append({"name": key, "parameters": entries, "rows": sum(p["rows"] for p in entries),
                       "columns": entries[0]["columns"]})
    return result


class Projector:
    def __init__(self, blocks, settings, device):
        import torch
        self.blocks, self.matrices = blocks, []
        side = settings["projection_side"]
        for block in blocks:
            values = []
            for label, width in (("left", block["rows"]), ("right", block["columns"])):
                seed = int(cache_key([settings["projection_seed"], block["name"], label])[:16], 16)
                generator = torch.Generator(device="cpu").manual_seed(seed)
                values.append((torch.randn(width, side, generator=generator, device="cpu",
                                           dtype=torch.float32) / math.sqrt(side)).to(device))
            self.matrices.append(values)

    def project(self, corrected):
        import torch
        results = []
        for block, (left, right) in zip(self.blocks, self.matrices, strict=True):
            value = corrected.new_zeros(len(corrected), left.shape[1], right.shape[1])
            offset = 0
            for p in block["parameters"]:
                g = corrected[:, p["start"]:p["stop"]].reshape(len(corrected), *p["shape"])
                if p["transpose"]:
                    g = g.transpose(1, 2)
                value += left[offset:offset + p["rows"]].T @ (g @ right)
                offset += p["rows"]
            results.append(value.flatten(1))
        return torch.cat(results, dim=1)


def whitening_basis(source, query, mix):
    """R=(1-mix) S.T S + mix Q.T Q; discard only numerical null directions."""
    import torch
    if (source.ndim != 2 or query.ndim != 2 or source.shape[1] != query.shape[1]
            or not 0 <= mix < 1 or min(source.shape) == 0 or len(query) == 0
            or not torch.isfinite(source).all() or not torch.isfinite(query).all()):
        raise ValueError("TrackStar requires finite source/query matrices and 0 <= eval_mix < 1")
    dimension = source.shape[1]
    values = torch.cat([math.sqrt(1 - mix) * source, math.sqrt(mix) * query])
    tolerance = max(values.shape) * torch.finfo(values.dtype).eps
    # Fit the inverse root from the weighted design instead of forming its Gram
    # matrix: projection can be ill-conditioned, especially in LoRA rank space.
    _, singular, vh = torch.linalg.svd(values, full_matrices=False)
    eigenvalues, vectors = singular.square(), vh.T
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("Nonfinite TrackStar curvature spectrum")
    keep = eigenvalues > eigenvalues.max().clamp_min(0) * tolerance
    # An all-zero block yields an empty basis and contributes zero to every norm.
    return vectors[:, keep] / eigenvalues[keep].sqrt()


def _matrix(path, shape, completed, identity, progress_path):
    import numpy as np
    try:
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("TrackStar cache identity changed")
        rows = saved[completed]
        matrix = np.load(path, mmap_mode="r+", allow_pickle=False)
        if (matrix.shape != shape or matrix.dtype != np.float32
                or matrix.offset + matrix.nbytes != path.stat().st_size):
            raise ValueError("Invalid TrackStar feature matrix")
        return matrix, rows
    except FileNotFoundError:
        return None, 0


def projected_features(output_dir, matrices, identity, checkpoint, parameter_groups, device,
                       *, count_factory, progress_factory=None):
    """Cache projection rows, whitening blocks, and final globally normalized features."""
    import numpy as np
    import torch
    if progress_factory is None:
        from tqdm import tqdm
        progress_factory = tqdm
    settings = identity["settings"]
    blocks = parameter_blocks(parameter_groups, settings["layer_blocks"])
    side = settings["projection_side"]
    block_width, width = side**2, len(blocks) * side**2
    directory = Path(output_dir) / "trackstar" / "features" / cache_key(identity)
    directory.mkdir(parents=True, exist_ok=True)
    signature = {"identity": identity, "blocks": blocks, "width": width}
    complete = directory / "complete.json"
    if complete.is_file() and json.loads(complete.read_text(encoding="utf-8")) == signature:
        results = [np.load(directory / f"{kind}_features.npy", mmap_mode="r", allow_pickle=False)
                   for kind in ("source", "query")]
        if any(m.shape != (len(raw), width) or m.dtype != np.float32 for m, raw in zip(results, matrices, strict=True)):
            raise ValueError("Invalid completed TrackStar feature cache")
        print("Reusing TrackStar projected, whitened, normalized features", flush=True)
        return directory, results
    projected, projector, denominator = [], None, None
    for kind, raw in zip(("source", "query"), matrices, strict=True):
        path = directory / f"{kind}_projected.npy"
        progress_path = directory / f"{kind}_projection.json"
        matrix, completed = _matrix(path, (len(raw), width), "completed_rows", signature, progress_path)
        if type(completed) is not int or not 0 <= completed <= len(raw):
            raise ValueError("Invalid TrackStar projection progress")
        if matrix is None:
            matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(raw), width))
            _write_json(progress_path, {"identity": signature, "completed_rows": 0})
        if completed < len(raw):
            if projector is None:
                projector = Projector(blocks, settings, device)
                first, second, groups = load_moments(checkpoint, parameter_groups, device)
                del first
                denominator = second.sqrt()
                for start, end, _, _, eps in groups:
                    denominator[start:end].add_(eps)
                if len(denominator) != raw.shape[1]:
                    raise ValueError("TrackStar optimizer and raw gradient dimensions differ")
            counts = np.asarray(count_factory(kind))
            if counts.shape != (len(raw),) or not np.isfinite(counts).all() or (counts <= 0).any():
                raise ValueError("TrackStar needs positive response-token counts")
            with progress_factory(total=len(raw), initial=completed, desc=f"TrackStar {kind} projection") as progress:
                for start in range(completed, len(raw), settings["batch_size"]):
                    end = min(start + settings["batch_size"], len(raw))
                    batch = torch.from_numpy(np.array(raw[start:end], copy=True)).to(device)
                    batch *= torch.as_tensor(counts[start:end], dtype=batch.dtype, device=device)[:, None]
                    values = projector.project(batch / denominator)
                    if not torch.isfinite(values).all():
                        raise ValueError("Nonfinite TrackStar projected gradients")
                    matrix[start:end] = values.cpu().numpy()
                    matrix.flush()
                    _write_json(progress_path, {"identity": signature, "completed_rows": end})
                    progress.update(end - start)
                    del batch, values
        projected.append(matrix)
    del projector, denominator
    progress_path = directory / "whitening_progress.json"
    if any(not (directory / f"{kind}_whitened.npy").is_file() for kind in ("source", "query")):
        progress_path.unlink(missing_ok=True)
    whitened, completed = [], None
    for kind, raw in zip(("source", "query"), matrices, strict=True):
        path = directory / f"{kind}_whitened.npy"
        matrix, blocks_done = _matrix(path, (len(raw), width), "completed_blocks", signature, progress_path)
        if matrix is None:
            matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(raw), width))
            blocks_done = 0
        completed = blocks_done if completed is None else min(completed, blocks_done)
        whitened.append(matrix)
    if type(completed) is not int or not 0 <= completed <= len(blocks):
        raise ValueError("Invalid TrackStar whitening progress")
    with torch.no_grad(), progress_factory(total=len(blocks), initial=completed, desc="TrackStar whitening blocks") as progress:
        for number in range(completed, len(blocks)):
            start, end = number * block_width, (number + 1) * block_width
            values = [torch.from_numpy(np.array(m[:, start:end], copy=True)).to(device) for m in projected]
            basis = whitening_basis(*values, settings["eval_mix"])
            for target, value in zip(whitened, values, strict=True):
                # Pad the retained coordinates to keep fixed block offsets. Norms
                # and inner products equal those from the symmetric inverse root.
                target[:, start:end] = 0
                target[:, start:start + basis.shape[1]] = (value @ basis).cpu().numpy()
                target.flush()
            _write_json(progress_path, {"identity": signature, "completed_blocks": number + 1})
            progress.update(1)
            del values, basis, value
    results = []
    for kind, matrix in zip(("source", "query"), whitened, strict=True):
        result = np.lib.format.open_memmap(
            directory / f"{kind}_features.npy", mode="w+", dtype=np.float32, shape=matrix.shape,
        )
        for start in range(0, len(matrix), settings["batch_size"]):
            values = np.array(matrix[start:start + settings["batch_size"]], copy=True)
            norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
            result[start:start + len(values)] = values / np.where(norms > 0, norms, 1)
        result.flush()
        results.append(result)
    _write_json(complete, signature)
    return directory, results

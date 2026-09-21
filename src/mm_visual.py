import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, List, Literal, Optional, Any, Tuple
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datasets import load_dataset, DatasetDict
import random
from scipy.stats import kstest, ks_2samp
from tqdm import tqdm
from dataclasses import dataclass

from .rept_mm import ReptMM
from .tracein_mm import TraceInMM


@dataclass
class Stats:
    per_dim_mean: float
    per_dim_var: float
    grad_norm_mean: float
    grad_norm_std: float


@dataclass
class MMV_Metrics:
    modality: str
    hessian: float
    smoothness: float
    sparsity: float
    representation_stats: Stats
    gradient_stats: Stats


class MultimodalVisual:
    def __init__(self):
        self.rem = ReptMM()
        self.tim = TraceInMM()

    def sample_flickr(self, n: int = 10) -> Tuple[Any, str]:
        """Note: len(ds['train'])=6000 and len(ds['test'])=1000"""
        ds: DatasetDict = load_dataset("jxie/flickr8k")
        samples = []
        for i in random.sample(range(len(ds["train"])), n):
            img = ds["train"][i]["image"]
            caption = ds["train"][i]["caption_0"]  # has up to 5 captions
            samples.append((img, caption))
        return samples

    def tracein(
        self,
        model,
        processor,
        datasets_by_modality: Dict[str, Dict[str, List[str]]],
        device: str = "cuda",
    ):
        device_t = torch.device(device)
        model.to(device_t)
        grads = {}
        for modality, ds in datasets_by_modality.items():
            modality_grads = []
            z = zip(
                ds["prompts"],
                ds["response"],
                ds.get("images", [None] * len(ds["prompts"])),
            )
            for prompt, response, img_path in tqdm(
                z, total=len(ds["prompts"]), desc=f"Running {modality}: "
            ):
                gv = self.tim.get_grads(
                    model,
                    processor,
                    modality,
                    prompt,
                    response,
                    device_t,
                    image_path=img_path,
                )
                modality_grads.append(gv)
            grads[modality] = np.stack(modality_grads)
        return grads

    def rept(
        self,
        model,
        processor,
        datasets_by_modality: Dict[str, Dict[str, List[str]]],
        layer: int,
        device: str = "cuda",
    ):
        device_t = torch.device(device)
        model.to(device_t)
        reps = {}
        grads = {}
        for modality, ds in datasets_by_modality.items():
            modality_reps = []
            modality_grads = []
            z = zip(
                ds["prompts"],
                ds["response"],
                ds.get("images", [None] * len(ds["prompts"])),
            )
            for prompt, response, img_path in tqdm(
                z, total=len(ds["prompts"]), desc=f"Running {modality}: "
            ):
                rep = self.rem.get_grad_reps(
                    model,
                    processor,
                    modality,
                    prompt,
                    response,
                    layer,
                    device_t,
                    image_path=img_path,
                )
                # split back into H and g_H if you want separate arrays
                dim = rep.shape[0] // 2
                modality_reps.append(rep[:dim])
                modality_grads.append(rep[dim:])
            reps[modality] = np.stack(modality_reps)
            grads[modality] = np.stack(modality_grads)
        return reps, grads

    # |----------------->><<-----------------|
    # |------------Visual Methods------------|
    # |----------------->><<-----------------|
    def plot_space_norms_stats(self, reps: Dict[str, np.ndarray]):
        fig, axes = plt.subplots(1, len(reps), figsize=(5 * len(reps), 4))
        if len(reps) == 1:
            axes = [axes]
        for ax, (modality, X) in zip(axes, reps.items()):
            norms = np.linalg.norm(X, axis=1)
            sns.histplot(norms, kde=True, ax=ax)
            ax.set_title(f"Modality: {modality}")
            ax.set_xlabel("Norm")
        plt.tight_layout()
        plt.show()

    def get_embedding_stats(self, reps: Dict[str, np.ndarray]):
        ms = reps.keys()
        stats = dict(
            zip(
                ms,
                [{} for _ in ms],
            )
        )
        for modality, X in reps.items():
            norms = np.linalg.norm(X, axis=1)
            stats[modality]["norms"] = norms
            stats[modality]["grad_mean_norm"] = norms.mean()
            stats[modality]["grad_std_norm"] = norms.std()
            stats[modality]["per_dim_mean"] = X.mean(axis=0).mean()
            stats[modality]["per_dim_var"] = X.var(axis=0).mean()
        return stats

    def plot_space_pca_tsne(
        self, reps: Dict[str, np.ndarray], method: Literal["pca", "tsne"], label: str
    ):
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = sns.color_palette("tab10", len(reps))
        for (modality, X), c in zip(reps.items(), colors):
            if method == "pca":
                reducer = PCA(n_components=2)
            elif method == "tsne":
                reducer = TSNE(
                    n_components=2, perplexity=30, init="random", learning_rate="auto"
                )
            else:
                raise ValueError("method must be 'pca' or 'tsne'")
            Z = reducer.fit_transform(X)
            ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.6, label=modality, color=c)
        ax.set_title(f"{label}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_gradient_manifold(self, grads, method="pca"):
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = sns.color_palette("tab10", len(grads))
        for (modality, G), c in zip(grads.items(), colors):
            reducer = PCA(n_components=2) if method == "pca" else TSNE(n_components=2)
            Z = reducer.fit_transform(G)
            ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.6, label=modality, color=c)
        ax.set_title(f"Gradient manifold ({method.upper()})")
        ax.legend()
        plt.show()


    def plot_gradient_norms_stats(self, grads: Dict[str, np.ndarray], label: str):
        # ---- Compute global min/max across all modalities ----
        all_norms = []
        for G in grads.values():
            G_flat = G.reshape(G.shape[0], -1) if G.ndim > 2 else G
            norms = np.linalg.norm(G_flat, axis=1)
            all_norms.append(norms)

        all_norms = np.concatenate(all_norms)
        global_min = all_norms.min()
        global_max = all_norms.max()

        # ---- Compute global y-axis max using consistent bins ----
        # Choose number of bins (same as seaborn default)
        bins = 30
        bin_edges = np.linspace(global_min, global_max, bins + 1)
        global_ymax = 0
        for norms in all_norms.reshape(-1, 1):
            counts, _ = np.histogram(all_norms, bins=bin_edges)
            global_ymax = max(global_ymax, counts.max())
        fig, axes = plt.subplots(1, len(grads), figsize=(5 * len(grads), 4))
        if len(grads) == 1:
            axes = [axes]

        for ax, (modality, G) in zip(axes, grads.items()):
            G_flat = G.reshape(G.shape[0], -1) if G.ndim > 2 else G
            norms = np.linalg.norm(G_flat, axis=1)

            sns.histplot(norms, kde=True, bins=bins, ax=ax)
            ax.set_title(f"{label} Modality: {modality}")
            ax.set_xlabel("Norm")
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(0, global_ymax/1.5)

            # ---- Compute mean and add red dashed line ----
            mean_val = norms.mean()
            if mean_val < 1e-4:
                label_text = f"mean = {mean_val:.2e}"
            else:
                label_text = f"mean = {mean_val:.2f}"

            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=label_text,
            )
            ax.legend()

        plt.tight_layout()
        plt.show()

    def get_gradient_stats(self, grads: Dict[str, np.ndarray]):
        ms = grads.keys()
        stats = dict(
            zip(
                ms,
                [{} for _ in ms],
            )
        )
        for modality, G in grads.items():
            if G.ndim > 2:
                G_flat = G.reshape(G.shape[0], -1)
            else:
                G_flat = G
            norms = np.linalg.norm(G_flat, axis=1)
            stats[modality]["norms"] = norms
            stats[modality]["grad_mean_norm"] = norms.mean()
            stats[modality]["grad_std_norm"] = norms.std()
            stats[modality]["per_dim_mean"] = G_flat.mean(axis=0).mean()
            stats[modality]["per_dim_var"] = G_flat.var(axis=0).mean()
        return stats

    def plot_gradient_directionality(
        self, grads: Dict[str, np.ndarray], n_samples: int = 100
    ):
        fig, axes = plt.subplots(1, len(grads), figsize=(5 * len(grads), 4))
        if len(grads) == 1:
            axes = [axes]
        for ax, (modality, G) in zip(axes, grads.items()):
            if G.ndim > 2:
                G_flat = G.reshape(G.shape[0], -1)
            else:
                G_flat = G
            # subsample
            idx = np.random.choice(
                G_flat.shape[0], min(n_samples, G_flat.shape[0]), replace=False
            )
            G_sub = G_flat[idx]
            # pairwise cosine similarities
            G_norm = G_sub / (np.linalg.norm(G_sub, axis=1, keepdims=True) + 1e-8)
            cos_sim = G_norm @ G_norm.T
            # take upper triangle (excluding diagonal)
            iu = np.triu_indices_from(cos_sim, k=1)
            vals = cos_sim[iu]
            sns.histplot(vals, kde=True, ax=ax)
            ax.set_title(f"{modality} gradient directionality (pairwise cos)")
            ax.set_xlabel("cos(g_i, g_j)")
        plt.tight_layout()
        plt.show()

    def collect_layerwise_grad_norms(
        self,
        model,
        processor,
        datasets_by_modality: Dict[str, Dict[str, List[str]]],
        layers: List[int],
        device: str = "cuda",
    ):
        model_device = torch.device(device)
        model.to(model_device)
        layerwise = {
            modality: {layer: [] for layer in layers}
            for modality in datasets_by_modality
        }
        for modality, ds in datasets_by_modality.items():
            z = zip(ds["prompts"], ds["response"], ds["images"])
            for prompt, response, image in tqdm(
                z, total=len(ds["prompts"]), desc=f"Running {modality} layer sens: "
            ):
                for layer in layers:
                    gH = self.rem.get_mm_representation_gradient(
                        model=model,
                        processor=processor,
                        modality=modality,
                        prompt=prompt,
                        expected_response=response,
                        layer=layer,
                        device=torch.device(device),
                        image_path=image,
                    )
                    if gH.ndim > 1:
                        g_flat = gH.reshape(-1)
                    else:
                        g_flat = gH
                    layerwise[modality][layer].append(np.linalg.norm(g_flat))
        # convert to arrays
        for modality in layerwise:
            for layer in layers:
                layerwise[modality][layer] = np.array(layerwise[modality][layer])
        return layerwise

    def plot_layerwise_sensitivity(
        self, layerwise: Dict[str, Dict[int, np.ndarray]], layers: List[int]
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for modality, layer_dict in layerwise.items():
            means = [layer_dict[layer].mean() for layer in layers]
            ax.plot(layers, means, marker="o", label=modality)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Mean gradient norm")
        ax.set_title("Layer-wise gradient sensitivity across modalities")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # |----------------->><<-----------------|
    # |------------Metrics Methods------------|
    # |----------------->><<-----------------|
    def compute_gradient_shift(
        self, grad_text: np.ndarray, grad_multi: np.ndarray, epsi: float = 1e-12
    ) -> Dict:
        dot = np.sum(grad_text * grad_multi, axis=1)
        denom = np.linalg.norm(grad_text, axis=1) * np.linalg.norm(grad_multi, axis=1)
        delta_cos = 1 - (dot / (denom + epsi))
        delta_l2 = np.linalg.norm(grad_multi - grad_text, axis=1)
        return {
            "delta_l2_mean": delta_l2.mean(),
            "delta_l2_std": delta_l2.std(),
            "delta_cos_mean": delta_cos.mean(),
            "delta_cos_std": delta_cos.std(),
            "delta_l2": delta_l2,
            "delta_cos": delta_cos,
        }

    def ks_test(self, grads: Dict[str, np.ndarray]):
        G_text = grads["text"]
        G_image = grads["image"]
        Gt_flat = G_text.reshape(G_text.shape[0], -1) if G_text.ndim > 2 else G_text
        Gi_flat = G_image.reshape(G_image.shape[0], -1) if G_image.ndim > 2 else G_image
        norms_text = np.linalg.norm(Gt_flat, axis=1)
        norms_image = np.linalg.norm(Gi_flat, axis=1)
        stat, p = ks_2samp(norms_text, norms_image)
        return {
            "ks_value": float(stat),
            "p_value": float(p),
            "mean_text": float(norms_text.mean()),
            "std_text": float(norms_text.std()),
            "mean_image": float(norms_image.mean()),
            "std_image": float(norms_image.std()),
        }

    def compute_sparsity(self, reps, eps: float = 1e-6):
        sparsity = {}
        z = reps.items()
        for modality, X in tqdm(
            z, total=len(reps["text"]), desc=f"Computing Sparsity: "
        ):
            sparsity[modality] = (np.abs(X) < eps).mean(axis=1)
        return sparsity

    def compute_smoothness(self, reps, grads, eps=1e-3):
        smoothness = {}
        for modality in reps:
            X = reps[modality]
            G = grads[modality]
            scores = []
            z = zip(X, G)
            for x, g in tqdm(
                z, total=len(X), desc=f"Computing Smoothness {modality}: "
            ):
                direction = g / (np.linalg.norm(g) + 1e-8)
                x_pert = x + eps * direction
                scores.append(np.linalg.norm(x_pert - x))
            smoothness[modality] = np.array(scores)
        return smoothness

    def compute_effective_rank(self, reps, eps=1e-3):
        ranks = {}
        z = reps.items()
        for modality, X in tqdm(
            z, total=len(reps["text"]), desc=f"Computing Eff. Rank: "
        ):
            cov = np.cov(X.T)
            eigvals = np.linalg.eigvalsh(cov)
            ranks[modality] = (eigvals > eps).sum()
        return ranks

    def compute_hessian_proxy(self, reps, grads, eps=1e-3):
        hessian_scores = {}
        for modality in reps:
            X = reps[modality]
            G = grads[modality]
            scores = []
            for x, g in tqdm(
                zip(X, G), total=len(X), desc=f"Computing Hessian {modality}: "
            ):
                g_norm = np.linalg.norm(g)
                if g_norm < 1e-8:
                    scores.append(0)
                    continue
                direction = g / g_norm
                x_plus = x + eps * direction
                x_minus = x - eps * direction
                curvature = np.linalg.norm(x_plus - x_minus) / (2 * eps)
                scores.append(curvature)
            hessian_scores[modality] = np.array(scores)
        return hessian_scores

    def fetch_metrics(self, reps: Dict, grads: Dict):
        reps_stats = self.get_embedding_stats(reps)
        grads_stats = self.get_gradient_stats(grads)
        smoothness = self.compute_smoothness(reps, grads)
        sparsity = self.compute_sparsity(reps, eps=1e-9)
        hessian_proxy = self.compute_hessian_proxy(reps, grads)
        modalities = list(reps.keys())
        metrics = dict(zip(modalities, [None for _ in modalities]))
        for mode in modalities:
            metrics[mode] = MMV_Metrics(
                modality=mode,
                hessian=hessian_proxy[mode].mean(0),
                smoothness=smoothness[mode].mean(0),
                sparsity=sparsity[mode].mean(),
                representation_stats=Stats(
                    per_dim_mean=reps_stats[mode]["per_dim_mean"],
                    per_dim_var=reps_stats[mode]["per_dim_var"],
                    grad_norm_mean=reps_stats[mode]["grad_mean_norm"],
                    grad_norm_std=reps_stats[mode]["grad_std_norm"],
                ),
                gradient_stats=Stats(
                    per_dim_mean=grads_stats[mode]["per_dim_mean"],
                    per_dim_var=grads_stats[mode]["per_dim_var"],
                    grad_norm_mean=grads_stats[mode]["grad_mean_norm"],
                    grad_norm_std=grads_stats[mode]["grad_std_norm"],
                ),
            )
        return metrics

    def convert_metrics_to_df(self, metrics: Dict[str, MMV_Metrics]):
        rows = []
        for modality, m in metrics.items():
            rows.append(
                {
                    "modality": modality,
                    "hessian": m.hessian,
                    "smoothness": m.smoothness,
                    "sparsity": m.sparsity,
                    "rep_per_dim_mean": m.representation_stats.per_dim_mean,
                    "rep_per_dim_var": m.representation_stats.per_dim_var,
                    "rep_grad_norm_mean": m.representation_stats.grad_norm_mean,
                    "rep_grad_norm_std": m.representation_stats.grad_norm_std,
                    "grad_per_dim_mean": m.gradient_stats.per_dim_mean,
                    "grad_per_dim_var": m.gradient_stats.per_dim_var,
                    "grad_norm_mean": m.gradient_stats.grad_norm_mean,
                    "grad_norm_std": m.gradient_stats.grad_norm_std,
                }
            )
        df = pd.DataFrame(rows)
        return df

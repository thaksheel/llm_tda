import numpy as np
from typing import Literal, List, Optional, Dict, Any
import pandas as pd 
import torch
from sklearn.metrics import average_precision_score
from peft import PeftModel
from tqdm import tqdm
from dataclasses import dataclass

from . import instance_RepT, TracInLN


@dataclass
class Result:
    top_k: int
    auPRC: float
    P_at_k: float
    R_at_k: float
    MRR: float
    method: str 

    def __repr__(self):
        return (
            f"Result(top_k={self.top_k}, "
            f"method='{self.method}', "
            f"auPRC={self.auPRC:.4f}, "
            f"P@k={self.P_at_k:.4f}, "
            f"R@k={self.R_at_k:.4f}, "
            f"MRR={self.MRR:.4f})"
        )


class TDA:
    def __init__(
        self,
        model,
        tokenizer,
        device: Literal["cuda", "cpu"],
    ):
        self.model: PeftModel = model
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.model.to(self.device)

    def _sim(self):
        return {
            "dot": lambda a, b: np.dot(a, b),
            "cosine": lambda a, b: np.dot(a, b)
            / (np.linalg.norm(a) * np.linalg.norm(b)),
        }

    def compute_attribution(
        self,
        source_data,
        eval_data,
        source_vector,
        eval_vector,
        topk,
        metric,
        method,
    ):
        topk = sorted(topk, reverse=True)
        y_scores, y_trues, precision, recall, rrs = (
            {k: [] for k in topk},
            {k: [] for k in topk},
            {k: [] for k in topk},
            {k: [] for k in topk},
            {k: [] for k in topk},
        )
        for i, vector in enumerate(eval_vector):
            sim_score = np.array(
                [self._sim()[metric](vector, vector_s) for vector_s in source_vector]
            )
            y_trues_map = np.array(
                [int(eval_data["label"][i] == label) for label in source_data["label"]]
            )
            top_max_k_indices = np.argsort(sim_score)[-topk[0] :]
            sorted_idx = np.argsort(sim_score)[::-1]
            for k in topk:
                top_k_indices = top_max_k_indices[-k:]
                precision[k].append(y_trues_map[top_k_indices].sum() / k)
                correct_total = y_trues_map.sum()
                recall[k].append(
                    y_trues_map[top_k_indices].sum() / correct_total
                    if correct_total > 0
                    else 0
                )
                rr = 0
                for rank, idx in enumerate(sorted_idx):
                    if y_trues_map[idx] == 1:
                        rr = 1.0 / (rank + 1)
                        break
                rrs[k].append(rr)
                for j in top_k_indices:
                    y_scores[k].append(sim_score[j])
                    y_trues[k].append(y_trues_map[j])
        results = [
            Result(
                top_k=k,
                auPRC=average_precision_score(y_trues[k], y_scores[k]),
                P_at_k=np.mean(precision[k]),
                R_at_k=np.mean(recall[k]),
                MRR=np.mean(rrs[k]),
                method=method,
            )
            for k in topk
        ]
        return results

    def tracing(
        self,
        source_dataset: pd.DataFrame,
        eval_dataset: pd.DataFrame,
        method: Literal["RepT", "TracInLN"],
        topk: List[int],
        layer: int = -1,
    ):
        sources, evls = [], []
        metric = "dot" if method == "TracInLN" else "cosine"
        for idx in tqdm(range(len(source_dataset["prompts"]))):
            prompt = source_dataset["prompts"].iloc[idx]
            response = source_dataset["response"].iloc[idx]
            if method == "RepT":
                gv_source = instance_RepT(
                    self.model,
                    self.tokenizer,
                    prompt,
                    response,
                    layer,
                    self.device,
                )
            elif method == "TracInLN":
                gv_source = TracInLN(
                    self.model,
                    self.tokenizer,
                    prompt,
                    response,
                )
            sources.append(gv_source)
        for idx in tqdm(range(len(eval_dataset["prompts"]))):
            prompt = eval_dataset["prompts"].iloc[idx]
            expected_response = eval_dataset["expected_response"].iloc[idx]
            if method == "RepT":
                evl = instance_RepT(
                    self.model,
                    self.tokenizer,
                    prompt,
                    expected_response,
                    layer,
                    self.device,
                )
            elif method == "TracInLN":
                evl = TracInLN(self.model, self.tokenizer, prompt, expected_response)
            evls.append(evl)
        results = self.compute_attribution(
            source_dataset, eval_dataset, sources, evls, topk, metric, method
        )
        return results

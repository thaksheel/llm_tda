import numpy as np
from typing import Literal, List, Optional, Dict, Any
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from peft import PeftModel
from tqdm import tqdm

from . import Result, TDA, Params


class TDAEvaluation:
    def __init__(
        self,
        model,
        tokenizer,
        device: Literal["cuda", "cpu"],
        params: Params,
        display: bool = False,
    ):
        self.model: PeftModel = model
        self.device = torch.device(device)
        self.params = params
        self.tokenizer = tokenizer
        self.model.to(self.device)
        self.tda = TDA(
            self.model,
            tokenizer,
            params,
            self.device,
            display,
        )

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
                [
                    int(eval_data["label"].tolist()[i] == label)
                    for label in source_data["label"]
                ]
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
        method: Literal["rept", "tracein", "rapid_in", "less", "lorif"],
        topk: List[int],
    ):
        source_signals, testing_signals = [], []
        metric = "dot" if method == "TracInLN" else "cosine"
        for i, row in tqdm(source_dataset.iterrows(), total=source_dataset.shape[0]):
            source_signals.append(
                self.tda.trace(
                    method=method,
                    prompt=row["prompts"],
                    expected_response=row["response"],
                )
            )
        for i, row in tqdm(eval_dataset.iterrows(), total=eval_dataset.shape[0]):
            testing_signals.append(
                self.tda.trace(
                    method=method,
                    prompt=row["prompts"],
                    expected_response=row["expected_response"],
                )
            )
        results = self.compute_attribution(
            source_dataset,
            eval_dataset,
            source_signals,
            testing_signals,
            topk,
            metric,
            method,
        )
        return results

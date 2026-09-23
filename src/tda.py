import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional, Literal
from datasets import Dataset
from transformers import TokenizersBackend
from transformers import TokenizersBackend
from peft import PeftModel
from numpy.typing import NDArray

from . import Params, BasicProjector, ProjectionType


class TDA:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: TokenizersBackend,
        params: Params,
        device=Literal["cpu", "cuda"],
        display: bool = False,
    ):
        self.model = model
        self.params = params
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.display = display

    def trace(
        self,
        method: Literal["rept", "tracein", "rapid_in", "less", "lorif"],
        prompt: str,
        expected_response: str,
    ) -> NDArray:
        if method == "rept":
            return self.rept(prompt, expected_response, layer=self.params.layer)
        elif method == "tracein":
            return self.trace_in(
                prompt, expected_response, layer_norm=self.params.layer_norm
            )
        elif method == "lorif":
            raise NotImplementedError
        elif method == "less":
            return self.less(
                prompt, expected_response, optimizer_state=self.params.optimizer_state
            )
        elif method == "rapid_in":
            return self.rapid_in(prompt, expected_response)
        else:
            raise NotImplementedError

    def _get_divisors(self, n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
        return divs

    def random_suffle(self, vector: torch.Tensor, num_shuffles=20):
        vec_len = vector.shape[0] * vector.shape[1]
        shuffled_v = vector.clone()
        divs = self._get_divisors(vec_len)
        # random shuffle
        for _ in range(num_shuffles):
            x_row = random.choice(divs)
            mat = shuffled_v.reshape(x_row, vec_len // x_row)
            row_indices = torch.randperm(mat.shape[0], device=vector.device)
            shuffled_v = mat[row_indices, :]
            x_col = random.choice(divs)
            mat = shuffled_v.reshape(vec_len // x_col, x_col)
            col_indices = torch.randperm(mat.shape[1], device=vector.device)
            shuffled_v = mat[:, col_indices]
        return shuffled_v.flatten()

    def rapid_in(self, prompt: str, expected_response: str):
        self.model.eval()
        self.model.zero_grad()
        inputs = self.get_tokenized_text(
            self.tokenizer,
            {"prompts": prompt, "response": expected_response},
            device=self.model.device,
        )
        outputs = self.model(**inputs)
        loss: torch.Tensor = outputs["loss"]
        loss.backward()
        gradient_vector = torch.cat(
            [
                p.grad.view(-1)
                for n, p in self.model.named_parameters()
                if p.grad is not None
            ]
        )
        gradient_vector = gradient_vector.reshape(
            self.model.config.num_hidden_layers, -1
        )
        gradient_vector = F.layer_norm(
            gradient_vector, normalized_shape=[gradient_vector.shape[-1]]
        )
        gradient_vector = self.random_suffle(gradient_vector)
        gradient_vector = self.random_projection(
            gradient_vector, proj_dim=2**16, block_size=1024
        )
        return gradient_vector.cpu().numpy()

    def prepare_optimizer_state(self, optimizer_state):
        names = [_ for _ in range(len(optimizer_state))]
        avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
        avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in names])
        avg = avg.to(self.device)
        avg_sq = avg_sq.to(self.device)
        return avg, avg_sq

    def random_projection(
        self, vector: torch.Tensor, proj_dim=8192, block_size=128, model_id=0
    ):
        projector = BasicProjector(
            grad_dim=vector.shape[0],
            proj_dim=proj_dim,
            seed=42,
            proj_type=ProjectionType.rademacher,
            device=vector.device,
            block_size=block_size,
        )
        projected_v = projector.project(vector.reshape(1, -1), model_id=model_id)
        return projected_v.view(-1)

    def less(self, prompt, expected_response, optimizer_state=None):
        self.model.eval()
        self.model.zero_grad()
        inputs = self.get_tokenized_text(
            self.tokenizer,
            {"prompts": prompt, "response": expected_response},
            device=self.device,
        )
        outputs = self.model(**inputs)
        loss: torch.Tensor = outputs["loss"]
        loss.backward()
        beta1, beta2, eps = 0.9, 0.999, 1e-08
        gradient_vector = torch.cat(
            [
                p.grad.view(-1)
                for n, p in self.model.named_parameters()
                if p.grad is not None
            ]
        )
        if optimizer_state is not None:
            avg, avg_sq = self.prepare_optimizer_state(optimizer_state)
            updated_avg = beta1 * avg + (1 - beta1) * gradient_vector
            updated_avg_sq = beta2 * avg_sq + (1 - beta2) * gradient_vector**2
            gradient_vector = updated_avg / torch.sqrt(updated_avg_sq + eps)
        gradient_vector = self.random_projection(gradient_vector)
        return gradient_vector.cpu().numpy()

    def get_gradient_vector(self, prompt, expected_response) -> torch.Tensor:
        self.model.train()
        self.model.zero_grad()
        inputs = self.get_tokenized_text(
            self.tokenizer,
            {"prompts": prompt, "response": expected_response},
            device=self.device,
        )
        outputs = self.model.forward(**inputs)
        loss: torch.Tensor = outputs["loss"]
        loss.backward()
        gv = torch.cat(
            [
                p.grad.view(-1)
                for n, p in self.model.named_parameters()
                if p.grad is not None
            ]
        )
        return gv.detach()

    def trace_in(self, prompt: str, expected_response: str, layer_norm: bool = True):
        gv = self.get_gradient_vector(prompt, expected_response)
        if layer_norm:
            gv = F.layer_norm(gv, normalized_shape=[gv.shape[-1]])
        gv = gv.view(-1).to(torch.float32)
        return gv.cpu().numpy()

    def get_representation(
        self,
        prompt,
        layer,
    ) -> NDArray:
        self.model.eval()
        if not (1 <= layer <= self.model.config.num_hidden_layers or layer == -1):
            raise ValueError(
                f"Layer index must be between 1 and {self.model.config.num_hidden_layers}. Got {layer}."
            )
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "[INST] " + prompt + " [/INST]"
        inputs = self.tokenizer(prompt, padding=True, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden: torch.Tensor = outputs["hidden_states"][layer][:, -1, :].to(
            torch.float32
        )
        return hidden.view(-1).cpu().numpy()

    def get_representation_gradient(
        self,
        prompt,
        expected_response,
        layer,
    ) -> NDArray:
        self.model.train()
        self.model.zero_grad()
        self.model.config.use_cache = False
        # model.set_requires_grad(requires_grad=True)
        captured_grads = []
        if not (1 <= layer <= self.model.config.num_hidden_layers or layer == -1):
            raise ValueError(
                f"Layer index must be between 1 and {self.model.config.num_hidden_layers}. Got {layer}."
            )
        inputs = self.get_tokenized_text(
            self.tokenizer,
            {"prompts": prompt, "response": expected_response},
            device=self.device,
        )
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        prompt_len = (inputs["labels"].cpu().numpy() == -100).sum()
        last_hidden_state: torch.Tensor = outputs["hidden_states"][layer]
        last_hidden_state.register_hook(lambda grad: captured_grads.append(grad))
        loss = outputs["loss"]
        loss.backward()
        self.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        # [0, 0, ..., 0, 0, r_response_1, ..., r_response_n, 0]
        cg: torch.Tensor = captured_grads[0][0][prompt_len - 1 : -1].to(torch.float32)
        return cg.view(-1).cpu().numpy()

    def rept(
        self,
        prompt: str,
        response: str,
        layer: int,
    ):
        H = self.get_representation(prompt, layer)
        g_H = self.get_representation_gradient(prompt, response, layer)[0]
        return np.hstack((H, g_H))

    def tokenize(self, tokenizer: TokenizersBackend, sample: Dict, max_length=512):
        if tokenizer.chat_template:
            message = [
                {"role": "user", "content": sample["prompts"]},
                {"role": "assistant", "content": sample["response"]},
            ]
            full_text = tokenizer.apply_chat_template(message, tokenize=False)
            full_tokenized = tokenizer(
                full_text, truncation=True, max_length=512, padding=False
            )
            prompt_only_text = tokenizer.apply_chat_template(
                [message[0]], tokenize=False, add_generation_prompt=True
            )
            prompt_length = len(
                tokenizer(
                    prompt_only_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"]
            )
        else:
            full_text = (
                "[INST] "
                + sample["prompts"]
                + " [/INST]"
                + sample["response"]
                + tokenizer.eos_token
            )
            full_tokenized = tokenizer(
                full_text, truncation=True, max_length=512, padding=False
            )
            prompt_length = len(
                tokenizer(
                    "[INST] " + sample["prompts"] + " [/INST]",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"]
            )

        input_ids = full_tokenized["input_ids"]
        labels = list(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full_tokenized["attention_mask"],
        }

    def get_tokenized_dataset(
        self,
        tokenizer: TokenizersBackend,
        dataset: Dataset,
        max_length=512,
    ):
        return dataset.map(
            self.tokenize,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": max_length,
            },
            remove_columns=list(dataset.features),
        )

    def get_tokenized_text(
        self, tokenizer: TokenizersBackend, sample, device, max_length=512
    ):
        if tokenizer.chat_template:
            message = [
                {"role": "user", "content": sample["prompts"]},
                {"role": "assistant", "content": sample["response"]},
            ]
            full_text = tokenizer.apply_chat_template(message, tokenize=False)
            full_tokenized = tokenizer(
                full_text, truncation=True, max_length=max_length, padding=False
            )
            prompt_only_text = tokenizer.apply_chat_template(
                [message[0]], tokenize=False, add_generation_prompt=True
            )
            prompt_length = len(
                tokenizer(
                    prompt_only_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"]
            )
        else:
            full_text = (
                "[INST] "
                + sample["prompts"]
                + " [/INST]"
                + sample["response"]
                + tokenizer.eos_token
            )
            full_tokenized = tokenizer(
                full_text, truncation=True, max_length=max_length, padding=False
            )
            prompt_length = len(
                tokenizer(
                    "[INST] " + sample["prompts"] + " [/INST]",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"]
            )
        input_ids = full_tokenized["input_ids"]
        labels = list(input_ids)
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": torch.tensor([input_ids]).to(device),
            "labels": torch.tensor([labels]).to(device),
            "attention_mask": torch.tensor([full_tokenized["attention_mask"]]).to(
                device
            ),
        }

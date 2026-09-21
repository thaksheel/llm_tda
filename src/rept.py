import torch
import numpy as np
from datasets import Dataset
from transformers import TokenizersBackend
from peft import PeftModel

from .utils import get_tokenized_text


def get_representation(
    model: PeftModel,
    tokenizer: TokenizersBackend,
    prompt,
    layer,
    device,
):
    model.eval()
    if not (1 <= layer <= model.config.num_hidden_layers or layer == -1):
        raise ValueError(
            f"Layer index must be between 1 and {model.config.num_hidden_layers}. Got {layer}."
        )
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "[INST] " + prompt + " [/INST]"
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs["hidden_states"][layer][:, -1, :].to(torch.float32)
    return hidden.view(-1).cpu().numpy()


def get_representation_gradient(
    model: PeftModel,
    tokenizer: TokenizersBackend,
    prompt,
    expected_response,
    layer,
    device,
):
    model.train()
    model.zero_grad()
    model.config.use_cache = False
    # model.set_requires_grad(requires_grad=True)
    captured_grads = []
    if not (1 <= layer <= model.config.num_hidden_layers or layer == -1):
        raise ValueError(
            f"Layer index must be between 1 and {model.config.num_hidden_layers}. Got {layer}."
        )
    inputs = get_tokenized_text(
        tokenizer,
        {"prompts": prompt, "response": expected_response},
        device=device,
    )
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    prompt_len = (inputs["labels"].cpu().numpy() == -100).sum()
    last_hidden_state: torch.Tensor = outputs["hidden_states"][layer]
    last_hidden_state.register_hook(lambda grad: captured_grads.append(grad))
    loss = outputs["loss"]
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    # [0, 0, ..., 0, 0, r_response_1, ..., r_response_n, 0]
    cg = captured_grads[0][0][prompt_len - 1 : -1].to(torch.float32)
    return cg.view(-1).cpu().numpy()


def instance_RepT(model, tokenizer, prompt, response, layer, device):
    H = get_representation(model, tokenizer, prompt, layer, device)
    g_H = get_representation_gradient(
        model, tokenizer, prompt, response, layer, device
    )[0]
    return np.hstack((H, g_H))

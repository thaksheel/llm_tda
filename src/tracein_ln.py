import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import TokenizersBackend
from .utils import get_tokenized_text


def get_gradient_vector(
    model: PeftModel, tokenizer: TokenizersBackend, prompt, expected_response
) -> torch.Tensor:
    model.train()
    model.zero_grad()
    inputs = get_tokenized_text(
        tokenizer,
        {"prompts": prompt, "response": expected_response},
        device=model.device,
    )
    outputs = model(**inputs)
    loss = outputs["loss"]
    loss.backward()
    num_layers = model.config.num_hidden_layers
    layer_grads = [[] for _ in range(num_layers)]
    # NOTE: works for llama/mistral style naming
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        for i in range(num_layers):
            if f"layers.{i}" in name:
                layer_grads[i].append(p.grad.view(-1))
                break
    for i in range(num_layers):
        if len(layer_grads[i]) == 0:
            layer_grads[i] = [torch.zeros(1, device=model.device)]
    layer_grads = [torch.cat(g) for g in layer_grads]
    gradient_vector = torch.cat(layer_grads)
    return gradient_vector


def TracInLN(model, tokenizer, prompt, expected_response):
    gradient_vector = get_gradient_vector(model, tokenizer, prompt, expected_response)
    gradient_vector = F.layer_norm(
        gradient_vector, normalized_shape=[gradient_vector.shape[-1]]
    )
    gv = gradient_vector.view(-1).to(torch.float32)
    return gv.cpu().numpy()

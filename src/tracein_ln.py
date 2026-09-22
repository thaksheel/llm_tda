import torch
import torch.nn.functional as F
from peft import PeftModel
import re 
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
    outputs = model.forward(**inputs)
    loss = outputs["loss"]
    loss.backward()
    gv = torch.cat([
        p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None
    ])
    return gv.detach()


def TracInLN(model, tokenizer, prompt, expected_response):
    gv = get_gradient_vector(model, tokenizer, prompt, expected_response)
    gv = F.layer_norm(
        gv, normalized_shape=[gv.shape[-1]]
    )
    gv = gv.view(-1).to(torch.float32)
    return gv.cpu().numpy()

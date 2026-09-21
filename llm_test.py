import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_from_disk

from src import TDA

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "Qwen/Qwen3-VL-2B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="cuda",
    # attn_implementation="flash_attention_2", # better acc and memory saving, not supported on windows 
)
processor = AutoProcessor.from_pretrained(model_name)


# -----------------------------
# Image-Text-to-text generation example
# -----------------------------
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(f"\n\n> output_text: {output_text[0]}") 


# -----------------------------
# Text-only generation example
# -----------------------------
text_messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Why is the sky blue"}],
    }
]

# Prepare inputs for text-only chat
text_inputs = processor.apply_chat_template(
    text_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
text_inputs = text_inputs.to(model.device)

# Generate
generated_ids = model.generate(
    **text_inputs,
    max_new_tokens=128,
)

# Trim input tokens from output
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(text_inputs.input_ids, generated_ids)
]

# Decode
text_output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n\n > text-only answer:")
print(text_output[0])

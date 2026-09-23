import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk

from src import TDAEvaluation, Params

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
lora_path = "./lora_adapter/TinyLlama/TinyLlama-1.1B-Chat-v1.0/datasets/harmful-tuning"
model = PeftModel.from_pretrained(model, lora_path)
optimizer_state = torch.load(lora_path + "/optimizer.pt", map_location="cpu")["state"]
# Unfreeze base model parameters for RepT
for p in model.base_model.parameters():
    p.requires_grad = True
# Also unfreeze LoRA parameters (they already are)
for p in model.parameters():
    p.requires_grad = True

source_dataset = load_from_disk("./data/datasets/harmful-tuning")
df_source = source_dataset.to_pandas()
evl_dataset = pd.read_csv("./data/datasets/harmful-tuning_test.csv")

tda = TDAEvaluation(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    params=Params(layer=-1, layer_norm=True, optimizer_state=optimizer_state),
)
results = tda.tracing(
    source_dataset=df_source,
    eval_dataset=evl_dataset,
    method="rept",
    topk=[1, 5, 10, 30, 50, 100, 250, 500, 1000],
)
df_results = pd.DataFrame([r.__dict__ for r in results])
df_results.to_excel("./exports/tda_results4.xlsx")

print(f"first 10 RepT results: \n\n{results[:10]}")
print("END")

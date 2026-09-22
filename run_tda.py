import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_from_disk

from src import TDA

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(
    model,
    "./lora_adapter/TinyLlama/TinyLlama-1.1B-Chat-v1.0/datasets/harmful-tuning",
)
# Unfreeze base model parameters for RepT
for p in model.base_model.parameters():
    p.requires_grad = True
# Also unfreeze LoRA parameters (they already are)
for p in model.parameters():
    p.requires_grad = True

source_dataset = load_from_disk("./data/datasets/harmful-tuning")
df_source = source_dataset.to_pandas()
evl_dataset = pd.read_csv("./data/datasets/harmful-tuning_test.csv")

tda = TDA(model=model, tokenizer=tokenizer, device="cuda")
results_tracein = tda.tracing(
    source_dataset=df_source.sample(n=100),
    eval_dataset=evl_dataset.sample(n=50),
    method="TracInLN",
    topk=[1, 5, 10, 30, 50, 100, 250, 500, 1000],
    layer=-1,
)
df_results = pd.DataFrame([r.__dict__ for r in results_tracein])
df_results.to_excel("./exports/tda_results4.xlsx")

results_rept = tda.tracing(
    source_dataset=df_source,
    eval_dataset=evl_dataset,
    method="RepT",
    topk=[1, 5, 10, 30, 50, 100, 250, 500, 1000],
    layer=-1,
)
df_ = pd.DataFrame([r.__dict__ for r in results_rept])
df_results = pd.concat([df_results, df_]) 
df_results['llm_model'] = [model_name] * len(df_results)
df_results.to_excel("./exports/tda_results4.xlsx")

print(f"first 10 RepT results: \n\n{results_rept[:10]}")
print("END")

import os
from transformers import TrainingArguments
from peft import LoraConfig
from huggingface_hub import login

from src import Tuner

login(token=os.getenv("HUGGING_FACE_TOKEN"))


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
    ],
)
training_args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    save_steps=200,
    fp16=True,
    gradient_checkpointing=True,
    num_train_epochs=3,
    logging_steps=50,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=1,
)

modelname = "meta-llama/Llama-2-70b-hf"
modelname = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
modelname = "Qwen/Qwen2.5-7B-Instruct"
modelname = "meta-llama/Llama-2-7b-hf"
modelname = "meta-llama/Llama-3.1-8B-Instruct"

tuner = Tuner(
    dataset_name="./data/datasets/harmful-tuning",
    model_name=modelname,
    training_args=training_args,
    lora_config=lora_config,
    load_in_4bit=True,
)
trainer = tuner.train(save_model=True)
model = tuner.model
tokenizer = tuner.tokenizer
dataset = tuner.dataset

print("END")

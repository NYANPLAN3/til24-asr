import os

from peft import LoraConfig
from transformers import Seq2SeqTrainingArguments

HOME = os.environ.get("HOME", "/root")
DATASET_FOLDER = f'{HOME}/advanced/audio'
JSONL_FILE_PATH = f'{HOME}/advanced/asr.jsonl'
MODEL_ARCH = "openai/whisper-large-v3"
DATASET_PATH = "/workspaces/til24-main/til24-asr/data/til24asr"

NPROC = os.cpu_count()

LORA_CFG = LoraConfig(r=32, lora_alpha=64, target_modules=[
    "q_proj", "v_proj"], lora_dropout=0.05, bias="none")
TRAIN_CFG = Seq2SeqTrainingArguments(
    output_dir="./runs/train",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    dataloader_num_workers=8,
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="steps",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=128,
    # max_steps=100, # only for testing purposes, remove this from your final run :)
    # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    remove_unused_columns=False,
    label_names=["labels"],  # same reason as above
)

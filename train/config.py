import os

from peft import LoraConfig
from transformers import Seq2SeqTrainingArguments

HOME = os.environ.get("HOME", "/root")
DATASET_FOLDER = f'{HOME}/advanced/audio'
JSONL_FILE_PATH = f'{HOME}/advanced/asr.jsonl'
MODEL_ARCH = "openai/whisper-large-v3"
DATASET_PATH = "/workspaces/til24-main/til24-asr/data/til24asr"

NPROC = os.cpu_count()

# https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig
LORA_CFG = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
    init_lora_weights="pissa_niter_16",
)
# https://huggingface.co/docs/transformers/v4.41.2/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments
TRAIN_CFG = Seq2SeqTrainingArguments(
    output_dir="./runs/train",
    eval_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    weight_decay=1e-4,
    num_train_epochs=64,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=16,
    save_strategy="steps",
    save_total_limit=1,
    fp16=True,
    fp16_full_eval=True,
    eval_steps=128,
    save_steps=128,
    report_to=["tensorboard"],
    # metric_for_best_model="wer",
    # greater_is_better=False,
    dataloader_num_workers=8,
    dataloader_persistent_workers=False,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    remove_unused_columns=False,
    label_names=["labels"],  # same reason as above
)

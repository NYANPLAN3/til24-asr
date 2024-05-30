import os
from pathlib import Path
from typing import Dict, List, Union

import torch
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer

from .config import *


def preprocess_dataset(ds: DatasetDict, model_arch=MODEL_ARCH):
    extractor = WhisperFeatureExtractor.from_pretrained(model_arch)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_arch, language="English", task="transcribe")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    ds = ds.map(prepare_dataset, num_proc=NPROC)
    return ds


def load_from_jsonl(ds_dir: str, ds_lbl_path: str):
    ds_dir = Path(ds_dir)
    data = load_dataset('json', data_files=ds_lbl_path)
    data = data.map(lambda x: {"audio": str(
        (ds_dir/x['audio']).resolve())}, num_proc=NPROC)
    data = data.rename_column("transcript", "sentence")
    data = data.cast_column("audio", Audio(sampling_rate=16000))
    data = data["train"]

    ds = DatasetDict()
    ds['test'] = data.take(500)
    ds['train'] = data.skip(500)
    return ds


def load_or_create_til_dataset():
    if os.path.exists(DATASET_PATH):
        ds = load_from_disk(DATASET_PATH)
    else:
        ds = load_from_jsonl(DATASET_FOLDER, JSONL_FILE_PATH)
        ds = preprocess_dataset(ds)
        ds.save_to_disk(DATASET_PATH)

    return ds


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, model_arch: str = MODEL_ARCH):
        self.processor = WhisperProcessor.from_pretrained(
            model_arch, language="English", task="transcribe")

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

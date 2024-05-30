from transformers import Seq2SeqTrainer

from .callbacks import SavePeftModelCallback
from .config import *
from .data import DataCollatorSpeechSeq2SeqWithPadding, load_or_create_til_dataset
from .model import prepare_model


def main():
    ds = load_or_create_til_dataset()
    collator = DataCollatorSpeechSeq2SeqWithPadding()
    model = prepare_model()
    trainer = Seq2SeqTrainer(
        args=TRAIN_CFG,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=collator,
        tokenizer=collator.processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    trainer.train()


if __name__ == "__main__":
    main()

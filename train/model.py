from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import WhisperForConditionalGeneration

from .config import *


def prepare_model():
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ARCH)
    model = prepare_model_for_kbit_training(model)

    def _make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(_make_inputs_require_grad)
    model = get_peft_model(model, LORA_CFG)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model

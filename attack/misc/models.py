"""Model management and initialization for the MIA framework."""

import logging
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoConfig, AutoModel
from peft import PeftModel


class ModelManager:
    """Manages model initialization and configuration."""

    @staticmethod
    def register_custom_models():
        """Register custom models with HuggingFace AutoClasses."""
        from trainer.model.llada.configuration_llada import LLaDAConfig
        from trainer.model.llada.modeling_llada import LLaDAModelLM
        from trainer.model.dream.configuration_dream import DreamConfig
        from trainer.model.dream.modeling_dream import DreamModel

        AutoConfig.register("llada", LLaDAConfig)
        AutoModel.register(LLaDAConfig, LLaDAModelLM)
        AutoConfig.register("Dream", DreamConfig)
        AutoModel.register(DreamConfig, DreamModel)
        logging.info("Custom models registered successfully")

    @staticmethod
    def init_model(model_name: str, tokenizer_name: str, device: torch.device,
                   lora_adapter_path: Optional[str] = None) -> Tuple:
        """Initialize the model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, load_in_4bit=True)
        logging.info(f"Base model {model_name} loaded. Type: {type(model).__name__}")

        if lora_adapter_path:
            logging.info(f"Loading LoRa adapter from: {lora_adapter_path}")
            model = model.to(device)
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            logging.info("LoRa adapter applied successfully")

        model = model.to(device)

        if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
            logging.info(f"Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        return model, tokenizer, device

    @staticmethod
    def get_model_nll_params(model) -> Tuple[bool, int]:
        """Determine model-specific NLL parameters."""
        unwrapped_model = model

        if isinstance(unwrapped_model, torch.nn.DataParallel):
            unwrapped_model = unwrapped_model.module
        if PeftModel is not None and isinstance(unwrapped_model, PeftModel):
            unwrapped_model = unwrapped_model.base_model.model

        determined_shift_logits = False
        determined_mask_id = 126336

        if hasattr(unwrapped_model, 'config') and hasattr(unwrapped_model.config, 'model_type'):
            model_type = unwrapped_model.config.model_type
            logging.info(f"Detected model_type: '{model_type}' from loaded model's config")

            if model_type == "Dream":
                determined_shift_logits = True
                determined_mask_id = 151666
        else:
            logging.warning("Could not determine model_type from model config. Using defaults.")

        logging.info(f"NLL params: shift_logits={determined_shift_logits}, mask_id={determined_mask_id}")
        return determined_shift_logits, determined_mask_id
import logging
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import (
    Trainer,
    PreTrainedModel,
)

# Import PEFT libraries for LoRA
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel # For type checking
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel = None, None, None, None, None

logger = logging.getLogger(__name__)


def forward_process(input_ids: torch.Tensor, eps: float = 1e-3, mask_id: int = 126336):
    """
    Randomly masks tokens in 'input_ids' with probability p_mask (sampled from U(eps,1)).
    p_mask is drawn independently for each sample in the batch.

    Returns:
      noisy_batch: [B, L] with masked tokens replaced by the special ID
      masked_indices: bool [B, L] where True means "that token was masked."
      p_mask: float [B, L], the mask probability for each token in that sample.
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1.0 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    return noisy_batch, masked_indices, p_mask


class DiffusionLLMTrainer(Trainer):
    """
    Custom Trainer for Diffusion-based LLMs. Overrides `compute_loss` to implement
    random masking and conditional logit shifting based on model type.
    """

    def __init__(
        self,
        train_mode: str = "pretraining",
        mask_id: int = 126336,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.train_mode = train_mode.lower().strip()
        self.mask_id = mask_id
        if self.train_mode not in ["pretraining", "sft"]:
            raise ValueError(f"train_mode must be 'pretraining' or 'sft', got {self.train_mode}")

        # Determine if logits should be shifted based on model type
        self.shift_logits = False
        model_to_check = self.model

        if PEFT_AVAILABLE and isinstance(self.model, PeftModel):
            model_to_check = self.model.base_model.model

        if hasattr(model_to_check, 'config') and hasattr(model_to_check.config, 'model_type'):
            actual_model_type_str = model_to_check.config.model_type
            logger.info(f"Detected model_type: '{actual_model_type_str}' from model config.")
            if actual_model_type_str == "Dream":
                self.shift_logits = True
        else:
            logger.warning(
                "Could not determine model_type from model config. Logit shifting decision will default to False. "
                f"Model config type: {type(model_to_check.config).__name__ if hasattr(model_to_check, 'config') else 'N/A'}"
            )

        logger.info(
            f"DiffusionLLMTrainer initialized. Logit shifting for loss computation is set to: {self.shift_logits} "
            f"based on model type: {type(model_to_check).__name__}"
        )

    def compute_loss(self,
                     model: PreTrainedModel,
                     inputs: Dict[str, torch.Tensor],
                     return_outputs: bool = False,
                     **kwargs
                     ):
        """
        1) If pretraining => mask all tokens with probability p_mask.
        2) If sft => do the same, except we do not mask the prompt portion.
        3) Applies logit shifting if self.shift_logits is True.
        """
        input_ids = inputs["input_ids"]

        if self.train_mode == "pretraining" and torch.rand(1).item() < 0.01:
            new_length = torch.randint(low=1, high=input_ids.shape[1] + 1, size=(1,)).item()
            input_ids = input_ids[:, :new_length]

        noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_id=self.mask_id)

        if self.train_mode == "sft":
            if "prompt_lengths" not in inputs:
                raise ValueError("SFT mode requires 'prompt_lengths' in the batch.")
            prompt_lengths = inputs["prompt_lengths"]
            batch_size, seq_len = input_ids.shape  # Use input_ids shape for consistency

            positions = torch.arange(seq_len, device=noisy_batch.device).unsqueeze(0)
            prompt_mask_sft = positions < prompt_lengths.unsqueeze(1)

            noisy_batch[prompt_mask_sft] = input_ids[prompt_mask_sft]
            masked_indices[prompt_mask_sft] = False

        # Use the model passed as argument, which is standard for Trainer's compute_loss
        outputs = model(input_ids=noisy_batch)
        raw_logits = outputs.logits  # [B, L, vocab]

        # --- Apply conditional logit shifting ---
        if self.shift_logits:
            # Assumes raw_logits is [B, L, V]. Shifts along L dimension.
            # Logits for token t become prediction for token t (from t-1 input in causal models)
            logits = torch.cat([raw_logits[:, :1, :], raw_logits[:, :-1, :]], dim=1)
        else:
            logits = raw_logits
        # --- End logit shifting ---

        bsz, seqlen, vocab_size = logits.shape
        logits_2d = logits.view(-1, vocab_size)
        labels_2d = input_ids.contiguous().view(-1)
        masked_indices_1d = masked_indices.view(-1)
        p_mask_1d = p_mask.view(-1)

        masked_logits = logits_2d[masked_indices_1d, :]
        masked_labels = labels_2d[masked_indices_1d]
        masked_pmask = p_mask_1d[masked_indices_1d]

        if masked_labels.numel() == 0:
            loss = masked_logits.sum() * 0.0
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=logits.device,
                                    requires_grad=model.training)
            elif model.training:
                loss.requires_grad_(True)

            return (loss, outputs) if return_outputs else loss

        ce_per_token = F.cross_entropy(masked_logits, masked_labels, reduction='none')
        scaled_ce = ce_per_token / masked_pmask

        if self.train_mode == "pretraining":
            total_tokens_in_batch = float(bsz * seqlen)  # Or input_ids.numel()
            loss = scaled_ce.sum() / total_tokens_in_batch
        else:  # SFT mode
            # prompt_mask_sft was calculated earlier for SFT mode
            answer_mask = ~prompt_mask_sft  # Boolean mask for answer tokens
            answer_lengths = answer_mask.sum(dim=1).clamp(
                min=1)  # Number of answer tokens per sample, min 1 to avoid div by zero

            row_id = torch.arange(bsz, device=masked_indices.device).unsqueeze(1).repeat(1, seqlen).view(-1)
            row_id_masked = row_id[masked_indices_1d]

            ans_len_for_masked = answer_lengths[row_id_masked]  # Already clamped earlier

            scaled_ce = scaled_ce / ans_len_for_masked.float()
            loss = scaled_ce.sum() / float(bsz)

        return (loss, outputs) if return_outputs else loss

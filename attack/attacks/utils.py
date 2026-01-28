import random
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer


def _forward_process_llada(
    seq: torch.Tensor,
    mask_id: int,
    prefix_mask: torch.Tensor,
    eps: float = 1e-3,
) -> (torch.Tensor, torch.Tensor):
    """
    Matches the official code's approach more closely:
      - Sample fraction p in [eps, 1 - eps].
      - For each non-prefix token, independently mask it with probability p (i.e., Bernoulli(p)).
      - For each masked token, p_mask = p (so we later divide cross_entropy by p).
    """
    device = seq.device
    length = seq.size(0)

    # Identify which positions we *can* mask (i.e., not prefix/pad).
    maskable_positions = (~prefix_mask).nonzero(as_tuple=True)[0]
    maskable_count = maskable_positions.numel()
    if maskable_count < 1:
        # Nothing to mask => return seq unchanged; p_mask = 1 everywhere
        return seq.clone(), torch.ones_like(seq, dtype=torch.float, device=device)

    # 1) Draw a fraction p in [eps, 1 - eps].
    p = (1.0 - eps) * torch.rand((), device=device) + eps

    # 2) For each maskable position, do a Bernoulli draw to decide if it becomes masked.
    bernoulli_draws = torch.rand(maskable_count, device=device)
    final_mask_bool = (bernoulli_draws < p)  # True => will be masked

    # 3) Construct perturbed_seq
    perturbed_seq = seq.clone()
    perturbed_seq[maskable_positions[final_mask_bool]] = mask_id

    # 4) Build p_mask. For masked tokens => p_mask[i] = p, else 1.
    p_mask = torch.ones_like(seq, dtype=torch.float, device=device)
    p_mask[maskable_positions[final_mask_bool]] = p

    return perturbed_seq, p_mask


@torch.no_grad()
def compute_nlloss(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    shift_logits: bool,
    ignore_prefix: Optional[int] = None,
    mc_num: int = 3,
    mask_id: int = 126336,
) -> np.ndarray:
    """
    Computes a denoising-style negative log-likelihood for diffusion-based LLMs.

    Arguments:
      model: A diffusion-based LLM that accepts input_ids and returns logits.
      token_ids: [batch_size, seq_len] input token IDs.
      attention_mask: [batch_size, seq_len] standard HF attention mask.
      shift_logits: bool, if True, logits are shifted before loss calculation.
      ignore_prefix: If set, skip the first N tokens from masking and loss.
      mc_num: How many random partial masks (“Monte Carlo passes”) per example.
      mask_id: The special token ID to replace masked tokens with.
    Returns:
      A float array of shape [batch_size], the negative log-likelihood.
    """
    model.eval()
    device = token_ids.device
    nll_array = np.zeros(token_ids.shape[0], dtype=np.float64)

    for i in range(token_ids.size(0)):  # This loop handles items in the batch
        seq_i = token_ids[i]  # Current sequence (1D tensor)
        attn_i = attention_mask[i]  # Current attention mask (1D tensor)
        seq_len = seq_i.size(0)
        current_seq_actual_len = attn_i.sum().item()

        # Determine prefix_mask (includes padding and explicit prefix)
        prefix_count_val = ignore_prefix if ignore_prefix is not None else 0
        effective_prefix_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if prefix_count_val > 0:
            # Ensure prefix_count doesn't exceed actual sequence length for safety
            actual_prefix_to_ignore = min(prefix_count_val, current_seq_actual_len)
            if actual_prefix_to_ignore > 0:
                effective_prefix_mask[:actual_prefix_to_ignore] = True

        padded_positions = (attn_i == 0)
        effective_prefix_mask = effective_prefix_mask | padded_positions

        accumulated_loss = 0.0
        actual_passes_with_masking = 0

        for _ in range(mc_num):
            # Pass the specific mask_id for this model
            noisy_seq, p_mask_values = _forward_process_llada(seq_i, mask_id, effective_prefix_mask)

            # Model expects a batch, so unsqueeze
            # outputs.logits will be (1, seq_len_for_noisy_seq, vocab_size)
            outputs = model(noisy_seq.unsqueeze(0))

            # --- Conditional Logit Shifting ---
            logits_batch = outputs.logits  # Shape (1, L, V)
            if shift_logits:
                # Shifts logits: new_logits[t] = old_logits[t-1] (for t>0)
                # This aligns predictions if model predicts token t+1 from input t
                logits_batch = torch.cat([logits_batch[:, :1, :], logits_batch[:, :-1, :]], dim=1)

            # Squeeze out the batch dimension of 1 to get (L, V)
            logits_for_loss_calc = logits_batch.squeeze(0)
            # --- End Conditional Logit Shifting ---

            # Identify positions that were ACTUALLY masked to mask_id AND are not part of the prefix/padding
            masked_positions_in_noisy = (noisy_seq == mask_id)
            final_masked_for_loss = masked_positions_in_noisy & (~effective_prefix_mask)

            if final_masked_for_loss.any():
                masked_logits = logits_for_loss_calc[final_masked_for_loss]
                masked_labels = seq_i[final_masked_for_loss]
                p_mask_at_masked_pos = p_mask_values[final_masked_for_loss]

                ce = F.cross_entropy(masked_logits, masked_labels, reduction='none')
                scaled_ce = ce / p_mask_at_masked_pos
                pass_loss = scaled_ce.sum()
                accumulated_loss += pass_loss.item()
                actual_passes_with_masking += 1
            else:
                pass  # No tokens were masked in this pass for this sample

        # Average the loss over passes where masking occurred, or over all mc_num passes
        if actual_passes_with_masking > 0:
            final_example_loss = accumulated_loss / float(actual_passes_with_masking)
        elif mc_num > 0:  # No masking happened, but we tried
            final_example_loss = 0.0  # Or accumulated_loss / mc_num which is 0.0
        else:  # mc_num is 0, should not happen if used properly
            final_example_loss = 0.0

        nll_array[i] = final_example_loss
    return nll_array


def batch_nlloss(
        batch: Dict[str, Any],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        shift_logits: bool,
        mask_id: int,
        ignore_prefix: Optional[int] = None,
        key: str = 'nlloss',
        max_length: int = 512,
        mc_num: int = 3
) -> Dict[str, np.ndarray]:  # Return type updated to reflect np.ndarray
    texts = batch['text']
    tokenized = tokenizer.batch_encode_plus(
        texts, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length
    )
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    # Call compute_nlloss directly with the full batch of token_ids and attention_mask
    all_losses_for_batch = compute_nlloss(
        model,
        token_ids,
        attention_mask,
        shift_logits=shift_logits,
        ignore_prefix=ignore_prefix,
        mc_num=mc_num,
        mask_id=mask_id
    )

    # compute_nlloss returns a numpy array, one NLL value per sample in the batch.
    return {key: all_losses_for_batch}


def make_recall_prefix(dataset, n_shots, perplexity_bucket=None):
    """Create a prefix from random samples in the dataset."""
    if perplexity_bucket is not None:
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)
    
    indices = random.sample(range(len(dataset)), n_shots)
    prefixes = [dataset[i]["text"] for i in indices]
    
    return " ".join(prefixes)


def get_model_nll_params(model_instance: torch.nn.Module,
                          default_llada_mask_id: int = 126336,
                          default_dream_mask_id: int = 151666):
    """
    Determines mask_id and shift_logits based on the model's type.
    """
    unwrapped_model = model_instance
    if isinstance(unwrapped_model, torch.nn.DataParallel):
        unwrapped_model = unwrapped_model.module
    # Ensure PeftModel is defined (e.g., from try-except import)
    if 'PeftModel' in globals() and PeftModel is not None and isinstance(unwrapped_model, PeftModel):
        unwrapped_model = unwrapped_model.base_model.model

    model_type_str = None
    config_mask_id = None

    if hasattr(unwrapped_model, 'config'):
        if hasattr(unwrapped_model.config, 'model_type'):
            model_type_str = unwrapped_model.config.model_type
        if hasattr(unwrapped_model.config, 'mask_id') and unwrapped_model.config.mask_id is not None:
            config_mask_id = unwrapped_model.config.mask_id

    determined_shift_logits = False
    determined_mask_id = config_mask_id  # Prioritize mask_id from model's own config if available

    # Match model_type using the keys you registered with AutoConfig (e.g., "llada", "Dream")
    if model_type_str == "Dream":  # Assuming "Dream" is the model_type for DreamConfig
        determined_mask_id = default_dream_mask_id  # Override with specific Dream mask_id
        determined_shift_logits = True
    elif model_type_str == "llada":  # Assuming "llada" is the model_type for LLaDAConfig
        determined_mask_id = default_llada_mask_id  # Override with specific LLaDA mask_id
        determined_shift_logits = False
    else:
        if determined_mask_id is None:  # If not found in config and type is unknown
            determined_mask_id = default_llada_mask_id  # Fallback to a general default
        print(
            f"Unknown model_type '{model_type_str}' for automatic NLL param setting or "
            f"mask_id not found in its config. Using mask_id={determined_mask_id}, "
            f"shift_logits={determined_shift_logits} (default for unknown). "
            f"Model class: {type(unwrapped_model).__name__}"
        )

    print(f"Model NLL params determined for {type(unwrapped_model).__name__} (type: {model_type_str}): "
          f"mask_id={determined_mask_id}, shift_logits={determined_shift_logits}")
    return determined_mask_id, determined_shift_logits
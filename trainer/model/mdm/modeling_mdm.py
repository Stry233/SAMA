import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel

import warnings

# Try importing flash-attn 2.x if available for fast attention:
try:
    from flash_attn import flash_attn_func
    _flash_attn_2_available = True
except ImportError:
    warnings.warn("flash-attn 2.x not available. Falling back to standard attention (slower).")
    _flash_attn_2_available = False

# Try importing xformers.ops.SwiGLU if available:
try:
    from xformers.ops import SwiGLU
    _has_xformers_swiglu = True
except ImportError:
    warnings.warn("xformers SwiGLU not available. Falling back to standard SwiGLU implementation (slower).")
    _has_xformers_swiglu = False

# Try importing flash_attn.layers.rms_norm for fused RMSNorm:
try:
    from flash_attn.ops.rms_norm import RMSNorm as FlashAttnRMSNorm
    _has_flash_attn_rmsnorm = True
except ImportError:
    warnings.warn("flash-attn RMSNorm not available. Falling back to standard RMSNorm (slower).")
    _has_flash_attn_rmsnorm = False


from .configuration_mdm import MDMConfig, MDMModelConfig


# -------------------------------------------------------
#      RMS Norm classes: Fused + fallback
# -------------------------------------------------------
class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm using flash_attn's RMSNorm if available; else fallback to a plain RMSNorm.
    This version registers only one weight param, exactly as a normal RMSNorm would do.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # We'll store our learned weight exactly as in normal RMSNorm:
        self.weight = nn.Parameter(torch.ones(dim))

        if _has_flash_attn_rmsnorm:
            # Construct flash-attn's RMSNorm module
            self.flash_rms = FlashAttnRMSNorm(self.dim, eps=self.eps)
            # Remove the default param in flash_rms and replace with ours
            del self.flash_rms.weight
            self.flash_rms.register_parameter("weight", self.weight)
        else:
            self.flash_rms = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flash_rms is not None:
            # uses flash-attn fused RMSNorm kernel
            return self.flash_rms(x)
        else:
            # Fallback: standard RMSNorm math
            norm_x = torch.mean(x * x, dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(norm_x + self.eps)
            return self.weight * x_normed


class RMSNorm(nn.Module):
    """
    Plain PyTorch RMSNorm fallback, matching many open-source references.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed


def build_norm(config: MDMModelConfig) -> nn.Module:
    """
    Build either FusedRMSNorm, RMSNorm, or nn.LayerNorm, depending on config._norm_class.
    """
    if config._norm_class == "FusedRMSNorm":
        return FusedRMSNorm(config.n_embd, eps=config.norm_eps)
    elif config._norm_class == "RMSNorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    elif config._norm_class == "LayerNorm":
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps)
    else:
        # fallback to PyTorch's LayerNorm if something else is specified
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps)


# -------------------------------------------------------
#      Rotary Embedding Utilities (Partial RoPE)
# -------------------------------------------------------
def build_rope_cache(
    seq_len: int,
    rotary_dim: int,
    condense_ratio: int,
    device,
    base=10000,
    dtype=torch.float16
):
    """
    Build cos/sin caches for partial RoPE usage.
    This matches the approach in GPT-NeoX with partial rotary dimension
    and includes optional 'condense_ratio' to effectively reduce freq scaling.
    """
    theta = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device) / rotary_dim))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)
    if dtype in (torch.float16, torch.bfloat16):
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return (cos, sin)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies RoPE only to the first rotary-dim portion; the trailing portion remains unrotated.
    `x` has shape: (batch, nHeads, seqLen, headDim).
    `cos` / `sin` have shape (seqLen, rotary_dim//2).
    """
    B, nH, T, hd = x.shape
    seqLen = T
    rotary_dim = sin.shape[1] * 2
    # clamp if needed
    if rotary_dim > hd:
        rotary_dim = hd

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # shape: cos_t, sin_t => (1, 1, T, rotary_dim//2)
    cos_t = cos[:seqLen, : (rotary_dim // 2)].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:seqLen, : (rotary_dim // 2)].unsqueeze(0).unsqueeze(0)

    x1 = x_rot[..., : (rotary_dim // 2)]
    x2 = x_rot[..., (rotary_dim // 2) :]

    # standard 2D rotation
    x1_ = x1 * cos_t - x2 * sin_t
    x2_ = x2 * cos_t + x1 * sin_t
    x_rotated = torch.cat([x1_, x2_], dim=-1)

    return torch.cat([x_rotated, x_pass], dim=-1)


# -------------------------------------------------------
#      MLP (Neox / LLaMA style) with optional SwiGLU
# -------------------------------------------------------
class MDMMLP(nn.Module):
    """
    The feed-forward (MLP) block. If _mlp_class=="LLaMAMLP", we do
    a LLaMA-like SwiGLU approach. Otherwise, a GPT-NeoX style MLP (GeLU).
    """
    def __init__(self, config: MDMModelConfig):
        super().__init__()
        hidden_size = config.intermediate_size
        self.config = config

        if config._mlp_class == "LLaMAMLP":
            # LLaMA style
            if _has_xformers_swiglu:
                # If xformers' SwiGLU is available, do that
                self.mlp = SwiGLU(config.n_embd, hidden_size, bias=False, _pack_weights=False)
                self.use_swiglu = True
                self.out_proj = None
            else:
                # minimal fallback for LLaMA MLP
                self.use_swiglu = False
                self.fc_1 = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
                self.fc_2 = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
                self.out_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        else:
            # GPT-NeoX MLP
            self.use_swiglu = False
            self.fc = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
            self.proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
            self.out_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config._mlp_class == "LLaMAMLP":
            if self.use_swiglu:
                # xformers' SwiGLU version
                return self.mlp(x)
            else:
                # fallback LLaMA MLP: silu gate
                x1 = self.fc_1(x)
                x2 = self.fc_2(x)
                x_gate = F.silu(x1) * x2
                return self.out_proj(x_gate)
        else:
            # GPT-NeoX style MLP using GeLU
            x_fc = self.fc(x)
            x_act = F.gelu(x_fc)
            return self.proj(x_act)


# -------------------------------------------------------
#      Self-Attention with GQA + optional flash-attn
# -------------------------------------------------------
class MDMAttention(nn.Module):
    """
    Self-attention that supports GQA / MQA:
      - if n_query_groups < n_head, we have multiple queries but fewer key-value sets.
      - optionally uses flash-attn if installed (and if in fp16/bf16 + on GPU).
    """
    def __init__(self, config: MDMModelConfig):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head

        # qkv_dim = (#queries per group + 1K + 1V) * n_query_groups * head_dim
        qkv_dim = (config.n_head // config.n_query_groups + 2) * config.n_query_groups * head_dim

        # Single linear for Q,K,V
        self.attn = nn.Linear(config.n_embd, qkv_dim, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, rope_cache=None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.attn(x)

        head_dim = self.config.n_embd // self.config.n_head
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # Q's + K + V

        # Reshape to [B, T, n_query_groups, total_qkv, head_dim]
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, head_dim)
        q, k, v = qkv.split([q_per_kv, 1, 1], dim=3)

        # Now reshape queries to [B, T, n_head, head_dim]
        q = q.reshape(B, T, self.config.n_head, head_dim)
        # K, V remain [B, T, n_query_groups, head_dim].

        k = k.reshape(B, T, self.config.n_query_groups, head_dim)
        v = v.reshape(B, T, self.config.n_query_groups, head_dim)

        # Apply RoPE if configured:
        if rope_cache is not None and self.config.rotary_percentage > 0.0:
            cos, sin = rope_cache

            # For the Q part:
            q = q.transpose(1, 2)  # [B, n_head, T, head_dim]
            # For the K part, also transpose:
            k = k.transpose(1, 2)  # [B, n_query_groups, T, head_dim]

            # If n_query_groups < n_head, we might replicate K/V across heads
            # BUT if flash-attn 2.x is installed and in the right dtype + device,
            # we skip manual replication because flash-attn can handle GQA directly.
            if not (
                _flash_attn_2_available
                and q.is_cuda
                and q.dtype in (torch.float16, torch.bfloat16)
            ):
                # replicate if needed
                if self.config.n_query_groups != self.config.n_head:
                    factor = self.config.n_head // self.config.n_query_groups
                    k = k.repeat_interleave(factor, dim=1)

                    v = v.transpose(1, 2)  # shape = [B, n_query_groups, T, head_dim]
                    v = v.repeat_interleave(factor, dim=1).transpose(1, 2)
            else:
                # if flash-attn 2.x is available, do not replicate
                # (we rely on flash-attn's built-in GQA support).
                pass

            # Apply partial RoPE
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # shape back to [B, T, nH, hd] for q
            q = q.transpose(1, 2)
            # shape back to [B, T, nH, hd] for k
            k = k.transpose(1, 2)

        out = self._scaled_dot(q, k, v)
        out = self.proj(out)
        return out

    def _scaled_dot(self, q, k, v) -> torch.Tensor:
        """
        Standard scaled dot-product, with optional flash-attn usage.
        q, k, v => shape [B, T, nH, hd], except k,v might have fewer heads if GQA and
        we haven't replicated them (flash-attn 2 can handle that).
        """
        B, T, nH, d = q.shape
        scale = 1.0 / math.sqrt(d)

        # Decide whether we can use flash-attn:
        use_flash = (
            _flash_attn_2_available
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )

        if not use_flash:
            # If we are not using flash-attn, or if we must replicate K/V:
            if k.size(2) < nH:
                factor = nH // k.size(2)
                k = k.repeat_interleave(factor, dim=2)
                v = v.repeat_interleave(factor, dim=2)

            # fallback to standard matmul
            q = q.permute(0, 2, 1, 3)  # [B, nH, T, d]
            k = k.permute(0, 2, 1, 3)  # [B, nH, T, d]
            v = v.permute(0, 2, 1, 3)  # [B, nH, T, d]
            att = torch.matmul(q, k.transpose(-1, -2)) * scale
            att = F.softmax(att, dim=-1)
            out = torch.matmul(att, v)
            out = out.permute(0, 2, 1, 3).contiguous().view(B, T, nH * d)
        else:
            # flash-attn supports GQA natively, so pass q,k,v as is
            q = q.permute(0, 2, 1, 3)  # [B, nH, T, d]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
            out = out.permute(0, 2, 1, 3).contiguous().view(B, T, nH * d)

        return out


# -------------------------------------------------------
#      MDM Block: Norm -> Attn -> (maybe second norm) -> MLP
# -------------------------------------------------------
class MDMBlock(nn.Module):
    """
    One block of the MDM model. We support:
     - parallel_residual or not
     - shared_attention_norm or not
    """
    def __init__(self, config: MDMModelConfig):
        super().__init__()
        self.config = config
        self.norm_1 = build_norm(config)

        # If not parallel_residual or no shared_attention_norm, we need a second norm:
        if (not config.parallel_residual) or (not config.shared_attention_norm):
            self.norm_2 = build_norm(config)
        else:
            self.norm_2 = None

        self.attn = MDMAttention(config)
        self.mlp = MDMMLP(config)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        # apply the first norm
        n_1 = self.norm_1(x)
        attn_out = self.attn(n_1, rope_cache=rope)

        if self.config.parallel_residual:
            # parallel residual sums attn and mlp at once
            if self.config.shared_attention_norm:
                # same norm for attn + MLP => reuse n_1
                mlp_out = self.mlp(n_1)
                x = x + attn_out + mlp_out
            else:
                # attn uses norm_1, MLP uses norm_2
                n_2 = self.norm_2(x)
                mlp_out = self.mlp(n_2)
                x = x + attn_out + mlp_out
        else:
            # sequential residual
            x = x + attn_out
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No known checkpoint uses non-parallel + shared norm."
                )
            else:
                # attn used norm_1, mlp uses norm_2
                n_2 = self.norm_2(x)
                mlp_out = self.mlp(n_2)
                x = x + mlp_out

        return x


# -------------------------------------------------------
#      The full MDM Model: Embeddings + N blocks + final LN
# -------------------------------------------------------
class MDMModel(nn.Module):
    """
    This is the main “backbone” that includes embeddings, N blocks, final norm,
    plus the output head for logits. We handle partial RoPE for each block if specified.
    """
    def __init__(self, config: MDMModelConfig):
        super().__init__()
        self.config = config
        vocab_size = config.padded_vocab_size or config.vocab_size

        # Embeddings:
        self.wte = nn.Embedding(vocab_size+1, config.n_embd)

        # Blocks:
        self.blocks = nn.ModuleList([MDMBlock(config) for _ in range(config.n_layer)])
        # Final LN:
        self.ln_f = build_norm(config)

        # Final LM head:
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # cached cos/sin for partial RoPE
        self.rope_cache = None

    def _build_rope_cache_if_needed(self, seq_len, device, dtype):
        if self.rope_cache is not None:
            # if we already have a rope cache big enough for seq_len, skip
            if self.rope_cache[0].shape[0] >= seq_len:
                return
        # else build a new rope cache for the full context length
        head_dim = self.config.n_embd // self.config.n_head
        rotary_dim = int(head_dim * self.config.rotary_percentage)
        if rotary_dim > 0:
            self.rope_cache = build_rope_cache(
                seq_len=self.config.block_size,
                rotary_dim=rotary_dim,
                condense_ratio=self.config.condense_ratio,
                device=device,
                dtype=dtype
            )
        else:
            self.rope_cache = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.wte(input_ids)

        # Build rope cache if needed
        if self.config.rotary_percentage > 0.0:
            self._build_rope_cache_if_needed(T, x.device, x.dtype)
        rope = self.rope_cache if self.rope_cache is not None else None

        # pass through blocks
        for block in self.blocks:
            x = block(x, rope=rope)

        # final norm
        x = self.ln_f(x)
        # project to logits
        logits = self.lm_head(x)
        return logits


# -------------------------------------------------------
#      HF PreTrainedModel wrappers for AutoModel
# -------------------------------------------------------
class MDMPreTrainedModel(PreTrainedModel):
    """
    The usual Hugging Face “PreTrainedModel” base class,
    so we can do from_pretrained(..., trust_remote_code=True).
    """
    config_class = MDMConfig
    base_model_prefix = "model"

    def _init_weights(self, module: nn.Module):
        """
        Specialized init to match “original code” style:
         - normal_ with ~ sqrt(2.0/5/n_embd)
         - zero out biases
         - optionally smaller scale for “out_proj” layers
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / 5 / self.config.model_config.n_embd),
            )
        elif isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / 5 / self.config.model_config.n_embd),
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

            # If you'd like to replicate the smaller init scale for final projections
            # e.g. "out_proj" or "proj" layers, we can do name-based or parent-based:
            mod_repr = repr(module)
            # A naive check:
            if ("out_proj" in mod_repr) or ("proj=" in mod_repr):
                # smaller init scale for these layers:
                scale = 1.0 / math.sqrt(self.config.model_config.n_embd)
                # further scale by 1 / #layers if desired (some GPT-NeoX variants do so)
                if hasattr(self.config.model_config, "n_layer") and self.config.model_config.n_layer > 0:
                    scale = scale / float(self.config.model_config.n_layer)
                with torch.no_grad():
                    module.weight.data *= scale

        # For RMSNorm or LN, typically weight=1.0, bias=0. so default init is fine.


class MDMForCausalLM(MDMPreTrainedModel):
    """
    The final HF “model for causal LM” wrapper,
    hooking up the forward pass with optional labels => loss.
    """
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        self.mdmcfg: MDMModelConfig = config.model_config
        self.model = MDMModel(self.mdmcfg)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            # shift by 1 for causal
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            loss=loss,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # no special caching implemented yet
        return {"input_ids": input_ids}

    def can_generate(self) -> bool:
        return True

    # HF-typical functions for retrieving or setting embeddings / output layers:
    def get_input_embeddings(self):
        return self.model.wte

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.model.wte = new_embeddings

    def get_output_embeddings(self):
        print(self.model.lm_head)
        return self.model.lm_head

    def set_output_embeddings(self, new_linear: nn.Module):
        self.model.lm_head = new_linear


# Finally register with HF so “from_pretrained(..., trust_remote_code=True)” works
AutoModel.register(MDMConfig, MDMForCausalLM)

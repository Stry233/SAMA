"""
Configuration class for the MDM-style "TransEncoder" model.

Modeled after `configuration_llada.py`, but for the "MDM" / "TransEncoder" architecture
found in  `lit_gpt.config.Config`.
"""

from typing import Optional, Literal
from dataclasses import dataclass
from transformers import PretrainedConfig, AutoConfig


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class MDMModelConfig:
    """
    The "data class" that stores MDM model hyperparameters, similar to `lit_gpt.config.Config`.
    Typically includes fields:
      - block_size
      - vocab_size
      - padded_vocab_size
      - n_layer, n_head, n_embd
      - rotary_percentage
      - parallel_residual
      - bias
      - n_query_groups
      - shared_attention_norm
      - _norm_class
      - norm_eps
      - _mlp_class
      - intermediate_size
      - condense_ratio
    etc.
    """
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class = "FusedRMSNorm",
    norm_eps: float = 1e-5
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"] = "GptNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1

    def __post_init__(self):
        # replicate your checks or computations, e.g. padded_vocab_size
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)

        if self.n_query_groups is None:
            self.n_query_groups = self.n_head

        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size` for LLaMAMLP")
            self.intermediate_size = 4 * self.n_embd


class MDMConfig(PretrainedConfig):
    """
    Hugging Face config class wrapping an `MDMModelConfig`.

    This lets us do:
       from transformers import AutoConfig
       config = AutoConfig.from_pretrained("my_mdm_checkpoint", trust_remote_code=True)
    and get a config that we can pass to an MDM model for auto-loading.

    We declare `model_type = "mdm"`.
    """
    model_type = "mdm"  # used by HF for auto classes

    def __init__(
        self,
        use_cache: bool = False,
        **kwargs
    ):
        # Construct a default MDMModelConfig (like the "model_config" in LLaDA).
        model_config = MDMModelConfig()
        # Merge any passed kwargs into `model_config`
        # e.g., if user passes `n_layer=12`, we update the data class
        for field_name in model_config.__dataclass_fields__:
            if field_name in kwargs:
                setattr(model_config, field_name, kwargs.pop(field_name))
        # Run __post_init__ on the data class
        model_config.__post_init__()

        # store them in self for reference if needed
        self.model_config = model_config

        # forward the rest to PretrainedConfig
        # e.g., architectures, etc.
        super().__init__(**kwargs)
        self.use_cache = use_cache
        # optionally set default architecture:
        if not hasattr(self, "architectures"):
            self.architectures = ["TransEncoderHF"]  # or similar

    def to_dict(self):
        """
        Override PretrainedConfig.to_dict so that we only serialize
        the raw dataclass fields plus any other primitives, and
        drop the actual model_config object.
        """
        # start with the dataclass fields from model_config
        out = {
            name: getattr(self.model_config, name)
            for name in self.model_config.__dataclass_fields__
        }
        # then any other HF‐level attributes we want to keep
        out["use_cache"] = self.use_cache
        out["model_type"] = self.model_type
        # architectures is often useful for HF pipelines
        if hasattr(self, "architectures"):
            out["architectures"] = self.architectures
        return out

    @property
    def block_size(self) -> int:
        return self.model_config.block_size

    @property
    def vocab_size(self) -> int:
        return self.model_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.model_config.n_embd

    @property
    def num_hidden_layers(self) -> int:
        return self.model_config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self.model_config.n_head

# Register the config class for Hugging Face auto-loading
AutoConfig.register("mdm", MDMConfig)
"""General utility functions for the MIA framework."""

import os
import random
import numpy as np
import torch
from typing import Optional
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def is_huggingface_repo_id(path_candidate: str) -> bool:
    """Check if the given string is likely a Hugging Face repository identifier."""
    if not path_candidate:
        return False
    if os.path.isabs(path_candidate):
        return False
    if path_candidate.startswith(("./", ".\\", "../", "..\\")):
        return False
    if os.path.sep == '\\' and '\\' in path_candidate:
        return False

    try:
        validate_repo_id(path_candidate)
        return True
    except (HFValidationError, Exception):
        return False


def resolve_path(path_val: str, base_dir: str, load_from_base_dir: bool = False) -> str:
    """Resolve a path, handling HuggingFace repo IDs and relative paths."""
    if not path_val:
        return path_val
    if is_huggingface_repo_id(path_val):
        return path_val
    if load_from_base_dir and not os.path.isabs(path_val):
        return os.path.join(base_dir, path_val)
    return path_val
import json
from typing import Dict, Any
import random
import os
import numpy as np
import torch
import transformers
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def set_seed(seed: int):
    """
    Set seed for reproducibility across various libraries and frameworks.
    """
    random.seed(seed)  # Python built-in
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed
    np.random.seed(seed)  # Numpy seed

    # PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hugging Face Transformers seed
    transformers.set_seed(seed)

    # Force deterministic multi-threading behavior
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def save_subset_info(subset, indices, file_path):
    """Save subset information and indices to a JSON file."""
    subset_info = {
        "data": subset,
        "indices": indices
    }
    with open(file_path, "w") as f:
        json.dump(subset_info, f, indent=4)

def format_param_count(num_params: int) -> str:
    """Return a human-friendly string of the parameter count."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)

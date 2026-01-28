"""I/O utilities for saving and loading data."""

import json
import logging
import os
import pickle
import yaml
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def save_metadata(metadata: Dict[str, Any], output_dir: str, filename: str = "metadata.pkl") -> None:
    """Save metadata dictionary to a pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)
    logging.info(f"Metadata saved to {path}")


def load_metadata(filepath: str) -> Dict[str, Any]:
    """Load metadata dictionary from a pickle file."""
    with open(filepath, 'rb') as f:
        metadata = pickle.load(f)
    logging.info(f"Metadata loaded from {filepath}")
    return metadata


def generate_metadata_filename(current_time: str, model_path: str, ds_info_list: list,
                               batch_size: int, test_samples: Optional[int],
                               fpr_thresholds: list, config_name: str, seed: int,
                               lora_adapter_path: Optional[str] = None) -> str:
    """Generate a descriptive filename for metadata."""
    ds_names = "_".join([get_printable_ds_name(ds) for ds in ds_info_list])
    model_name_base = os.path.basename(model_path)

    if lora_adapter_path:
        lora_name = os.path.basename(lora_adapter_path)
        model_name_base = f"{model_name_base}_lora-{lora_name}"

    config_base_name = os.path.splitext(os.path.basename(config_name))[0]
    test_samples_str = f"{test_samples}" if test_samples is not None else "all"
    fpr_str = "-".join(map(str, fpr_thresholds))

    filename = (f"metadata_{current_time}_model-{model_name_base}_config-{config_base_name}_"
                f"datasets-{ds_names}_bs-{batch_size}_ts-{test_samples_str}_"
                f"fpr-{fpr_str}_seed-{seed}.pkl")
    return filename
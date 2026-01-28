"""Dataset loading utilities for the MIA framework."""

import json
import logging
import os
from random import shuffle
from typing import Dict, Any, Optional
from datasets import Dataset, load_dataset

"""Dataset processing utilities including NLL computation."""

import logging
from typing import Dict, Any, Optional
import torch
from datasets import Dataset



class DatasetProcessor:
    """Handles dataset processing and NLL computation."""

    def __init__(self, model, tokenizer, device: torch.device, global_config: Dict[str, Any]):
        """Initialize the dataset processor."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.global_config = global_config

    def init_dataset(self, ds_info: Dict[str, Any], mask_id: int, shift_logits: bool,
                     test_samples: Optional[int] = None) -> Dataset:
        """Load and preprocess a dataset, calculating NLLoss based on model properties."""
        # Load dataset
        dataset = self._load_dataset(ds_info)

        # Sample if needed
        if test_samples is not None and 0 < test_samples < len(dataset):
            seed = self.global_config.get("seed", 42)
            dataset = dataset.shuffle(seed=seed).select(range(test_samples))

        logging.info(f"Processing dataset for NLL: mask_id={mask_id}, shift_logits={shift_logits}")

        # Process dataset with NLL computation
        dataset = self._compute_nll(dataset, mask_id, shift_logits)

        return dataset

    def _load_dataset(self, ds_info: Dict[str, Any]) -> Dataset:
        """Load dataset based on configuration."""
        if "json_train_path" in ds_info and "json_test_path" in ds_info:
            return load_json_dataset(ds_info["json_train_path"], ds_info["json_test_path"])
        elif "mimir_name" in ds_info:
            return load_mimir_dataset(name=ds_info["mimir_name"], split=ds_info["split"])
        elif "name" in ds_info:
            from datasets import load_dataset
            return load_dataset(
                ds_info["name"],
                name=ds_info.get("config_name"),
                split=ds_info["split"],
                trust_remote_code=True
            )
        else:
            raise ValueError("Dataset configuration is missing required keys")

    def _compute_nll(self, dataset: Dataset, mask_id: int, shift_logits: bool) -> Dataset:
        """Compute NLL for the dataset."""

        def process_batch(batch):
            nlloss_batch_result = batch_nlloss(
                batch,
                self.model,
                self.tokenizer,
                self.device,
                shift_logits=shift_logits,
                mask_id=mask_id,
                max_length=self.global_config["max_length"]
            )
            return {**batch, **nlloss_batch_result}

        # Columns to keep
        keep_columns = ['label', 'text', 'nlloss']
        remove_columns = [col for col in dataset.column_names if col not in keep_columns]

        return dataset.map(
            process_batch,
            batched=True,
            batch_size=self.global_config["batch_size"],
            remove_columns=remove_columns,
            load_from_cache_file=False,
            num_proc=self.global_config.get("dataset_map_num_proc", 1),
        )

def get_printable_ds_name(ds_info: Dict[str, Any]) -> str:
    """Generate a printable dataset name from the dataset configuration."""
    name_to_print = "unknown_dataset"

    if "name" in ds_info:
        name_to_print = ds_info["name"]
    elif "mimir_name" in ds_info:
        name_to_print = ds_info["mimir_name"]
    elif "json_train_path" in ds_info:
        parent_dir = os.path.basename(os.path.dirname(ds_info["json_train_path"]))
        name_to_print = parent_dir if parent_dir else "custom_json"

    if "split" in ds_info and ("name" in ds_info or "mimir_name" in ds_info):
        name_to_print = f"{name_to_print}_{ds_info['split']}"

    return name_to_print.replace("/", "_").replace("\\", "_")


def load_mimir_dataset(name: str, split: str) -> Dataset:
    """Load a dataset from the MIMIR collection."""
    dataset = load_dataset("iamgroot42/mimir", name, split=split)

    if 'label' not in dataset.column_names:
        if 'member' in dataset.column_names and 'nonmember' in dataset.column_names:
            all_texts = [dataset['member'][k] for k in range(len(dataset))]
            all_labels = [1] * len(dataset)
            all_texts += [dataset['nonmember'][k] for k in range(len(dataset))]
            all_labels += [0] * len(dataset)

            new_dataset = Dataset.from_dict({"text": all_texts, "label": all_labels})
            return new_dataset
        else:
            raise ValueError("Dataset does not contain required columns")

    return dataset


def load_json_dataset(train_path: str, test_path: str) -> Dataset:
    """Load dataset from JSON files, assigning labels accordingly."""
    # Load train subset (label 1)
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)['data']['text']
    train_labels = [1] * len(train_data)

    # Load test subset (label 0)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)['data']['text']
    test_labels = [0] * len(test_data)

    # Combine and shuffle
    all_texts = train_data + test_data
    all_labels = train_labels + test_labels

    combined = list(zip(all_texts, all_labels))
    shuffle(combined)
    all_texts, all_labels = zip(*combined)

    return Dataset.from_dict({"text": list(all_texts), "label": list(all_labels)})
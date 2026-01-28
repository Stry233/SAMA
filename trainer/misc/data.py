import logging
import multiprocessing
import os
import random
from typing import Dict, Any

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

def load_local_dataset(path: str) -> Dataset:
    """
    Load dataset from a directory or a single file (streaming or not).
    """
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.endswith(".jsonl.zst")
        ]
        return load_dataset("json", data_files=files, split="train", streaming=False)
    elif os.path.isfile(path):
        return load_dataset("json", data_files=path, split="train", streaming=False)
    else:
        raise FileNotFoundError(f"Dataset path {path} does not exist.")

def load_data(dataset_config: Dict[str, Any]) -> Dataset:
    """
    Load a dataset either locally or from the HF Hub
    depending on the 'type' field in dataset_config.
    """
    data_type = dataset_config["type"].lower()

    if data_type == "local":
        return load_local_dataset(dataset_config["path"])
    elif data_type == "huggingface":
        repo_id = dataset_config["repo_id"]
        subset = dataset_config.get("subset")
        split = dataset_config.get("split", "train")
        streaming = dataset_config.get("streaming", False)

        return load_dataset(
            repo_id,
            subset,
            split=split,
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {data_type}")


def preprocess_dataset(dataset: Dataset, tokenizer, max_length: int, streaming: bool) -> Dataset:
    """
    Tokenize the dataset with truncation/padding to max_length.
    """

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        # Explicitly keep the prompt_lengths from the original examples
        if "prompt_lengths" in examples:
            tokenized["prompt_lengths"] = examples["prompt_lengths"]
        return tokenized

    if streaming:
        # Streaming dataset does NOT support num_proc
        return dataset.map(preprocess_function,
                           remove_columns=["text"],
                           batched=True)
    else:
        num_cpu = multiprocessing.cpu_count()
        return dataset.map(preprocess_function,
                           batched=True,
                           remove_columns=["text"],
                           num_proc=num_cpu)


def subset_data(raw_data, subset_size, streaming):
    if streaming:
        # For streaming datasets, we cannot compute len(raw_data), so handle accordingly.
        if subset_size == -1:
            indices = None
        else:
            raise ValueError("Cannot use subset_size with streaming=True as length is unknown.")
    else:
        if subset_size == -1 or subset_size >= len(raw_data):
            indices = list(range(len(raw_data)))
        else:
            indices = random.sample(range(len(raw_data)), subset_size)
        raw_data = raw_data.select(indices)

    return raw_data, indices

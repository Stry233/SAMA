"""
This script processes the MIMIR dataset by:
1. Loading a specific split from 'iamgroot42/mimir' dataset on Hugging Face.
2. Separating the loaded dataset into 'member' and 'non-member' entries.
3. Converting each entry to a single field named 'text' (suitable for tokenization).
4. Writing two JSON files ('train.json' for members, 'test.json' for non-members).

These output files can be loaded by:
    from datasets import load_dataset
    data = load_dataset("json", data_files="/path/to/test.json", split="train", streaming=True)
which will print something like:
    IterableDataset({
        features: Unknown,
        num_shards: 1
    })
"""

import argparse
import os
import json
from datasets import load_dataset


def create_mimir_json(dataset_name="pile_cc", split_name="ngram_7_0.2", output_dir="."):
    """
    Load one split from the MIMIR dataset and generate two JSON files:
    'train.json' with member entries and 'test.json' with non-member entries.

    :param dataset_name: Name of the dataset within MIMIR (e.g., 'pile_cc').
    :param split_name: Name of the split to load (e.g., 'ngram_7_0.2').
    :param output_dir: Directory to store the resulting JSON files.
    """

    # 1. Load the specified split of the MIMIR dataset
    print(f"[INFO] Loading dataset '{dataset_name}' with split '{split_name}' from 'iamgroot42/mimir'...")
    dataset = load_dataset("iamgroot42/mimir", dataset_name, split=split_name)

    # 2. Convert dataset entries into two separate lists (member and non-member)
    print("[INFO] Separating dataset into member and non-member entries...")
    train_data = [{"text": entry["member"]} for entry in dataset]
    test_data = [{"text": entry["nonmember"]} for entry in dataset]

    # 3. Ensure the output directory exists
    output_dir = os.path.join(output_dir, "mimir-" + dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 4. Write the 'member' entries to train.json
    train_file = os.path.join(output_dir, "train.json")
    print(f"[INFO] Writing member entries to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 5. Write the 'non-member' entries to test.json
    test_file = os.path.join(output_dir, "test.json")
    print(f"[INFO] Writing non-member entries to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("[INFO] JSON files have been successfully created!")
    print(f"Train set (members) size: {len(train_data)}")
    print(f"Test set (non-members) size: {len(test_data)}")
    print(f"Output JSON files are stored in: {output_dir}")

ALL_MIMIR_DATASETS = [
    "arxiv", "github", "hackernews", "pile_cc", "pubmed_central", "wikipedia_(en)"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MIMIR benchmark datasets for membership inference experiments."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("SAMA_DATASET_PATH", "./"),
        help="Base output directory for prepared datasets (default: $SAMA_DATASET_PATH or ./)."
    )
    parser.add_argument(
        "--split-name", type=str, default="ngram_13_0.8",
        help="MIMIR split to load (default: ngram_13_0.8)."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_MIMIR_DATASETS,
        choices=ALL_MIMIR_DATASETS,
        help="Which MIMIR datasets to prepare (default: all)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for ds_name in args.datasets:
        create_mimir_json(
            dataset_name=ds_name,
            split_name=args.split_name,
            output_dir=args.output_dir,
        )

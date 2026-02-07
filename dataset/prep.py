import argparse
import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer


def create_subset_dataset(
        dataset_name,
        dataset_config_name=None,
        dataset_split="train",
        text_column="text",
        num_sample=100,
        output_dir="subset_dataset",
        mode="pretraining",  # 'pretraining' or 'sft'
        context=None,  # optional text context to prepend/append
        member_ratio=0.5,  # fraction of chosen subset that goes into "member"
        tokenizer_name=None  # if mode='sft'
):
    """
    Create a membership vs. non-membership subset from a given dataset.

    Two modes are supported:
      - 'pretraining':
          Each example is saved with a single "text" field from `text_column`.
      - 'sft':
          Expects both 'source_text' and 'target_text' columns in the dataset.
          Each example is saved with:
              text = source_text + target_text
              prompt_lengths = length of source_text tokens (using the tokenizer).

    Args:
        dataset_name (str):
            Name or path of the dataset to load (HF Hub or local).
        dataset_config_name (str, optional):
            Config or subset name for the dataset (e.g., "wikitext-103-v1").
        dataset_split (str, optional):
            Which split to load ("train", "test", etc.). Defaults to "train".
        text_column (str, optional):
            Which column to read as the main text for 'pretraining' mode.
        num_sample (int):
            How many total samples to select for this subset.
        output_dir (str):
            Directory to store the resulting JSON files.
        mode (str):
            Either 'pretraining' or 'sft'.
        context (str, optional):
            Optional context string to prepend (or append) to each text.
        member_ratio (float):
            Fraction of samples allocated to the "member" subset. Defaults to 0.5.
        tokenizer_name (str, optional):
            Name of the tokenizer to use if mode='sft'. E.g., "GSAI-ML/LLaDA-8B-Base".
    """

    # 1. Load the dataset (with optional config), e.g.:
    #    load_dataset("wikitext", "wikitext-103-v1", split="train")
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    print(f"[INFO] Loaded dataset '{dataset_name}' "
          f"config='{dataset_config_name}' split='{dataset_split}' with {len(dataset)} rows.")

    # 2. Shuffle the dataset for random sampling
    print("[INFO] Shuffling dataset...")
    shuffled_dataset = dataset.shuffle(seed=42)

    # 3. Select a random subset
    num_sample = min(num_sample, len(shuffled_dataset))
    print(f"[INFO] Selecting {num_sample} samples from the dataset ...")
    subset_dataset = shuffled_dataset.select(range(num_sample))

    # 4. Split into "member" and "non-member" subsets
    member_size = int(member_ratio * len(subset_dataset))
    member_dataset = subset_dataset.select(range(member_size))
    non_member_dataset = subset_dataset.select(range(member_size, len(subset_dataset)))

    # 5. If in 'sft' mode, we require 'source_text' and 'target_text'
    has_source_and_target = ("source_text" in subset_dataset.column_names and
                             "target_text" in subset_dataset.column_names)

    if mode == "sft" and not has_source_and_target:
        raise ValueError(
            f"[ERROR] 'sft' mode requested, but the dataset does not have "
            f"'source_text' and 'target_text' columns. Columns found: {subset_dataset.column_names}"
        )

    tokenizer = None
    if mode == "sft":
        if not tokenizer_name:
            raise ValueError(
                "[ERROR] mode='sft' but no tokenizer_name was provided. "
                "Please provide 'tokenizer_name'."
            )
        print(f"[INFO] Loading tokenizer '{tokenizer_name}' for 'sft' mode...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 6. Create data entries for the two subsets
    member_data = []
    non_member_data = []

    if mode == "pretraining":
        # We only need a single "text" field from `text_column`.
        for row in member_dataset:
            text_value = row[text_column]
            if context:
                text_value = f"{context} {text_value}"
            member_data.append({"text": text_value})

        for row in non_member_dataset:
            text_value = row[text_column]
            if context:
                text_value = f"{context} {text_value}"
            non_member_data.append({"text": text_value})

    else:  # mode == "sft"
        # We expect columns 'source_text' and 'target_text'
        for row in member_dataset:
            source = row.get("source_text", "")
            target = row.get("target_text", "")
            input_ids = tokenizer(source)["input_ids"]
            combined_text = f"{source} {target}"
            if context:
                combined_text = f"{context} {combined_text}"
            member_data.append({
                "text": combined_text,
                "prompt_lengths": len(input_ids),
            })

        for row in non_member_dataset:
            source = row.get("source_text", "")
            target = row.get("target_text", "")
            input_ids = tokenizer(source)["input_ids"]
            combined_text = f"{source} {target}"
            if context:
                combined_text = f"{context} {combined_text}"
            non_member_data.append({
                "text": combined_text,
                "prompt_lengths": len(input_ids),
            })

    # 7. Write out results as a single JSON array per file.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    member_file = os.path.join(output_dir, "train.json")
    non_member_file = os.path.join(output_dir, "test.json")

    print(f"[INFO] Writing member set => {member_file}")
    with open(member_file, "w", encoding="utf-8") as f:
        json.dump(member_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Writing non-member set => {non_member_file}")
    with open(non_member_file, "w", encoding="utf-8") as f:
        json.dump(non_member_data, f, ensure_ascii=False, indent=2)

    # 8. Print a sample entry from member/non-member to confirm correctness.
    if member_data:
        print("[INFO] Sample member entry:", member_data[0])
    else:
        print("[INFO] No member data to sample.")

    if non_member_data:
        print("[INFO] Sample non-member entry:", non_member_data[0])
    else:
        print("[INFO] No non-member data to sample.")

    print("[INFO] Subset dataset generation complete!")
    print(f"Dataset Name         : {dataset_name}")
    print(f"Config Name          : {dataset_config_name}")
    print(f"Split                : {dataset_split}")
    print(f"Mode                 : {mode}")
    print(f"Text Column          : {text_column}")
    print(f"Context Provided     : {context is not None}")
    print(f"Member set size      : {len(member_data)}")
    print(f"Non-member set size  : {len(non_member_data)}")
    print(f"Output directory     : {output_dir}")
    print("---------------------------------------------------\n")


def batch_generate_subsets(dataset_list, output_base_dir=".", num_sample=20000, mode="pretraining",
                           context=None, member_ratio=0.5, tokenizer_name="GSAI-ML/LLaDA-8B-Base"):
    """
    Given a list of dataset specifications, run create_subset_dataset on each.

    Args:
        output_base_dir: Base directory under which per-dataset subdirectories are created.
    """
    for ds_spec in dataset_list:
        ds_name = ds_spec["name"]
        ds_config = ds_spec.get("config", None)
        ds_split = ds_spec.get("split", "train")
        ds_column = ds_spec["text_column"]  # required

        # Build an output directory name, e.g., "subset_wikitext-wikitext-103-v1"
        # if config is present.
        if ds_config:
            subdir = f"{ds_name.split('/')[-1]}-{ds_config}-subset{'-sft' if mode == 'sft' else ''}"
        else:
            subdir = f"{ds_name.split('/')[-1]}-subset{'-sft' if mode == 'sft' else ''}"

        out_dir = os.path.join(output_base_dir, subdir)

        create_subset_dataset(
            dataset_name=ds_name,
            dataset_config_name=ds_config,
            dataset_split=ds_split,
            text_column=ds_column,
            num_sample=num_sample,
            output_dir=out_dir,
            mode=mode,
            context=context,
            tokenizer_name=tokenizer_name,
            member_ratio=member_ratio
        )


MAIN_DATASETS = [
    {
        "name": "EleutherAI/wikitext_document_level",
        "config": "wikitext-103-v1",
        "split": "train",
        "text_column": "page"
    },
    {
        "name": "sh0416/ag_news",
        "config": None,
        "split": "train",
        "text_column": "description"
    },
    {
        "name": "EdinburghNLP/xsum",
        "config": None,
        "split": "train",
        "text_column": "document"
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare standard NLP datasets for membership inference experiments."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("SAMA_DATASET_PATH", "./"),
        help="Base output directory for prepared datasets (default: $SAMA_DATASET_PATH or ./)."
    )
    parser.add_argument(
        "--num-samples", type=int, default=20000,
        help="Number of samples to draw from each dataset (default: 20000)."
    )
    parser.add_argument(
        "--mode", type=str, default="pretraining", choices=["pretraining", "sft"],
        help="Dataset preparation mode (default: pretraining)."
    )
    parser.add_argument(
        "--tokenizer", type=str, default="GSAI-ML/LLaDA-8B-Base",
        help="Tokenizer name for SFT mode (default: GSAI-ML/LLaDA-8B-Base)."
    )
    parser.add_argument(
        "--member-ratio", type=float, default=0.5,
        help="Fraction of samples allocated to the member subset (default: 0.5)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_generate_subsets(
        dataset_list=MAIN_DATASETS,
        output_base_dir=args.output_dir,
        num_sample=args.num_samples,
        mode=args.mode,
        tokenizer_name=args.tokenizer,
        member_ratio=args.member_ratio,
    )

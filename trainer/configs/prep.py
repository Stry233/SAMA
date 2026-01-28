import os

# -----------------------------------------------------------------------------
# 1. CONFIGURATION SECTION: Adjust all parameters here
# -----------------------------------------------------------------------------

# --- Output Directory ---
# The generated .yaml files will be saved in this folder.
OUTPUT_CONFIG_DIR = "./"

# --- Models & Datasets ---
# Lists of models and datasets to generate configs for.
MODELS = [
    "GSAI-ML/LLaDA-8B-Base",
    "Dream-org/Dream-v0-Base-7B",
]

DATASETS = [
    "mimir-arxiv",
    "mimir-github",
    "mimir-pubmed_central",
    "mimir-pile_cc",
    "wikitext-wikitext-103-v1",
    "ag_news",
    "xsum",
    "cosmopedia-auto_math_text",
    "cosmopedia-khanacademy",
    "cosmopedia-stanford",
    "cosmopedia-stories",
    "cosmopedia-web_samples_v2",
]

# --- Base Paths ---
# The base directory where your datasets are stored.
BASE_DATASET_PATH = "YOUR_PATH_HERE"

# --- W&B Settings ---
WANDB_PROJECT = "Diff_LLM"
WANDB_GROUP = "SAMA"

# --- Tokenizer Settings ---
TOKENIZER_MAX_LENGTH = 128

# --- Training Hyperparameters ---
# All training parameters are defined here for easy modification.
TRAINING_PARAMS = {
    "batch_size": 1,
    "gradient_accumulation_steps": 48,
    "learning_rate": "1.0e-5",
    "weight_decay": 0.1,
    "warmup_steps": 500,
    "num_train_epochs": 10,
    "max_steps": -1,  # Must be -1 to use num_train_epochs
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "save_total_limit": -1,
    "bf16": True,
    "fp16": False,
}

# -----------------------------------------------------------------------------
# 2. YAML TEMPLATE: The structure of the config file
# -----------------------------------------------------------------------------

# This is the template for the YAML files.
# Placeholders like {run_name}, {model_identifier}, etc., will be filled in by the script.
YAML_TEMPLATE = """
run_name: {run_name}
wandb_project: {wandb_project}
wandb_group: {wandb_group}

dataset:
  train:
    type: local
    path: {train_path}
  # val not provided -> We'll use test for val
  test:
    type: local
    path: {test_path}

tokenizer:
  identifier: {model_identifier}
  max_length: {tokenizer_max_length}

model:
  identifier: {model_identifier}
  load_pretrained: true

training:
  output_dir: {output_dir}
  # deepspeed_config: configs/ds_config.json
  batch_size: {batch_size}
  gradient_accumulation_steps: {gradient_accumulation_steps}
  learning_rate: {learning_rate}
  weight_decay: {weight_decay}
  warmup_steps: {warmup_steps}
  num_train_epochs: {num_train_epochs}         # For epoch-based training
  max_steps: {max_steps}                # Must be -1 to fully rely on epochs
  eval_strategy: {eval_strategy}
  save_strategy: {save_strategy}
  logging_strategy: {logging_strategy}
  save_total_limit: {save_total_limit}
  bf16: {bf16_str}
  fp16: {fp16_str}
  #early_stopping_patience: 3
  #early_stopping_threshold: 0.001
"""


# -----------------------------------------------------------------------------
# 3. SCRIPT LOGIC: Do not modify below this line
# -----------------------------------------------------------------------------

def generate_configs():
    """
    Generates YAML config files for all combinations of models and datasets.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_CONFIG_DIR, exist_ok=True)

    total_files = len(MODELS) * len(DATASETS)
    print(f"🚀 Starting config generation for {len(MODELS)} models and {len(DATASETS)} datasets.")
    print(f"Total files to be generated: {total_files}")
    print("-" * 40)

    for model_id in MODELS:
        for dataset_name in DATASETS:
            # --- Derive names and paths ---
            model_short_name = model_id.split('/')[-1]

            run_name = f"{model_short_name}-pretrained-{dataset_name}"

            # Manually format the learning rate string to avoid padded exponents (e.g., 'e-05' -> 'e-5')
            lr_val = TRAINING_PARAMS['learning_rate']
            # Construct the output directory name based on key parameters
            output_dir_suffix = (
                f"{TRAINING_PARAMS['batch_size']}_"
                f"{TRAINING_PARAMS['gradient_accumulation_steps']}_"
                f"{TRAINING_PARAMS['learning_rate']}_"
                f"{TRAINING_PARAMS['num_train_epochs']}_"
                f"{TOKENIZER_MAX_LENGTH}"
            )
            output_dir = f"./{run_name}-{output_dir_suffix}"

            # Construct dataset paths
            dataset_path_prefix = f"{BASE_DATASET_PATH}/{dataset_name}-subset"
            train_path = f"{dataset_path_prefix}/train.json"
            test_path = f"{dataset_path_prefix}/test.json"

            # --- Prepare values for the template ---
            format_args = {
                "run_name": run_name,
                "wandb_project": WANDB_PROJECT,
                "wandb_group": WANDB_GROUP,
                "train_path": train_path,
                "test_path": test_path,
                "model_identifier": model_id,
                "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
                "output_dir": output_dir,
                **TRAINING_PARAMS,  # Unpack all training params
                # Convert booleans to lowercase 'true'/'false' for YAML
                "bf16_str": str(TRAINING_PARAMS["bf16"]).lower(),
                "fp16_str": str(TRAINING_PARAMS["fp16"]).lower(),
            }

            # --- Generate the YAML content ---
            yaml_content = YAML_TEMPLATE.format(**format_args)

            # --- Write the file ---
            file_name = f"{run_name}.yaml"
            file_path = os.path.join(OUTPUT_CONFIG_DIR, file_name)

            with open(file_path, 'w') as f:
                f.write(yaml_content.strip())

            print(f"✅ Generated: {file_path}")

    print("-" * 40)
    print(f"🎉 Success! All {total_files} config files have been generated in the '{OUTPUT_CONFIG_DIR}' directory.")


if __name__ == "__main__":
    generate_configs()
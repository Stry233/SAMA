import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


from misc.data import load_data, preprocess_dataset, subset_data
from misc.env_setup import setup_environment, prepare_save_paths
from misc.models import load_model, prepare_tokenizer
from misc.utils import load_config, set_seed, save_subset_info
from train import initialize_trainer, run_training

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train & track router signals in a MoE model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/LLaDA-115m-Base-raw-pretraining-lora-16-mimir-arxiv.yaml",
        help="Path to your config file (YAML)."
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="./",
        help="Base path for saving outputs."
    )
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=-1,
        help="Size of training subset."
    )
    parser.add_argument(
        "--ref_subset_size",
        type=int,
        default=-1,
        help="Size of reference (test/val) subset."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For DeepSpeed or torch.distributed: local rank."
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        choices=["nccl", "gloo"],
        default="nccl",
        help="Distributed backend for torch.distributed: 'nccl' for GPUs, 'gloo' for CPU or troubleshooting."
    )
    return parser.parse_args()

def load_and_prepare_datasets(config):
    """
    Load train, test, and val datasets from config. Apply fallback logic where needed.
    """
    logger.info("Loading TRAIN dataset...")
    train_config = config["dataset"]["train"]
    raw_train = load_data(train_config)

    logger.info("Loading TEST dataset...")
    test_config = config["dataset"].get("test")
    raw_test = load_data(test_config) if test_config else None

    logger.info("Loading VAL dataset...")
    val_config = config["dataset"].get("val")
    raw_val = load_data(val_config) if val_config else None

    # Fallback logic: use test as val if val not provided, or vice versa
    if raw_val is None and raw_test is not None:
        logger.warning("VAL dataset not found. Using TEST dataset as fallback for VAL.")
        raw_val = raw_test

    if raw_test is None and raw_val is not None:
        logger.warning("TEST dataset not found. Using VAL dataset as fallback for TEST.")
        raw_test = raw_val

    if raw_test is None and raw_val is None:
        raise ValueError("Neither 'test' nor 'val' dataset is provided. Please specify at least one.")

    return raw_train, raw_test, raw_val


def subset_and_save(raw_train, raw_test, train_config, test_config, output_dir,
                    train_subset_size, test_subset_size):
    """
    Take subsets from train/test datasets (if not streaming),
    then save the subset indices and data for reference.
    """
    train_streaming = train_config.get("streaming", False)
    test_streaming = test_config.get("streaming", False) if test_config else False

    raw_train, train_indices = subset_data(raw_train, train_subset_size, train_streaming)
    raw_test, test_indices = subset_data(raw_test, test_subset_size, test_streaming)

    train_subset_file = os.path.join(output_dir, "train_subset.json")
    test_subset_file = os.path.join(output_dir, "test_subset.json")

    if not train_streaming:
        save_subset_info(raw_train.to_dict(), train_indices, train_subset_file)
        logger.info(f"Saved train subset information to {train_subset_file}")
    else:
        logger.info("Train dataset is streaming; skipping saving subset information.")

    if not test_streaming:
        save_subset_info(raw_test.to_dict(), test_indices, test_subset_file)
        logger.info(f"Saved test subset information to {test_subset_file}")
    else:
        logger.info("Test dataset is streaming; skipping saving subset information.")

    return raw_train, raw_test


def main():
    # 1. Parse arguments
    args = parse_arguments()
    set_seed(42)

    # 2. Load config + environment
    config = load_config(args.config_path)
    config = prepare_save_paths(args.base_path, config)
    setup_environment(config, args)

    # 3. Prepare tokenizer
    tokenizer = prepare_tokenizer(config["tokenizer"]["identifier"])

    # 4. Load raw datasets + fallback
    raw_train, raw_test, raw_val = load_and_prepare_datasets(config)

    # 5. Subset + save
    output_dir = config["training"]["output_dir"]
    train_config = config["dataset"]["train"]
    test_config = config["dataset"].get("test") if config["dataset"].get("test") else config["dataset"].get("val")
    raw_train, raw_test = subset_and_save(
        raw_train, raw_test, train_config, test_config,
        output_dir, args.train_subset_size, args.ref_subset_size
    )

    # 6. Preprocess
    max_len = config["tokenizer"]["max_length"]
    train_streaming = train_config.get("streaming", False)
    test_streaming = test_config.get("streaming", False) if test_config else False
    train_size = len(raw_train) if not train_streaming else 'streaming'
    test_size = len(raw_test) if not test_streaming else 'streaming'
    logger.info(f"Tokenizing train={train_size}, test={test_size} ...")

    tokenized_train = preprocess_dataset(raw_train, tokenizer, max_len, train_streaming)
    tokenized_test = preprocess_dataset(raw_test, tokenizer, max_len, test_streaming)

    if not(config["dataset"].get("test") and config["dataset"].get("val")):
        tokenized_val = tokenized_test
    else:
        val_config = config["dataset"]["val"]
        val_streaming = val_config.get("streaming", False) if val_config else False
        tokenized_val = preprocess_dataset(raw_val, tokenizer, max_len, val_streaming)

    # 7. Load model
    model_id = config["model"]["identifier"]
    checkpoint_path = config["model"].get("checkpoint_path")
    load_pretrained = config["model"].get("load_pretrained")
    logger.info(f"Loading model from {model_id} ...")
    model = load_model(config, model_id, load_pretrained, checkpoint_path)
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    mask_id = config["tokenizer"].get("mask_id", 126336)

    # 8. Initialize Trainer
    trainer = initialize_trainer(
        config,
        model,
        tokenized_train,
        tokenized_val,
        tokenizer,
        mask_id
    )

    # 9. Run training
    run_training(trainer, tokenizer, config)

    logger.info("All done!")


if __name__ == "__main__":
    main()

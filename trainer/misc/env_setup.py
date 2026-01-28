import os
import logging
from typing import Dict, Any
import torch.distributed as dist

logger = logging.getLogger(__name__)

def setup_environment(config: Dict[str, Any], args) -> None:
    """
    Set environment variables for W&B and any other environment-level settings.
    """
    os.environ["WANDB_PROJECT"] = config["wandb_project"]
    os.environ["WANDB_RUN_GROUP"] = config["wandb_group"]
    logger.info("Environment variables for W&B are set.")

    if "LOCAL_RANK" in os.environ:
        logger.info(f"Initializing torch.distributed with backend='{args.dist_backend}'")
        dist.init_process_group(backend=args.dist_backend, init_method="env://")


def prepare_save_paths(base_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare and update output/log paths in the config based on base_path.
    """
    abs_base_path = os.path.abspath(base_path)
    config["training"]["output_dir"] = os.path.join(abs_base_path, config["training"]["output_dir"])
    config["training"]["logging_dir"] = os.path.join(config["training"]["output_dir"], "logs")
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    os.makedirs(config["training"]["logging_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["training"]["output_dir"], "ckpts"), exist_ok=True)

    logger.info(
        "Output directories set to: %s (base), %s (logs)",
        config["training"]["output_dir"],
        config["training"]["logging_dir"],
    )
    return config

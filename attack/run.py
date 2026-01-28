#!/usr/bin/env python
"""Main entry point for the MIA framework."""

import argparse
import logging
import os
import time
from typing import Optional

import torch
from tabulate import tabulate

from attack.misc.attack import load_attack
from attack.misc.config import ConfigManager
from attack.misc.dataset import DatasetProcessor, get_printable_ds_name
from attack.misc.io import generate_metadata_filename, save_metadata
from attack.misc.metric import results_with_bootstrapping
from attack.misc.models import ModelManager
from attack.misc.utils import set_seed, resolve_path

logging.basicConfig(level=logging.INFO)


class MIARunner:
    """Main runner for Membership Inference Attacks."""

    def __init__(self, args):
        self.args = args
        self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.config_manager = ConfigManager(args.config)
        self.global_config = self.config_manager.get_global_config()
        self.global_config['seed'] = args.seed

        set_seed(args.seed)
        os.makedirs(args.output, exist_ok=True)

        # Register custom models
        ModelManager.register_custom_models()

        # Initialize model
        self.model, self.tokenizer, self.device = self._initialize_model()

        # Get model-specific parameters
        self.shift_logits, self.mask_id = ModelManager.get_model_nll_params(self.model)

        # Initialize dataset processor
        self.dataset_processor = DatasetProcessor(
            self.model, self.tokenizer, self.device, self.global_config
        )

    def _initialize_model(self):
        """Initialize the model with proper path resolution."""
        load_from_base_dir = self.global_config.get("load_from_base_dir", False)

        # Resolve model path
        model_path_from_config = self.global_config.get('target_model')
        model_path = self.args.target_model if self.args.target_model else model_path_from_config

        if not model_path:
            raise ValueError("Target model path must be provided")

        model_path = resolve_path(model_path, self.args.base_dir, load_from_base_dir)

        # Resolve tokenizer path
        tokenizer_path = self.global_config.get('tokenizer', model_path)
        tokenizer_path = resolve_path(tokenizer_path, self.args.base_dir, load_from_base_dir)

        # Resolve LoRA path
        lora_path = self.args.lora_path or self.global_config.get('lora_adapter_path')
        lora_path = resolve_path(lora_path, self.args.base_dir, load_from_base_dir)

        logging.info(f"Loading base model from: {model_path}")
        if lora_path:
            logging.info(f"Loading LoRa adapter from: {lora_path}")

        device = torch.device(self.global_config["device"])
        return ModelManager.init_model(model_path, tokenizer_path, device, lora_path)

    def run(self):
        """Run the MIA attacks on all configured datasets."""
        header = ["MIA Attack", "AUC"] + [f"TPR@FPR={t}" for t in self.global_config["fpr_thresholds"]]
        results_to_print = {}
        metadata = self._initialize_metadata()

        for ds_info in self.config_manager.get_datasets():
            ds_name = get_printable_ds_name(ds_info)
            logging.info(f"Processing dataset: {ds_name}")

            # Process dataset
            dataset = self.dataset_processor.init_dataset(
                ds_info, self.mask_id, self.shift_logits,
                self.global_config.get("test_samples")
            )

            # Run attacks
            attack_results = self._run_attacks_on_dataset(dataset, ds_name)

            # Store results
            results_to_print[ds_name] = tabulate(attack_results, headers=header, tablefmt="outline")
            self._update_metadata(metadata, ds_name, attack_results, dataset, header)

        # Save and display results
        self._save_and_display_results(metadata, results_to_print)

    def _run_attacks_on_dataset(self, dataset, ds_name):
        """Run all configured attacks on a dataset."""
        attack_results = []

        for attack_name, attack_module in self.config_manager.get_available_attacks().items():
            logging.info(f"Running attack '{attack_name}' on dataset '{ds_name}'")

            # Get attack config
            attack_config = self.config_manager.get_attack_config(attack_name)
            attack_config['base_dir'] = self.args.base_dir
            attack_config['model_mask_id'] = self.mask_id
            attack_config['model_shift_logits'] = self.shift_logits

            # Reset seed for reproducibility
            set_seed(self.args.seed)

            # Load and run attack
            attack_instance = load_attack(
                attack_name, attack_module, self.model,
                self.tokenizer, attack_config, self.device
            )
            processed_dataset = attack_instance.run(dataset)

            # Compute metrics
            y_true = processed_dataset['label']
            y_score = processed_dataset[attack_name]

            metrics = results_with_bootstrapping(
                y_true, y_score,
                self.global_config["fpr_thresholds"],
                self.global_config["n_bootstrap_samples"]
            )

            attack_results.append([attack_name] + metrics)
            logging.info(f"Attack '{attack_name}' results: {metrics}")

        return attack_results

    def _initialize_metadata(self):
        """Initialize metadata dictionary."""
        return {
            "timestamp": self.current_time,
            "model": self._get_model_description(),
            "config_file": self.args.config,
            "config_content": self.config_manager.config,
            "results": {}
        }

    def _get_model_description(self):
        """Get a description of the model including LoRA if applicable."""
        model_path = self.global_config.get('target_model', 'unknown')
        lora_path = self.args.lora_path or self.global_config.get('lora_adapter_path')

        if lora_path:
            return f"{model_path} (LoRa: {os.path.basename(lora_path)})"
        return model_path

    def _update_metadata(self, metadata, ds_name, attack_results, dataset, header):
        """Update metadata with results for a dataset."""
        metadata["results"][ds_name] = {
            "attacks": [row[0] for row in attack_results],
            "results_table_rows": attack_results,
            "results_header": header,
            "ground_truth_labels": list(map(int, dataset['label'])),
        }

    def _save_and_display_results(self, metadata, results_to_print):
        """Save metadata and display results."""
        metadata_filename = generate_metadata_filename(
            self.current_time,
            self.global_config.get('target_model', 'unknown'),
            self.config_manager.get_datasets(),
            self.global_config["batch_size"],
            self.global_config.get("test_samples"),
            self.global_config["fpr_thresholds"],
            self.args.config,
            self.args.seed,
            self.args.lora_path or self.global_config.get('lora_adapter_path')
        )

        save_metadata(metadata, self.args.output, metadata_filename)

        # Display results
        for ds_name, table_str in results_to_print.items():
            print(f"\nResults for Dataset: {ds_name}")
            print(table_str)

        print(f"\nMetadata saved to: {os.path.join(self.args.output, metadata_filename)}")
        print(f"Model: {self._get_model_description()}")
        print(f"Shift Logits: {self.shift_logits}, Mask ID: {self.mask_id}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Membership Inference Attacks on Diffusion LLMs.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to the configuration YAML file.")
    parser.add_argument('--output', type=str, required=True,
                        help="Directory to save results and metadata.")
    parser.add_argument('--target-model', type=str,
                        help="Path to the base model. Overrides config if provided.")
    parser.add_argument('--lora-path', type=str,
                        help="Optional path to LoRa adapter. Overrides config if provided.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--base-dir', type=str, default="./",
                        help='Base directory for resolving relative paths in config.')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    runner = MIARunner(args)
    runner.run()


if __name__ == '__main__':
    main()
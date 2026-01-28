import logging
import math
import os
from typing import Dict, Any, Optional, Union

import torch
from opacus import PrivacyEngine

logger = logging.getLogger(__name__)

from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback, get_scheduler,
)

# Attempt to import PEFT libraries for LoRA
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Attempt to import DP libraries
try:
    import opacus
    from dp_transformers import DPCallback
    from opacus.accountants import RDPAccountant
    from prv_accountant import Accountant as PRVAccountant
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

from model.llada.configuration_llada import LLaDAConfig
from model.llada.modeling_llada import LLaDAModelLM
from model.dream.configuration_dream import DreamConfig
from model.dream.modeling_dream import DreamModel
from misc.trainer import DiffusionLLMTrainer

AutoConfig.register("llada", LLaDAConfig)
AutoModel.register(LLaDAConfig, LLaDAModelLM)
AutoConfig.register("Dream", DreamConfig)
AutoModel.register(DreamConfig, DreamModel)


def initialize_trainer(
        config: Dict[str, Any],
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        tokenizer: AutoTokenizer,
        mask_id: int = 126336,
) -> Union[DiffusionLLMTrainer, Trainer]:
    """
    Construct the Trainer with TrainingArguments, optionally enabling differential privacy
    using Opacus PrivacyEngine and the provided DPCallback.
    """
    logger_init = logging.getLogger(__name__)
    training_cfg = config["training"]
    lora_cfg = config.get("lora", {})
    privacy_args = config.get("privacy", {})

    train_mode = config.get("train_mode", "pretraining").lower().strip()
    run_name = config.get("run_name", "diffusion-llm-run")

    use_dp = privacy_args.get("enabled", False)
    actual_noise_multiplier = None  # Will be set if use_dp

    if lora_cfg.get("enabled", False):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "LoRA is enabled but PEFT library is not available. Please install it with `pip install peft`.")
        logger_init.info("LoRA is enabled. Preparing model for PEFT...")
        # (LoRA model preparation logic as in the original code)
        is_quantized = getattr(model, "is_loaded_in_8bit", False) or \
                       getattr(model, "is_loaded_in_4bit", False)
        use_gradient_checkpointing_for_prep = training_cfg.get("gradient_checkpointing", False)

        if not training_cfg.get("deepspeed_config"):
            if use_gradient_checkpointing_for_prep and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        if is_quantized or (use_gradient_checkpointing_for_prep and not training_cfg.get("deepspeed_config")):
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=use_gradient_checkpointing_for_prep
            )
            logger_init.info(
                f"Model prepared for k-bit/gradient checkpointing compatibility (use_gradient_checkpointing_for_prep={use_gradient_checkpointing_for_prep}).")

        lora_r = lora_cfg.get("r", 8)
        lora_alpha = lora_cfg.get("lora_alpha", 16)
        lora_dropout = lora_cfg.get("lora_dropout", 0.05)
        lora_target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        lora_bias = lora_cfg.get("bias", "none")
        raw_task_type = lora_cfg.get("task_type", "CAUSAL_LM")
        peft_task_type = getattr(TaskType, raw_task_type.upper(), None)

        if peft_task_type is None:
            logger_init.warning(f"Invalid LoRA task_type '{raw_task_type}'. Defaulting to CAUSAL_LM.")
            peft_task_type = TaskType.CAUSAL_LM
        elif peft_task_type.value != raw_task_type.upper() and hasattr(TaskType, raw_task_type.upper()):
            logger_init.warning(
                f"LoRA task_type '{raw_task_type}' resolved to '{peft_task_type}'. Ensure this is intended."
            )

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=peft_task_type,
        )
        model = get_peft_model(model, peft_config)
        logger_init.info("Successfully applied LoRA to the model.")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    optimizer_to_pass_to_trainer = None
    scheduler_to_pass_to_trainer = None

    # Store temp_data_loader_for_opacus if DP is used, for logging warning later
    # This variable will be defined inside the `if use_dp` block
    _temp_loader_len_for_warning = 0

    if use_dp:
        if not DP_AVAILABLE:
            raise ImportError("Differential Privacy is enabled but opacus library is not available.")

        logger_init.info("DP is enabled. Initializing Opacus PrivacyEngine...")
        privacy_engine = PrivacyEngine(
            accountant=privacy_args.get("accountant", "prv"),
            secure_mode=privacy_args.get("secure_mode", False)
        )

        initial_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"]
        )

        data_collator_for_temp_loader = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        temp_data_loader_for_opacus = DataLoader(
            train_dataset,
            batch_size=training_cfg["batch_size"],
            collate_fn=data_collator_for_temp_loader,
            shuffle=True
        )
        _temp_loader_len_for_warning = len(temp_data_loader_for_opacus)

        num_samples = len(train_dataset)
        eff_batch_size_for_epochs_calc = (
                training_cfg["batch_size"] *
                training_cfg.get("gradient_accumulation_steps", 1) *
                int(os.environ.get("WORLD_SIZE", 1))  # Estimate world size, TrainingArgs will have definitive
        )

        if training_cfg.get("max_steps", -1) > 0:
            num_total_optimizer_steps = training_cfg["max_steps"]
            if eff_batch_size_for_epochs_calc > 0 and num_samples > 0:
                optimizer_steps_per_epoch = math.ceil(num_samples / eff_batch_size_for_epochs_calc)
                if optimizer_steps_per_epoch > 0:
                    num_epochs_for_opacus = math.ceil(num_total_optimizer_steps / optimizer_steps_per_epoch)
                else:  # Should not happen if num_samples and eff_batch_size > 0
                    num_epochs_for_opacus = training_cfg.get("num_train_epochs", 1)
                num_epochs_for_opacus = max(1, int(num_epochs_for_opacus))
                logger_init.info(f"Calculated num_epochs_for_opacus based on max_steps: {num_epochs_for_opacus}")
            else:  # Fallback if eff_batch_size or num_samples is zero
                num_epochs_for_opacus = int(training_cfg.get("num_train_epochs", 1))
                logger_init.warning(
                    f"Using num_train_epochs for Opacus: {num_epochs_for_opacus} due to zero effective batch size or dataset size for max_steps calculation.")

        else:
            num_epochs_for_opacus = int(training_cfg.get("num_train_epochs", 1))
            logger_init.info(f"Using num_train_epochs for Opacus: {num_epochs_for_opacus}")

        logger_init.info(
            f"Making model, optimizer, and dataloader private with target_epsilon={privacy_args['target_epsilon']}, "
            f"target_delta={privacy_args['target_delta']}")

        model, wrapped_optimizer, _ = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=initial_optimizer,
            data_loader=temp_data_loader_for_opacus,
            target_epsilon=privacy_args["target_epsilon"],
            target_delta=privacy_args["target_delta"],
            epochs=num_epochs_for_opacus,
            max_grad_norm=privacy_args["max_grad_norm"],
            poisson_sampling=privacy_args.get("opacus_poisson_sampling", True),
            clipping=privacy_args.get("clipping", "flat"),
        )
        logger_init.info("Opacus PrivacyEngine has prepared the DP model and DP optimizer.")

        actual_noise_multiplier = wrapped_optimizer.noise_multiplier
        logger_init.info(f"Actual noise multiplier determined by Opacus: {actual_noise_multiplier}")

        optimizer_to_pass_to_trainer = wrapped_optimizer

        # Calculate num_training_steps for the scheduler
        if training_cfg.get("max_steps", -1) > 0:
            num_training_steps_for_scheduler = training_cfg["max_steps"]
        else:
            # This needs the per-device batch size for the temp_data_loader_for_opacus
            optimizer_steps_per_epoch_for_scheduler = math.ceil(
                len(temp_data_loader_for_opacus) / training_cfg["gradient_accumulation_steps"])
            optimizer_steps_per_epoch_for_scheduler = max(1, optimizer_steps_per_epoch_for_scheduler)
            num_training_steps_for_scheduler = num_epochs_for_opacus * optimizer_steps_per_epoch_for_scheduler

        scheduler_to_pass_to_trainer = get_scheduler(
            name=training_cfg.get("lr_scheduler_type", "linear"),
            optimizer=wrapped_optimizer,
            num_warmup_steps=training_cfg.get("warmup_steps", 0) * training_cfg.get("gradient_accumulation_steps", 1),
            num_training_steps=num_training_steps_for_scheduler
        )
        logger_init.info("Created LR scheduler for the DP optimizer.")

    training_args_dict = {
        "output_dir": os.path.join(training_cfg["output_dir"], "ckpts"),
        "num_train_epochs": training_cfg.get("num_train_epochs", 5),
        "max_steps": training_cfg.get("max_steps", -1),
        "per_device_train_batch_size": training_cfg["batch_size"],
        "per_device_eval_batch_size": training_cfg.get("eval_batch_size", training_cfg["batch_size"]),
        "eval_accumulation_steps": training_cfg.get("eval_accumulation_steps"),
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "learning_rate": training_cfg["learning_rate"],
        "weight_decay": training_cfg["weight_decay"],
        "warmup_steps": training_cfg.get("warmup_steps", 0),
        "save_total_limit": training_cfg.get("save_total_limit", -1),
        "fp16": training_cfg.get("fp16", False),
        "bf16": training_cfg.get("bf16", True),
        "eval_strategy": training_cfg.get("eval_strategy", "epoch") if eval_dataset is not None else "no",
        "save_strategy": training_cfg.get("save_strategy", "epoch"),
        "logging_strategy": training_cfg.get("logging_strategy", "steps"),
        "eval_steps": training_cfg.get("eval_steps", None) if (
                    eval_dataset is not None and training_cfg.get("eval_strategy") == "steps") else None,
        "save_steps": training_cfg.get("save_steps", None) if training_cfg.get("save_strategy") == "steps" else None,
        "logging_steps": training_cfg.get("logging_steps", 50),
        "load_best_model_at_end": training_cfg.get("load_best_model_at_end", False),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "eval_loss") if training_cfg.get(
            "load_best_model_at_end", False) else None,
        "report_to": ["wandb"] if config.get("wandb_project") else training_cfg.get("report_to", "none"),
        "run_name": run_name,
        "logging_dir": training_cfg.get("logging_dir", os.path.join(training_cfg["output_dir"], "logs")),
        "deepspeed": training_cfg.get("deepspeed_config", None),
        "remove_unused_columns": False if train_mode == "sft" or use_dp else True,
        "gradient_checkpointing": training_cfg.get("gradient_checkpointing", False) and not training_cfg.get(
            "deepspeed_config"),
    }

    if training_args_dict["load_best_model_at_end"] and not training_args_dict["metric_for_best_model"]:
        training_args_dict["metric_for_best_model"] = "eval_loss"
        logger_init.info("load_best_model_at_end is True, setting metric_for_best_model to 'eval_loss'.")

    training_args = TrainingArguments(**training_args_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    callbacks = []
    if "early_stopping_patience" in training_cfg:
        if not training_args.load_best_model_at_end:
            training_args.load_best_model_at_end = True
            training_args.metric_for_best_model = "eval_loss"
            logger_init.info("Enabled load_best_model_at_end and metric_for_best_model for early stopping.")

        early_stopping_threshold = training_cfg.get("early_stopping_threshold", 0.0)
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_cfg["early_stopping_patience"],
                early_stopping_threshold=early_stopping_threshold
            )
        )
        logger_init.info(
            "EarlyStopping enabled: patience=%s, threshold=%.4f",
            training_cfg["early_stopping_patience"], early_stopping_threshold
        )

    # Add DP callback if enabled
    if use_dp:
        # actual_noise_multiplier has been retrieved from Opacus optimizer
        if actual_noise_multiplier is None:  # Should not happen if use_dp is true and previous block ran
            raise ValueError("actual_noise_multiplier is not set despite use_dp=True.")

        effective_batch_size_for_callback = (
                training_args.per_device_train_batch_size *
                training_args.world_size *  # This is correctly taken from TrainingArguments
                training_args.gradient_accumulation_steps
        )
        if len(train_dataset) == 0:
            raise ValueError("train_dataset is empty, cannot calculate sampling_probability for DPCallback.")

        sampling_probability_for_callback = effective_batch_size_for_callback / len(train_dataset)

        rdp_accountant_for_dp_callback = RDPAccountant()
        prv_accountant_for_dp_callback = PRVAccountant(
            noise_multiplier=actual_noise_multiplier,
            sampling_probability=sampling_probability_for_callback,
            delta=privacy_args["target_delta"],
            eps_error=privacy_args.get("eps_error", 0.1),
            max_compositions=privacy_args.get("max_compositions", 10000)
        )

        dp_callback_instance = DPCallback(
            noise_multiplier=actual_noise_multiplier,
            target_delta=privacy_args["target_delta"],
            sampling_probability=sampling_probability_for_callback,
            rdp_accountant=rdp_accountant_for_dp_callback,
            prv_accountant=prv_accountant_for_dp_callback,
            max_epsilon=privacy_args.get("max_epsilon", float('inf'))
        )
        callbacks.append(dp_callback_instance)
        logger_init.info("Added DPCallback, configured with Opacus-derived noise multiplier.")
        if _temp_loader_len_for_warning > 0:  # Check if _temp_loader_len_for_warning was set
            opacus_internal_sample_rate = 1 / _temp_loader_len_for_warning
            logger_init.warning(
                "DPCallback is active alongside PrivacyEngine's internal accountant. "
                "Ensure you are tracking epsilon from the intended source (likely DPCallback's logs via Trainer). "
                "Sampling rate interpretation may differ: "
                f"PrivacyEngine's internal accountant uses physical batch rate (approx {opacus_internal_sample_rate:.2e}), "
                f"while DPCallback's accountants use effective batch rate ({sampling_probability_for_callback:.2e}). "
                "These rates will differ if using >1 GPU or gradient accumulation >1, affecting parallel epsilon calculations."
            )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_constructor_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "callbacks": callbacks,
        "tokenizer": tokenizer,
    }

    if optimizer_to_pass_to_trainer is not None:
        trainer_constructor_args["optimizers"] = (optimizer_to_pass_to_trainer, scheduler_to_pass_to_trainer)
        logger_init.info("Passing Opacus DP optimizer and its LR scheduler to the Trainer.")

    if training_cfg.get("use_custom_diffusion_trainer", True):
        trainer = DiffusionLLMTrainer(
            train_mode=train_mode,
            mask_id=mask_id,
            **trainer_constructor_args
        )
        logger_init.info(
            f"Initialized DiffusionLLMTrainer with train_mode={train_mode}, run_name={run_name}, DP={use_dp}.")
    else:
        trainer = Trainer(**trainer_constructor_args)
        logger_init.info(f"Initialized Standard Hugging Face Trainer with run_name={run_name}, DP={use_dp}.")

    if lora_cfg.get("enabled", False) and PEFT_AVAILABLE and hasattr(model, 'peft_config'):
        logger_init.info(f"LoRA Config for the model: {model.peft_config if hasattr(model, 'peft_config') else 'N/A'}")

    return trainer

def run_training(trainer: Trainer, tokenizer: AutoTokenizer, config: Dict[str, Any]) -> None:
    """
    Run training and save final model+tokenizer.
    """
    current_logger = logging.getLogger(__name__)
    current_logger.info("Starting training...")
    training_cfg = config["training"]

    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", None)
    if resume_from_checkpoint is True:
        current_logger.info(f"Attempting to resume training from the latest checkpoint in {trainer.args.output_dir}.")
        train_output = trainer.train(resume_from_checkpoint=True)
    elif isinstance(resume_from_checkpoint, str):
        current_logger.info(f"Attempting to resume training from checkpoint: {resume_from_checkpoint}.")
        train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_output = trainer.train()

    current_logger.info("Training complete.")
    current_logger.info(f"TrainOutput: {train_output}")

    final_model_save_path = config["training"]["output_dir"]
    os.makedirs(final_model_save_path, exist_ok=True)

    current_logger.info(f"Saving model to {final_model_save_path}...")
    model_to_save = trainer.model

    # Check if the model is wrapped with GradSampleModule (DP)
    if DP_AVAILABLE and isinstance(model_to_save, opacus.GradSampleModule):
        current_logger.info("GradSampleModule detected. Accessing underlying model for saving.")
        model_to_save = model_to_save._module  # Access the underlying model

    # Check if the model is a PeftModel (LoRA)
    if PEFT_AVAILABLE and isinstance(model_to_save, PeftModel):
        current_logger.info("LoRA (PEFT) model detected. Saving adapter model (delta weights).")
    else:
        current_logger.info("Saving full model.")

    model_to_save.save_pretrained(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    current_logger.info(f"Model/Adapter and tokenizer saved to {final_model_save_path}.")

    lora_cfg = config.get("lora", {})
    if PEFT_AVAILABLE and isinstance(model_to_save, PeftModel) and lora_cfg.get("save_merged_model_at_end", False):
        merged_model_path = os.path.join(final_model_save_path, "merged_model")
        os.makedirs(merged_model_path, exist_ok=True)
        current_logger.info(f"Merging LoRA weights and saving full merged model to {merged_model_path}...")
        try:
            merged_model = model_to_save.merge_and_unload()
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            current_logger.info(f"Merged model saved to {merged_model_path}.")
        except Exception as e:
            current_logger.error(f"Could not merge and save LoRA model: {e}. "
                                 "Ensure you have enough CPU RAM/GPU RAM (depending on where merge happens) "
                                 "and the model supports merging (e.g., not all quantized models merge easily).")
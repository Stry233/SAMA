"""Attack loading utilities for the MIA framework."""

import importlib
import logging
from typing import Any, Dict


def load_attack(attack_name: str, real_attack_name: str, model, tokenizer,
                config: Dict[str, Any], device):
    """
    Load and instantiate an attack class dynamically.

    Args:
        attack_name: The name of the attack in the config
        real_attack_name: The actual module name of the attack
        model: The model to attack
        tokenizer: The tokenizer for the model
        config: Configuration dictionary for the attack
        device: The device to run on
    """
    module = importlib.import_module(f"attacks.{config['module']}")

    # Convert attack module name to class name
    class_name = ''.join(word.capitalize().split("-")[0]
                         for word in real_attack_name.split('_')) + 'Attack'
    class_name = class_name.replace('OfWords', 'ofWords')

    logging.info(f"Loading attack: {class_name} from module attacks.{config['module']}")

    attack_class = getattr(module, class_name)
    return attack_class(
        name=attack_name,
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )
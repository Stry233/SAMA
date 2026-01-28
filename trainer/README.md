# Trainer

This repository contains the essential scripts and configurations to train the DLM using a custom trainer.

## Repository Structure
```
./train
├── configs/                     
│   └── *YAML configuration files*         # Contains various YAML files defining training setups 
│                                          # and model configurations (e.g., raw data, masking options).
│                                          # Also includes a subfolder with model definition JSONs.
├── LICENSE
├── llada/                          
│   ├── configuration_llada.py
│   ├── generation_config.json
│   └── modeling_llada.py
├── misc/
│   ├── data.py                      # Data loading and preprocessing routines
│   ├── env_setup.py                 # Environment setup utilities
│   ├── models.py                    # Utilities for loading pretrained models or raw weights from JSON
│   └── utils.py                     # Additional helper functions
├── README.md                        # This file
├── requirements.txt
├── run.py                           # Main script to initiate train.py using a selected config
├── setup.py
└── train.py

```

## Setup Instructions

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

Use `run.py` to start the training process with your desired configuration.

### Command Line Arguments:
```bash
python run.py \
  --config_path "./train/configs/LLaDA-1B-Base-json.yaml" \
  --base_path "./outputs" \
  --train_subset_size 10000 \
  --ref_subset_size 1000
```

Note that `transformers==4.38.2` is required if we need to replicate results from original LLaDA implementation

### Argument Details:
- `--config_path`: Path to YAML config file specifying model and training details.
- `--base_path`: Directory to save outputs and logs.
  - Note that the actual path would be the concatenation of both `base_path` & `output_dir` specified in your yaml
- `train_subset_size` & `ref_subset_size`: Control sizes of training and validation subsets.

## Training Modes
`trainer.py` supports two modes defined in the YAML configs:
- **Pretraining**: Masks input tokens randomly to pretrain the model from scratch.
- **SFT**: Masks tokens selectively (excluding prompts) for fine-tuning.

### Training Logic Overview
- The `trainer.py` overrides the default training logic from Hugging Face's Trainer.
- Implements token masking logic to introduce noise and improve robustness during training given by LLaDA's [GUIDELINE.md](https://huggingface.co/spaces/multimodalart/LLaDA/blob/main/GUIDELINES.md)

Adjust YAML files in `configs` to change model size, dataset locations, and training parameters.

## Dependencies
Install dependencies before training:
```bash
pip install -r requirements.txt
```

Just in case, ensure `datasets` paths and model identifiers in YAML configs are correctly set.

Good luck!


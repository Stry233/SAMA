# Membership Inference Attacks on Finetuned Diffusion Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 11.0+](https://img.shields.io/badge/CUDA-11.0+-76b900.svg)](https://developer.nvidia.com/cuda-downloads)

This repository contains the implementation of SAMA (Subset-Aggregated Membership Attack) and baseline methods for evaluating membership inference vulnerabilities in Diffusion Language Models (DLMs).

## Repository Structure

```
.
├── trainer/         # DLM training module
│   ├── model/       # Model architectures (LLaDA, Dream, MDM)
│   ├── configs/     # Training configurations
│   └── train.py     # Main training script
├── attack/          # MIA implementation module  
│   ├── attacks/     # Attack implementations (SAMA, baselines)
│   ├── configs/     # Attack configurations
│   └── run.py       # Main attack script
├── dataset/         # Dataset preparation utilities
└── scripts/         # Execution scripts
```

## Requirements

### Environment Setup
```bash
# Create conda environment
conda create -n dlm-mia python=3.8
conda activate dlm-mia

# Install dependencies
pip install -r trainer/requirements.txt
pip install -r attack/requirements.txt
```

## Usage

### Step 1: Prepare Datasets

```bash
# Download and prepare MIMIR benchmark datasets
python dataset/prep_mimir.py

# Prepare standard NLP datasets
python dataset/prep.py
```

### Step 2: Train Target DLM Models

```bash
# Train LLaDA-8B on ArXiv dataset
python trainer/train.py \
    --config trainer/configs/LLaDA-8B-Base-arxiv.yaml \
    --output_dir ./models/llada-arxiv
```

### Step 3: Run Membership Inference Attacks

```bash
# Run all attack
bash attack_full.sh

```

## Attack Configurations

Edit `attack/configs/config_all.yaml` to modify attack parameters:

```yaml
sama:
  steps: 16              # Progressive masking steps
  min_mask_frac: 0.05    # Starting mask fraction  
  max_mask_frac: 0.50    # Ending mask fraction
  num_subsets: 128       # Subsets per step
  subset_size: 10        # Tokens per subset
  batch_size: 8          # Batch size
  
# Baseline configurations
loss:
  batch_size: 32
  mask_ratio: 0.15

```

## Implemented Attacks

### Main Method
- **SAMA**: Novel attack exploiting DLMs' bidirectional masking with robust aggregation

### Baseline Methods
- **Autoregressive-adapted**: Loss, ZLIB, Lowercase, Min-K%, Min-K%++, BoWs, ReCall, CON-ReCall, Ratio
- **Diffusion-specific**: SecMI, PIA

## Output Format

Results are saved as JSON files containing:
- Attack scores for each sample
- AUC and TPR metrics
- Metadata for analysis (if enabled)

```json
{
  "scores": [0.82, 0.31, ...],
  "metrics": {
    "auc": 0.850,
    "tpr_at_10fpr": 0.586,
    "tpr_at_1fpr": 0.178
  }
}
```

## Cite our work

TBD
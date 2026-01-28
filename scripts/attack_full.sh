#!/bin/bash


# Base directory for all paths
BASE_DIR="YOUR_BASE_DIR"

cd ..
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Run the evaluation
python attack/run.py \
    -c attack/configs/config_all.yaml \
    --output "$BASE_DIR/attack_res/full_eval" \
    --base-dir $BASE_DIR

echo "Complete."
echo "========================================"

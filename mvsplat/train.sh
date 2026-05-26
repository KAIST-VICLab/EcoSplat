#!/usr/bin/env bash
# Launch EcoSplat IGF stage-2 finetune of mvsplat on RE10K.
# Run from the mvsplat/ directory.
#
# Prereq (one-time):
#   pip install -e ../ecosplat_wrapper
#
# Usage:
#   ./train.sh
#   CUDA_VISIBLE_DEVICES=0,1 ./train.sh
#   ./train.sh data_loader.train.batch_size=8 wandb.mode=online
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python -m src.main +experiment=re10k_igf "$@"

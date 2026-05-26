#!/usr/bin/env bash
# Launch IGF stage-2 finetune with backbone + mean head FROZEN (stage2 SPFSplat style).
# Only the IGF merge head (`to_gaussians` + `to_disparity` clones), the 256-ch
# rate_embed, and the 256->32 igf_rate_proj are trainable.
#
# All other ZPressor settings inherit from re10k_igf.yaml.
#
# Prereq (one-time):
#   pip install -e ../ecosplat_wrapper
#
# Usage:
#   ./train_frozen.sh
#   CUDA_VISIBLE_DEVICES=0,1 ./train_frozen.sh
#   ./train_frozen.sh data_loader.train.batch_size=2 wandb.mode=online
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

python -m src.main +experiment=re10k_igf \
    model.encoder.freeze_pretrained=true \
    "$@"

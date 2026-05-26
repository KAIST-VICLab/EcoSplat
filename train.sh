#!/usr/bin/env bash
# Launch EcoSplat (SPFSplat base) IGF stage-2 training on RE10K (10-view).
# Fine-tune from a stage-1 checkpoint; override CKPT / CUDA as needed:
#   CKPT=pretrained_weights/re10k_10view.ckpt CUDA_VISIBLE_DEVICES=0,1 ./train.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5}"
export CUDA_VISIBLE_DEVICES
CKPT="${CKPT:-pretrained_weights/re10k_10view.ckpt}"

python -m src.main +experiment=spfsplat/re10k_10view \
    wandb.mode=offline wandb.name=ecowrapper_spfsplat \
    checkpointing.load="${CKPT}" \
    checkpointing.resume=true "$@"

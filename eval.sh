#!/usr/bin/env bash
# Evaluate an EcoSplat (SPFSplat base) checkpoint: novel-view synthesis + pose.
# Required: CKPT. Optional: EVAL_INDEX, CUDA_VISIBLE_DEVICES.
#   CKPT=path/to/checkpoint.ckpt ./eval.sh
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
CKPT="${CKPT:?set CKPT=path/to/checkpoint.ckpt}"
EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"

python -m src.main +experiment=spfsplat/re10k \
    mode=test \
    wandb.mode=disabled wandb.name=eval_re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path="${EVAL_INDEX}" \
    checkpointing.load="${CKPT}" \
    test.save_image=true test.align_pose=true \
    test.output_path=outputs/test/eval_re10k "$@"

#!/usr/bin/env bash
# Resume from the IGF stage-2 checkpoint and CONTINUE training with L_io DROPPED.
#
# Why: the released ckpt's merged Gaussians are over-opaque (mean ~0.86 vs ~0.26
# for the full set) because L_io (the importance-mask BCE) pins kept-Gaussian
# opacity toward 1. That over-coverage + the exposed far Gaussians are the haze /
# sky-floater artifacts. Here we KEEP the top-k merged rendering and the photometric
# losses (MSE + LPIPS) + the alpha-coverage term, but DROP the mask-BCE (io_weight=0)
# so the network self-calibrates opacity / placement from the image loss.
#
# The backbone is already trainable (the release was NOT frozen). We load WEIGHTS
# ONLY (resume=false) and start a fresh optimizer + cosine LR for this finetune
# phase — the prior 300k cosine already decayed to ~0, so a real resume would train
# at LR~0. The PLGC rho-curriculum re-anneals from step 0.
#
# Run from mvsplat/ (the script cd's there). Override anything via "$@", e.g.:
#   CUDA_VISIBLE_DEVICES=7 ./train_resume_noio.sh wandb.mode=online
set -euo pipefail
cd "$(dirname "$0")"

: "${CUDA_VISIBLE_DEVICES:=7}"
export CUDA_VISIBLE_DEVICES

# Use the zpressor conda env's python (override with PYTHON=... or `conda activate zpressor`).
PYTHON="${PYTHON:-/home/viclab/anaconda3/envs/zpressor/bin/python}"
CKPT="${CKPT:-/home/quan/20228248/spfsplat_ori/datasets/spfsplt_outputs/exp_ecowrapper_spfsplat/epoch_22-step_300000.ckpt}"

"$PYTHON" -m src.main +experiment=re10k_igf \
    checkpointing.load="${CKPT}" \
    checkpointing.resume=false \
    model.encoder.unimatch_weights_path=null \
    +model.encoder.igf.io_weight=0.0 \
    +train.random_bg=true \
    optimizer.lr=2.e-5 \
    optimizer.warm_up_steps=2000 \
    trainer.max_steps=100001 \
    data_loader.train.batch_size=1 \
    wandb.name=re10k_igf_noio \
    wandb.mode=offline \
    "$@"

# --- Variants ---
# Fully disable the IGF loss (also drops the alpha-coverage term L_acc):
#     +model.encoder.igf.loss_weight=0.0
# True continue (same optimizer/global_step) instead of a fresh phase (LR ~0):
#     checkpointing.resume=true trainer.max_steps=400001

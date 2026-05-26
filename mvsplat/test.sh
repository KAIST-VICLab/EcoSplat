#!/usr/bin/env bash
# Test the EcoSplat IGF stage-2 checkpoint on RE10K (multi-view eval split).
# Run from the mvsplat/ directory.
#
# Usage:
#   ./test.sh
#   CUDA_VISIBLE_DEVICES=0 ./test.sh
#   ./test.sh test.save_image=true test.save_video=true
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=6}"
export CUDA_VISIBLE_DEVICES

CKPT="${CKPT:?set CKPT=path/to/igf_checkpoint.ckpt}"
INDEX="assets/re10k_evaluation/evaluation_index_re10k_small_24views.json"
OUT="outputs/test/re10k_igf-frozen-24views-p07"

python -m src.main +experiment=re10k_igf \
    mode=test \
    checkpointing.load="${CKPT}" \
    checkpointing.resume=false \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path="${INDEX}" \
    +model.encoder.igf.inference_rho=0.7 \
    test.compute_scores=true \
    test.save_image=true \
    test.save_gt_image=false \
    test.save_input_images=false \
    test.save_video=false \
    test.stablize_camera=false \
    test.dec_chunk_size=8 \
    test.output_path="${OUT}" \
    wandb.mode=disabled \
    "$@"

# INDEX="assets/re10k_evaluation/evaluation_index_re10k_small_24views.json"
# OUT="outputs/test/re10k_igf-epoch17-step230000-24views-p01"

# python -m src.main +experiment=re10k_igf \
#     mode=test \
#     checkpointing.load="${CKPT}" \
#     checkpointing.resume=false \
#     dataset/view_sampler=evaluation \
#     dataset.view_sampler.index_path="${INDEX}" \
#     +model.encoder.igf.inference_rho=0.1 \
#     test.compute_scores=true \
#     test.save_image=true \
#     test.save_gt_image=false \
#     test.save_input_images=false \
#     test.save_video=false \
#     test.stablize_camera=false \
#     test.dec_chunk_size=8 \
#     test.output_path="${OUT}" \
#     wandb.mode=disabled \
#     "$@"

# INDEX="assets/re10k_evaluation/evaluation_index_re10k_small_24views.json"
# OUT="outputs/test/re10k_igf-epoch17-step230000-24views-p002"

# python -m src.main +experiment=re10k_igf \
#     mode=test \
#     checkpointing.load="${CKPT}" \
#     checkpointing.resume=false \
#     dataset/view_sampler=evaluation \
#     dataset.view_sampler.index_path="${INDEX}" \
#     +model.encoder.igf.inference_rho=0.02 \
#     test.compute_scores=true \
#     test.save_image=true \
#     test.save_gt_image=false \
#     test.save_input_images=false \
#     test.save_video=false \
#     test.stablize_camera=false \
#     test.dec_chunk_size=8 \
#     test.output_path="${OUT}" \
#     wandb.mode=disabled \
#     "$@"

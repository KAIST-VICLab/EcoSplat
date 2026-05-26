#!/usr/bin/env bash
# Evaluate the IGF-enabled model with protect_rate (rho) = 0.10 (10%).
#
# The encoder reads cfg.igf.inference_rho at eval (training=False, no per-call
# override), so overriding model.encoder.igf.inference_rho=0.1 sets rho=0.10
# for every forward pass. See src/model/encoder/encoder_spfsplat.py:236 and
# ecosplat_wrapper/igf_module.py::IGFModule.get_rho.
#
# Test split: 24 context views, 3 target views per scene.
# Override EVAL_INDEX with your multi-view eval index json.

set -euo pipefail

EVAL_INDEX="${EVAL_INDEX:-assets/evaluation_index_re10k.json}"
CKPT="${CKPT:?set CKPT=path/to/your/igf_checkpoint.ckpt}"
RUN_NAME="re10k_24view_rho0p7"
OUT="outputs/test/${RUN_NAME}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
python -m src.main +experiment=spfsplat/re10k_10view \
    mode=test \
    wandb.mode=disabled \
    wandb.name="${RUN_NAME}" \
    +model.encoder.igf.inference_rho=0.7 \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path="${EVAL_INDEX}" \
    dataset.re10k.view_sampler.num_context_views=24 \
    checkpointing.load="${CKPT}" \
    test.save_image=false \
    test.align_pose=true \
    test.compute_scores=true \
    test.output_path="${OUT}"
<p align="center">
  <h1 align="center"><img src="https://kaist-viclab.github.io/ecosplat-site/static/image/icon.jpg" width="28" style="position: relative; top: 3px;"> EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images</h1>
  <h2 align="center">Training code — SPFSplat base</h2>
  <p align="center">
    <a href="https://sites.google.com/view/jongmin-park">Jongmin Park</a><sup>1*</sup>
    ·
    <a href="https://quan5609.github.io/">Minh-Quan Viet Bui</a><sup>1*</sup>
    ·
    <a href="https://sites.google.com/view/juan-luis-gb/home">Juan Luis Gonzalez Bello</a><sup>1</sup>
    ·
    <a href="https://sites.google.com/view/jaehomoon">Jaeho Moon</a><sup>1</sup>
    ·
    <a href="https://cmlab.cau.ac.kr/">Jihyong Oh</a><sup>2†</sup>
    ·
    <a href="https://www.viclab.kaist.ac.kr/">Munchurl Kim</a><sup>1†</sup>
    <br>
    <sup>1</sup>KAIST, South Korea, <sup>2</sup>Chung-Ang University, South Korea
    <br>
    *Co-first authors (equal contribution), †Co-corresponding authors
  </p>
  <h3 align="center">CVPR 2026 (Highlight)</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2512.18692">Paper</a> | <a href="https://kaist-viclab.github.io/ecosplat-site/">Project Page</a> | <a href="https://github.com/KAIST-VICLab/EcoSplat">Code</a> | <a href="https://huggingface.co/ImJongminPark/EcoSplat">Models</a> </h3>
</p>

---

This branch holds the **training and inference code for EcoSplat on the [SPFSplat](https://github.com/ranrhuang/SPFSplat) base** (pose-free, sparse-view feed-forward 3DGS). EcoSplat is **also released on a [ZPressor](https://github.com/ziplab/ZPressor) base** for scalable many-view input — see the [`eco_zpressor`](https://github.com/KAIST-VICLab/EcoSplat/tree/eco_zpressor) branch.

EcoSplat's efficiency control is implemented as a reusable, base-agnostic package, [`ecosplat_wrapper/`](ecosplat_wrapper) (the **IGF** training strategy), vendored into this branch. To understand the mechanism or apply it to another base model, read [`ecosplat_wrapper/README.md`](ecosplat_wrapper/README.md); for the exact changes we made to SPFSplat (forward + training), see [`INTEGRATION.md`](INTEGRATION.md).

> **EcoSplat base variants — pick the branch for your setup:**
> - **SPFSplat** — *this branch* ([`eco_spfsplat`](https://github.com/KAIST-VICLab/EcoSplat/tree/eco_spfsplat)): pose-free.
> - **ZPressor** — [`eco_zpressor`](https://github.com/KAIST-VICLab/EcoSplat/tree/eco_zpressor): both inter-view and intra-view compression (MVSplat baseline + IGF).

## Table of Contents
- [Installation](#installation)
- [Datasets](#datasets)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [Training](#training)
- [Evaluation](#evaluation)
- [Camera Conventions](#camera-conventions)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

This branch is a fork of **[SPFSplat](https://github.com/ranrhuang/SPFSplat)** — set up its environment first (conda env, dependencies, and the optional CroCo RoPE CUDA kernels) following [SPFSplat's installation](https://github.com/ranrhuang/SPFSplat#installation). Then install the EcoSplat efficiency-control package (IGF) into that environment:

```bash
pip install -e ecosplat_wrapper
```

## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Pre-trained Checkpoints
The EcoSplat (IGF) checkpoint is on [Hugging Face 🤗](https://huggingface.co/quan5609/EcoSplat) — `ecosplat-spfsplat-re10k.ckpt`. Download it into `pretrained_weights/`:

```bash
wget -P pretrained_weights https://huggingface.co/quan5609/EcoSplat/resolve/main/ecosplat-spfsplat-re10k.ckpt
```

EcoSplat training starts from a **stage-1 SPFSplat checkpoint** — download one from the [SPFSplat model zoo](https://huggingface.co/RanranHuang/SPFSplat) (e.g. `re10k.ckpt`, `re10k_10view.ckpt`).

## Training

EcoSplat's **IGF** training finetunes a converged **stage-1 base checkpoint** (a standard SPFSplat model — see [Pre-trained Checkpoints](#pre-trained-checkpoints)). IGF is controlled by the `model.encoder.igf` config (an empty `{}` uses the paper defaults from [`IGFConfig`](ecosplat_wrapper/README.md#igfconfig): `loss_weight=0.1`, `io_weight=0.1`, PLGC `0.85→0.95`); the provided `spfsplat/re10k_10view` experiment already sets `igf: {}`.

```bash
python -m src.main +experiment=spfsplat/re10k_10view \
    checkpointing.load=pretrained_weights/re10k_10view.ckpt checkpointing.resume=false \
    wandb.mode=online wandb.name=re10k_igf
```

**`bash train.sh`** wraps the multi-GPU launch used for the released model (set `CKPT` to your stage-1 checkpoint). Tune the strategy via `model.encoder.igf` fields, e.g. `model.encoder.igf.io_weight=0.2`.

## Evaluation

Evaluate novel-view synthesis at a chosen **primitive budget** — the protect rate κ (`inference_rho`); lower κ → fewer rendered Gaussians. The bundled launchers `eval_igf_rho{0p7,0p4,0p1,0p02}.sh` sweep the budget — edit the checkpoint / index paths inside them, then run e.g. `bash eval_igf_rho0p4.sh`.

The underlying command:

```bash
python -m src.main +experiment=spfsplat/re10k_10view mode=test wandb.name=re10k_igf \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=pretrained_weights/ecosplat-spfsplat-re10k.ckpt \
    +model.encoder.igf.inference_rho=0.4 \
    test.compute_scores=true test.align_pose=true
```

Sweep `inference_rho` (`0.7 → 0.4 → 0.1 → 0.02`, default `0.4`) to trade primitive count against quality. The published inference release exposes a **related** knob, `model.encoder.primitive_ratio`, on its dedicated `encoder_ecosplat` — it re-allocates the budget **per view** by high-frequency content, whereas this wrapper applies a single **uniform** rate.

## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system: normalized intrinsics (first row ÷ width, second row ÷ height), and OpenCV-style camera-to-world extrinsics (+X right, +Y down, +Z into the screen).

## Acknowledgements
This project is built upon these excellent repositories: [SPFSplat](https://github.com/ranrhuang/SPFSplat), [NoPoSplat](https://github.com/cvg/NoPoSplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). We thank the original authors for their excellent work.

## Citation
If you find EcoSplat useful, please cite:

```bibtex
@inproceedings{park2025ecosplat,
      title={EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images}, 
      author={Jongmin Park and Minh-Quan Viet Bui and Juan Luis Gonzalez Bello and Jaeho Moon and Jihyong Oh and Munchurl Kim},
        year = {2026},
      booktitle={CVPR},
      }
```

Please also consider citing the SPFSplat base:

```bibtex
@article{huang2025spfsplat,
      title={No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views},
      author={Huang, Ranran and Mikolajczyk, Krystian},
      journal={arXiv preprint arXiv: 2508.01171},
      year={2025}
    }
```

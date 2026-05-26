<p align="center">
  <h1 align="center"><img src="https://kaist-viclab.github.io/ecosplat-site/static/image/icon.jpg" width="28" style="position: relative; top: 3px;"> EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images</h1>
  <h2 align="center">Training code — ZPressor base</h2>
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

This branch holds the **training and inference code for EcoSplat built on the [ZPressor](https://github.com/ziplab/ZPressor) base** — applying EcoSplat's efficiency control to scalable, many-view feed-forward 3DGS.

EcoSplat's efficiency control is implemented as a reusable, base-agnostic package, [`ecosplat_wrapper/`](ecosplat_wrapper) (the **IGF** training strategy), vendored into this branch. To understand the mechanism or apply it to another baseline, read [`ecosplat_wrapper/README.md`](ecosplat_wrapper/README.md); for the exact changes we made to ZPressor's MVSplat baseline (forward + training), see [`INTEGRATION.md`](INTEGRATION.md).

> **Scope:** This branch ships the **MVSplat** baseline (`mvsplat/`) — the one EcoSplat IGF is wired into. ZPressor's other baselines (DepthSplat, pixelSplat) are not included here; get them from upstream [ZPressor](https://github.com/ziplab/ZPressor) and port IGF with the wrapper guide.
>
> **Other base models:** EcoSplat's IGF on the SPFSplat base lives on the [`eco_spfsplat`](https://github.com/KAIST-VICLab/EcoSplat/tree/eco_spfsplat) branch.

## Installation

This branch is a fork of **[ZPressor](https://github.com/ziplab/ZPressor)** — set up its environment first (conda env, dependencies, and the `zpressor` package) following [ZPressor's installation](https://github.com/ziplab/ZPressor#installation). Then install the EcoSplat efficiency-control package (IGF) into that environment:

```bash
pip install -e ecosplat_wrapper
```

## Model Zoo

EcoSplat (ZPressor base) finetunes from a converged ZPressor stage-1 checkpoint. The EcoSplat (IGF) checkpoint is on [Hugging Face 🤗](https://huggingface.co/quan5609/EcoSplat); the ZPressor stage-1 baseline is on [Hugging Face](https://huggingface.co/lhmd/ZPressor). Place weights in `mvsplat/pretrained/`.

| Model | Role | Training data | Download |
| --- | --- | --- | --- |
| `ecosplat-zpressor-re10k.ckpt` | EcoSplat (IGF) | RealEstate10K | [download](https://huggingface.co/quan5609/EcoSplat/resolve/main/ecosplat-zpressor-re10k.ckpt) |
| `mvsplat-re10k-zpressor-n200-256x256.ckpt` | ZPressor stage-1 (MVSplat) | RealEstate10K | [download](https://huggingface.co/lhmd/ZPressor/resolve/main/mvsplat-re10k-zpressor-n200-256x256.ckpt) |

(See the [ZPressor model zoo](https://huggingface.co/lhmd/ZPressor) for the DepthSplat / pixelSplat variants.)

## Datasets

The MVSplat baseline trains and evaluates on RealEstate10K. Acquire it following [pixelSplat](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) (preprocessed copies: [RE10K](https://huggingface.co/datasets/lhmd/re10k_torch), [ACID](https://huggingface.co/datasets/lhmd/acid_torch)). Expected layout:

```
datasets
└── re10k
    ├── train/{000000.torch, ..., index.json}
    └── test/{000000.torch, ..., index.json}
```

Symlink the dataset folder into the baseline you are running:

```bash
ln -s ./datasets ./mvsplat/
```

## Training

EcoSplat's **IGF** training finetunes a converged **ZPressor stage-1 checkpoint**. Download `mvsplat-re10k-zpressor-n200-256x256.ckpt` (see [Model Zoo](#model-zoo)) into `mvsplat/pretrained/`, or train your own with [ZPressor](https://github.com/ziplab/ZPressor).

The `re10k_igf` experiment adds the IGF path (merge head + `L_io`) on top of the ZPressor recipe, loading the stage-1 weights via `stage1_weights_path`:

```bash
cd mvsplat
./train.sh
# equivalently:
python -m src.main +experiment=re10k_igf
```

Key settings live in [`mvsplat/config/experiment/re10k_igf.yaml`](mvsplat/config/experiment/re10k_igf.yaml): `model.encoder.igf={}` activates IGF (defaults from [`IGFConfig`](ecosplat_wrapper/README.md#igfconfig)), `stage1_weights_path` points at the ZPressor checkpoint, and `optimizer.igf_lr_multiplier=10.0` trains the IGF head / rate-embed at a higher LR. IGF adds extra renders, so reduce `data_loader.train.batch_size` if memory is tight:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./train.sh data_loader.train.batch_size=1 wandb.mode=online
```

## Evaluation

Evaluate novel-view synthesis at a chosen **primitive budget** — the protect rate κ (`inference_rho`); lower → fewer rendered Gaussians. The bundled `mvsplat/test.sh` runs it (set `CKPT`):

```bash
cd mvsplat
CKPT=pretrained/ecosplat-zpressor-re10k.ckpt ./test.sh
```

Or invoke directly, choosing the eval index and κ:

```bash
cd mvsplat
python -m src.main +experiment=re10k_igf mode=test \
    checkpointing.load=pretrained/ecosplat-zpressor-re10k.ckpt checkpointing.resume=false \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.index_path=assets/re10k_evaluation/evaluation_index_re10k_small_24views.json \
    +model.encoder.igf.inference_rho=0.7 \
    test.compute_scores=true test.save_image=true test.dec_chunk_size=8 \
    test.output_path=outputs/test/igf
```

Sweep `inference_rho` (e.g. `0.7 → 0.1 → 0.02`) to trade primitive count against quality; bundled eval indices cover 16 / 20 / 24 context views (`mvsplat/assets/re10k_evaluation/`). The published inference release exposes a **related** knob, `model.encoder.primitive_ratio` (its `encoder_ecosplat` distributes the budget per view); this wrapper applies a uniform rate.

## Acknowledgements
This project applies EcoSplat to [ZPressor](https://github.com/ziplab/ZPressor), which is developed with [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat) and [DepthSplat](https://github.com/cvg/depthsplat). We thank the original authors for their excellent work.

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

Please also consider citing the ZPressor base:

```bibtex
@article{wang2025zpressor,
  title={ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS},
  author={Wang, Weijie and Chen, Donny Y and Zhang, Zeyu and Shi, Duochao and Liu, Akide and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2505.23734},
  year={2025}
}
```

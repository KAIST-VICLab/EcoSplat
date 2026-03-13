<p align="center">
  <h1 align="center">EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images</h1>
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
  <h3 align="center">CVPR 2026</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2512.18692">Paper</a> | <a href="https://kaist-viclab.github.io/ecosplat-site/">Project Page</a> | <a href="https://github.com/KAIST-VICLab/EcoSplat">Code</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/KAIST-VICLab/EcoSplat/blob/main/assets/teaser.png?raw=true" alt="EcoSplat Teaser" width="100%">
  </a>
</p>




## 📧 News
- **March 09, 2026:** Released inference code and pre-trained models.
- **Jan 03, 2026:** Initial repository created.

## 🚀 Code Release Plan
**The full code and pretrained models will be released soon.**

- ✅ Inference code
- ✅ Pretrained models
- ⬛ Training scripts
- ⬛ Dataset generation scripts

## 🛠️ Installation

Our code is developed using PyTorch 2.5.1, CUDA 12.4, and Python 3.11.


```bash
git clone https://github.com/KAIST-VICLab/EcoSplat.git
cd EcoSplat

conda create -y -n ecosplat python=3.11
conda activate ecosplat
bash setup.sh
```

## 📦 Model Zoo

Our pre-trained models are hosted on [Hugging Face 🤗](https://huggingface.co/ImJongminPark/EcoSplat).

We assume the downloaded weights are located in the `pretrained_weights` directory.


## 📂 Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## 💻 Running the Code
### Evaluation

To evaluate EcoSplat on RealEstate10K, run the following command. You can adjust the `primitive_ratio` as needed.

```bash
# RealEstate10K (enable test.align_pose=true if using evaluation-time pose alignment)
python -m src.main +experiment=ecosplat/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k_small_16views.json \
    checkpointing.load=./pretrained_weights/ecosplat-stage2-re10k.ckpt \
    model.encoder.primitive_ratio=<PRIMITIVE_RATIO> \ 
    test.save_image=true test.align_pose=true \ 
    test.output_path=<YOUR_OUTPUT_PATH>

```

## 🙏 Acknowledgements
This project is built upon these excellent repositories: [SPFSplat](https://github.com/ranrhuang/SPFSplat), [NoPoSplat](https://github.com/cvg/NoPoSplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). We thank the original authors for their excellent work.

## 🌱 Citation
```
@article{park2025ecosplat,
  title={EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images},
  author={Park, Jongmin and Bui, Minh-Quan Viet and Bello, Juan Luis Gonzalez and Moon, Jaeho and Oh, Jihyong and Kim, Munchurl},
  journal={arXiv preprint arXiv:2512.18692},
  year={2025}
}
```
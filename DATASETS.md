# Datasets

## Training datasets
For training, we mainly use [RealEstate10K](https://google.github.io/realestate10k/index.html), and [ACID](https://infinite-nature.github.io/) datasets. We provide the data processing scripts to convert the original datasets to pytorch chunk files which can be directly loaded with this codebase. 

Expected folder structure:

```
├── datasets
│   ├── re10k
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── acid
│   ├── ├── train
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
│   ├── ├── test
│   ├── ├── ├── 000000.torch
│   ├── ├── ├── ...
│   ├── ├── ├── index.json
```

By default, we assume the datasets are placed in `datasets/re10k`, and `datasets/acid`. Otherwise you will need to specify your dataset path with `dataset.DATASET_NAME.roots=[YOUR_DATASET_PATH]` in the running script.

We also provide instructions to convert additional datasets to the desired format.



### RealEstate10K and ACID

For experiments on RealEstate10K, we primarily follow [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat) to train and evaluate on 256x256 resolution.

Please refer to [here](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) for acquiring the processed 360p dataset (360x640 resolution).

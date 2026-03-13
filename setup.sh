#!/bin/bash
echo "Installing PyTorch..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "Installing requirements..."
pip install -r requirements.txt

echo ">> Building PyTorch3D..."
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@055ab3a2e3e611dff66fa82f632e62a315f3b5e7

echo "Installing custom CUDA extensions..."
pip install --no-build-isolation \
    git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git@b1cedf5cb565c676a1df3dc823da4d1d2cec3806 \
    git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe \
    git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 \
    git+https://github.com/nerfstudio-project/gsplat.git@6f378361f32ba4f1be86f65ae0db4948efb37dd5

echo "Installing utilities..."
pip install git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787
pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e

echo "Installation complete!"
"""Composable IGF stage-2 module.

Holds the cloned merge head(s), the cloned image embed (used as rate embed),
and the L_io loss. Host-agnostic: exposes raw modules + 3 primitive methods.
Host orchestrates merge-head invocation, rate_feat broadcast, and rendering.
"""

import copy
from typing import Union

import torch
from torch import nn

from .igf_config import IGFConfig
from .loss_importance_opacity import (
    LossImportanceOpacity,
    LossImportanceOpacityCfg,
    LossImportanceOpacityCfgWrapper,
)
from .mask import generate_importance_mask
from .plgc import plgc_protect_rate


class IGFModule(nn.Module):
    def __init__(
        self,
        heads_to_clone: Union[nn.Module, list],
        *,
        image_embed: nn.Module = None,
        igf_cfg: IGFConfig = None,
    ):
        super().__init__()
        self.cfg = igf_cfg or IGFConfig()

        if isinstance(heads_to_clone, nn.Module):
            heads_to_clone = [heads_to_clone]
        self.merge_heads = nn.ModuleList([copy.deepcopy(h) for h in heads_to_clone])

        # Match stage-2 SPFSplat: fresh 3->256 conv stack with default Kaiming-uniform init.
        # `image_embed` kwarg retained for backwards compat but unused.
        del image_embed
        self.rate_embed = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )

        self._loss = LossImportanceOpacity(
            LossImportanceOpacityCfgWrapper(
                importance_opacity=LossImportanceOpacityCfg(
                    weight=self.cfg.loss_weight,
                    io_weight=self.cfg.io_weight,
                    acc_weight=self.cfg.acc_weight,
                )
            )
        )

    def sample_rho(self, global_step: int) -> float:
        return plgc_protect_rate(
            global_step,
            k_min0=self.cfg.plgc_min0,
            floor=self.cfg.plgc_floor,
            k_max=self.cfg.plgc_max,
            decay=self.cfg.plgc_decay,
            decay_steps=self.cfg.plgc_decay_steps,
        )

    def get_rho(self, global_step: int, training: bool, override: float = None) -> float:
        """Training: PLGC-sampled. Inference: cfg.inference_rho (or `override`)."""
        if override is not None:
            return float(override)
        if training:
            return self.sample_rho(global_step)
        return float(self.cfg.inference_rho)

    def compute_distill(
        self,
        *,
        ori_gaussians,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        protect_rate: float,
        depth: torch.Tensor = None,
    ) -> dict:
        B, V, _, H, W = image.shape
        if hasattr(ori_gaussians, "means"):
            pts3d = ori_gaussians.means.view(B, V, H, W, 3)
            cov_feat = ori_gaussians.covariances.view(B, V, H, W, 3, 3)
        elif isinstance(ori_gaussians, dict) and "means_pix" in ori_gaussians:
            pts3d = ori_gaussians["means_pix"]
            cov_feat = ori_gaussians["cov_pix"]
        else:
            pts3d = ori_gaussians["pts3d"]
            cov_feat = ori_gaussians["cov_feat"]

        if depth is None:
            w2c = torch.inverse(extrinsics).view(B * V, 4, 4)
            pts_flat = pts3d.reshape(B * V, H * W, 3)
            cam = torch.einsum("bij,bnj->bni", w2c[:, :3, :3], pts_flat) + w2c[:, None, :3, 3]
            depth = cam[..., 2].view(B, V, H, W)

        return generate_importance_mask(
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            pts3d=pts3d,
            cov_feat=cov_feat,
            protect_rate=protect_rate,
        )

    def compute_loss(self, distill_infos: dict, output, ori_output) -> torch.Tensor:
        return self._loss(distill_infos, output, ori_output)

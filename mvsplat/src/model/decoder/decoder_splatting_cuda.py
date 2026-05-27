from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput
from ...global_cfg import get_cfg


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        render_alpha: bool = False,
    ) -> DecoderOutput:
        # gsplat returns (color, depth, alpha) in one call; alpha is free.
        # `render_alpha` kwarg kept for API stability but ignored.
        b, v, _, _ = extrinsics.shape
        # Random background during training (opt-in via train.random_bg): forces the
        # render's alpha->1 to match the opaque GT, suppressing coverage holes/floaters.
        # Fixed background at eval (self.training is False), so metrics stay comparable.
        _cfg = get_cfg()
        if self.training and _cfg is not None and _cfg.train.get("random_bg", False):
            background = torch.rand(
                (b * v, 3), device=extrinsics.device, dtype=self.background_color.dtype
            )
        else:
            background = repeat(self.background_color, "c -> (b v) c", b=b, v=v)
        color, _depth, alpha = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            background,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v),
            repeat(gaussians.scales, "b g xyz -> (b v) g xyz", v=v),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        alpha = rearrange(alpha, "(b v) h w -> b v h w", b=b, v=v)

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
            alpha,
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v),
            repeat(gaussians.scales, "b g xyz -> (b v) g xyz", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)

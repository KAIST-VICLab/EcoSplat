from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ...global_cfg import get_cfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


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
        b, v, _, _ = extrinsics.shape

        # Random background during training (opt-in via train.random_bg): forces the
        # render's alpha->1 to match the opaque GT, suppressing coverage holes/floaters.
        # Fixed background at eval (self.training is False), so metrics stay comparable.
        _cfg = get_cfg()
        use_random_bg = bool(
            self.training and _cfg is not None and _cfg.train.get("random_bg", False)
        )
        if use_random_bg:
            background = torch.rand(
                (b * v, 3), device=extrinsics.device, dtype=self.background_color.dtype
            )
        else:
            background = repeat(self.background_color, "c -> (b v) c", b=b, v=v)

        # Tile inputs from per-scene to per-(scene, view).
        ext = rearrange(extrinsics, "b v i j -> (b v) i j")
        intr = rearrange(intrinsics, "b v i j -> (b v) i j")
        near_v = rearrange(near, "b v -> (b v)")
        far_v = rearrange(far, "b v -> (b v)")
        means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
        covariances = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v)
        harmonics = repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v)
        opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)

        # Pass 1: photometric color render with the real SH (matches the paper's
        # diff_gaussian_rasterization recipe — no alpha output from the kernel).
        color = render_cuda(
            ext, intr, near_v, far_v, image_shape, background,
            means, covariances, harmonics, opacities,
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        # Pass 2 (only when needed for L_io / L_acc): extract alpha by re-rendering
        # with every Gaussian's color set to 1 and the background set to 0:
        #   C(x) = Σ_i c_i · α_i · T_i + T_final · bg
        #        = Σ_i 1   · α_i · T_i + 0
        #        = A(x).
        # use_sh=False bypasses SH evaluation entirely (cuda_splatting forwards the
        # d_sh=1 tensor straight to dgr's colors_precomp), so the alpha pass is
        # ~30-40% faster than a full color render. The 3 RGB channels all carry the
        # same alpha; take channel 0.
        alpha: Tensor | None = None
        if render_alpha:
            ones_colors = torch.ones(
                (b * v, harmonics.shape[1], 3, 1),
                device=harmonics.device, dtype=harmonics.dtype,
            )
            zero_bg = torch.zeros(
                (b * v, 3), device=extrinsics.device, dtype=self.background_color.dtype,
            )
            alpha_render = render_cuda(
                ext, intr, near_v, far_v, image_shape, zero_bg,
                means, covariances, ones_colors, opacities,
                use_sh=False,
            )
            alpha = rearrange(alpha_render, "(b v) c h w -> b v c h w", b=b, v=v)[:, :, 0, :, :]
            alpha = alpha.clamp(0.0, 1.0)

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
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)

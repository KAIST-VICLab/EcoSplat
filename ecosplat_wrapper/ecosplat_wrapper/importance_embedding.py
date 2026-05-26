import copy
from typing import Optional

import torch
from torch import Tensor, nn


def make_rate_embed(host: nn.Module, color_attr: str, zero_init: bool = True) -> nn.Module:
    src = getattr(host, color_attr)
    rate_mod = copy.deepcopy(src)
    if zero_init:
        for p in rate_mod.parameters():
            nn.init.zeros_(p)
    return rate_mod


class ImportanceEmbedding(nn.Module):
    def __init__(self, projector: nn.Module, in_ch: int = 3):
        super().__init__()
        self.projector = projector
        self.in_ch = in_ch

    @classmethod
    def from_color_projector(
        cls,
        host: nn.Module,
        color_attr: str,
        zero_init: bool = True,
        in_ch: int = 3,
    ) -> "ImportanceEmbedding":
        return cls(make_rate_embed(host, color_attr, zero_init=zero_init), in_ch=in_ch)

    def forward(
        self,
        rho: float | Tensor,
        ref_image: Tensor,
    ) -> Tensor:
        if ref_image.dim() != 5:
            raise ValueError(
                f"ref_image must be (B, V, C, H, W); got shape {tuple(ref_image.shape)}"
            )
        b, v, _, h, w = ref_image.shape
        if torch.is_tensor(rho):
            rho_t = rho.to(device=ref_image.device, dtype=ref_image.dtype)
        else:
            rho_t = torch.as_tensor(rho, device=ref_image.device, dtype=ref_image.dtype)
        while rho_t.dim() < 5:
            rho_t = rho_t.unsqueeze(-1)
        rho_t = rho_t.expand(b, v, 1, 1, 1)
        x = rho_t.expand(b, v, self.in_ch, h, w).reshape(b * v, self.in_ch, h, w)
        out = self.projector(x)
        return out


def inject_shallow_add(features: Tensor, R: Tensor) -> Tensor:
    return features + R


def assert_rate_matches_color(
    host: nn.Module,
    color_attr: str,
    rho_sample: float | Tensor,
    image_sample: Tensor,
) -> None:
    """Sanity check: rate embed output shape == color projector output shape."""
    color_mod = getattr(host, color_attr)
    b, v, c, h, w = image_sample.shape
    color_out = color_mod(image_sample.reshape(b * v, c, h, w))
    rate = ImportanceEmbedding.from_color_projector(
        host, color_attr, zero_init=False, in_ch=c
    )
    rate_out = rate(rho_sample, image_sample)
    if color_out.shape != rate_out.shape:
        raise AssertionError(
            f"Rate-embed output {tuple(rate_out.shape)} != color skip output {tuple(color_out.shape)}"
        )

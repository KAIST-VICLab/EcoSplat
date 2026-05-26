"""Importance mask + KLD-merged covariance generator.

Ported from the SPFSplat-based EcoSplat encoder (`encoder_spfsplat.py`).
Same arithmetic, no logic change. Output dict keys follow paper notation:
`importance_mask` (Ω_i), `kld_cov` (merged covariance), `valid_mask`, `pixel_pts`.
"""

from typing import Optional

import torch
from torch import Tensor

from ._utils import (
    bipartite_soft_matching_random2d,
    get_grad_map,
    get_normal_map,
    init_generator,
    normalized_view_plane_uv,
)


def _kld_cov(mu_merge: Tensor, mu: Tensor, cov: Tensor) -> Tensor:
    mu_diff = mu - mu_merge
    mu_cov = torch.matmul(mu_diff.unsqueeze(-1), mu_diff.unsqueeze(-2))
    return cov + mu_cov


def _world_to_camera(extrinsics_c2w: Tensor, pts_world: Tensor) -> Tensor:
    """extrinsics_c2w: (B*V, 4, 4). pts_world: (B*V, N, 3). Returns (B*V, N, 3)."""
    w2c = torch.inverse(extrinsics_c2w)
    return torch.einsum("bij, bnj -> bni", w2c[:, :3, :3], pts_world) + w2c[:, None, :3, 3]


def generate_importance_mask(
    image: Tensor,            # (B, V, 3, H, W)
    depth: Tensor,            # (B, V, H, W)
    intrinsics: Tensor,       # (B, V, 3, 3)
    extrinsics: Tensor,       # (B, V, 4, 4) camera-to-world
    pts3d: Tensor,            # (B, V, H, W, 3)
    cov_feat: Tensor,         # (B, V, H, W, 3, 3)
    protect_rate: float,
    *,
    patch: int = 4,
    rng_generator: Optional[torch.Generator] = None,
) -> dict[str, Tensor]:
    b, v, _, h, w = image.shape
    bv = b * v

    # --- Step 1: binary variation map (photometric + geometric gradients) ---
    normal_map = get_normal_map(depth.view(bv, h, w), intrinsics.view(bv, 3, 3)).view(b, v, h, w, 3)
    normal_grad = (
        get_grad_map(normal_map.view(bv, h, w, 3).permute(0, 3, 1, 2))
        .view(b, v, 1, h, w)
        .permute(0, 1, 3, 4, 2)
    )
    img_grad = (
        get_grad_map(image.view(bv, 3, h, w))
        .view(b, v, 1, h, w)
        .permute(0, 1, 3, 4, 2)
    )
    grad_map = torch.cat([normal_grad, img_grad], dim=-1).mean(-1).view(bv, h, w)

    sx, sy = patch, patch
    patch_grad_map = (
        grad_map.view(bv, h // sy, sy, w // sx, sx)
        .transpose(2, 3)
        .reshape(bv, h // sy, w // sx, sx * sy)
    )
    grad_thes = torch.quantile(patch_grad_map.sum(-1).flatten(1, 2), 1.0 - protect_rate, dim=1)
    merge_mask = torch.le(patch_grad_map.sum(-1), grad_thes[:, None, None])
    full_merge_mask = (
        merge_mask.unsqueeze(-1)
        .expand(-1, -1, -1, sx * sy)
        .view(bv, h // sy, w // sx, sy, sx)
        .transpose(2, 3)
        .reshape(bv, h, w)
    )

    # --- Step 2: bipartite-soft-matching merge over the low-variation set ---
    norm_uv = normalized_view_plane_uv(w, h, dtype=pts3d.dtype, device=pts3d.device)
    pts_flat = pts3d.view(b, v, h * w, 3)
    agg_features = torch.cat(
        [
            pts3d.flatten(0, 3).view(bv, h * w, -1),
            (norm_uv.flatten(0, 1))[None].expand(bv, -1, -1),
        ],
        dim=-1,
    )

    if rng_generator is None:
        rng_generator = init_generator(agg_features.device)

    m, _vis, u = bipartite_soft_matching_random2d(
        agg_features,
        w=w, h=h, sx=sx, sy=sy,
        r=int(agg_features.shape[1] * 0.5),
        p=int(agg_features.shape[1] * 0.2),
        generator=rng_generator,
        merge_mask=full_merge_mask,
    )

    merged_pts_all = m(pts_flat.view(bv, h * w, 3))
    unmerged_pts_all = u(merged_pts_all)
    n_tokens = merged_pts_all.shape[1]

    # --- Step 3: project merged 3D centers to image planes → importance mask Ω_i ---
    cam_pts = _world_to_camera(
        extrinsics.view(bv, 4, 4), merged_pts_all
    ).view(b, v, n_tokens, 3, 1)
    pixel_pts = torch.matmul(intrinsics[:, :, None], cam_pts).squeeze(-1)
    pixel_pts = pixel_pts / (pixel_pts[..., 2:] + 1e-8)

    valid_mask = (
        (pixel_pts[..., 0] >= 0)
        & (pixel_pts[..., 0] < 1)
        & (pixel_pts[..., 1] >= 0)
        & (pixel_pts[..., 1] < 1)
        & (cam_pts.squeeze(-1)[..., 2] > 0)
    )

    pixel_pts_norm = 2 * pixel_pts[..., :2].clone().flatten(0, 1) - 1.0

    pixel_pts[..., 0] = torch.clamp(pixel_pts[..., 0] * w, 0, w - 1)
    pixel_pts[..., 1] = torch.clamp(pixel_pts[..., 1] * h, 0, h - 1)
    pixel_pts = pixel_pts.long()

    flatten_index = pixel_pts[..., 1] * w + pixel_pts[..., 0]
    projection = torch.zeros(b, v, h * w, dtype=image.dtype, device=image.device)
    projection.scatter_(
        2,
        flatten_index,
        torch.ones_like(flatten_index, dtype=projection.dtype) * valid_mask.type(projection.dtype),
    )

    # --- Step 4: KLD-merged covariance (used by optional cov-distill loss) ---
    kld_cov_pix = _kld_cov(
        unmerged_pts_all,
        pts_flat.view(bv, h * w, 3),
        cov_feat.view(bv, h * w, 3, 3),
    )
    merged_cov = (
        0.5 * m(kld_cov_pix.flatten(-2, -1), mode="sum").view(b, v, n_tokens, 9)
        + 0.5 * m(kld_cov_pix.flatten(-2, -1)).view(b, v, n_tokens, 9)
    )

    return {
        "importance_mask": projection,                              # Ω_i, shape (B, V, H*W)
        "pixel_pts": pixel_pts_norm,                                # (B*V, n_tokens, 2) in [-1, 1]
        "valid_mask": valid_mask.view(bv, n_tokens),                # (B*V, n_tokens)
        "kld_cov": merged_cov.view(bv, n_tokens, 3, 3),             # (B*V, n_tokens, 3, 3)
    }

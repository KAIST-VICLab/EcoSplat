from math import isqrt
from typing import Literal

import torch
from einops import einsum, rearrange, repeat
from gsplat import rasterization
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points
from ..encoder.costvolume.conversions import depth_to_relative_disparity


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],  # unused, kept for sig stability
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> tuple[
    Float[Tensor, "batch 3 height width"],
    Float[Tensor, "batch height width"],
    Float[Tensor, "batch height width"],
]:
    """gsplat rasterization. Returns (color, depth, alpha).

    Mirrors the spfsplat_stage2 call signature: passes world-space (quats,
    scales) directly to gsplat. `gaussian_covariances` is ignored.
    """
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_means = gaussian_means * scale[:, None, None]
        gaussian_scales = gaussian_scales * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = gaussian_means.shape
    h, w = image_shape

    view_matrix = extrinsics.inverse()
    test_intr = intrinsics.clone()
    test_intr[:, 0] = intrinsics[:, 0] * w
    test_intr[:, 1] = intrinsics[:, 1] * h

    # mvsplat stores quaternions as (x, y, z, w); gsplat expects (w, x, y, z).
    gaussian_rotations_wxyz = gaussian_rotations[..., [3, 0, 1, 2]]

    all_images = []
    all_depths = []
    all_alphas = []
    for i in range(b):
        # gsplat `colors` layout:
        #   SH path (sh_degree set):   (N, K, 3) per-Gaussian SH coefficients.
        #   Precomp path (sh_degree=None): (N, channels) per-Gaussian RGB.
        if use_sh:
            colors_i = shs[i]
        else:
            colors_i = shs[i, :, 0, :]  # (N, 3)
        image, rendered_alpha, _ = rasterization(
            gaussian_means[i],
            gaussian_rotations_wxyz[i],
            gaussian_scales[i],
            gaussian_opacities[i],
            colors_i,
            view_matrix[i][None],
            test_intr[i][None],
            w,
            h,
            sh_degree=degree if use_sh else None,
            render_mode="RGB+D",
            packed=False,
            near_plane=1e-10,
            backgrounds=background_color[i].unsqueeze(0),
            radius_clip=0.0,  # match mvsplat baseline (diff_gaussian_rasterization has no radius cull)
            rasterize_mode="classic",
        )
        image_with_depth = image.squeeze(0).permute(2, 0, 1)
        image, rendered_depth = torch.split(image_with_depth, [3, 1], dim=0)
        rendered_alpha = rendered_alpha.squeeze(0).permute(2, 0, 1)
        all_images.append(image)
        all_depths.append(rendered_depth.squeeze(0))
        all_alphas.append(rendered_alpha.squeeze(0))
    return torch.stack(all_images), torch.stack(all_depths), torch.stack(all_alphas)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],  # unused; kept for sig stability
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    """gsplat orthographic rasterization (camera_model='ortho')."""
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["width"] = width
        dump["height"] = height
        dump["near"] = near
        dump["far"] = far

    view_matrix = extrinsics.inverse()

    # Build ortho K per batch: maps world units -> pixel coords.
    Ks = torch.zeros((b, 3, 3), dtype=extrinsics.dtype, device=extrinsics.device)
    Ks[:, 0, 0] = w / width
    Ks[:, 1, 1] = h / height
    Ks[:, 0, 2] = w * 0.5
    Ks[:, 1, 2] = h * 0.5
    Ks[:, 2, 2] = 1.0

    # mvsplat stores quaternions as (x, y, z, w); gsplat expects (w, x, y, z).
    gaussian_rotations_wxyz = gaussian_rotations[..., [3, 0, 1, 2]]

    all_images = []
    for i in range(b):
        colors_i = shs[i] if use_sh else shs[i, :, 0, :]
        image, _alpha, _ = rasterization(
            gaussian_means[i],
            gaussian_rotations_wxyz[i],
            gaussian_scales[i],
            gaussian_opacities[i],
            colors_i,
            view_matrix[i][None],
            Ks[i][None],
            w,
            h,
            sh_degree=degree if use_sh else None,
            render_mode="RGB",
            packed=False,
            near_plane=float(near[i].item()) if near[i].numel() == 1 else 1e-10,
            far_plane=float(far[i].item()) if far[i].numel() == 1 else 1e10,
            backgrounds=background_color[i].unsqueeze(0),
            rasterize_mode="classic",
            camera_model="ortho",
        )
        all_images.append(image.squeeze(0).permute(2, 0, 1))
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    gaussian_scales: Float[Tensor, "batch gaussian 3"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth-as-color; read it back from the color channel.
    b, _ = fake_color.shape
    color, _depth, _alpha = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        gaussian_rotations,
        gaussian_scales,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return color.mean(dim=1)

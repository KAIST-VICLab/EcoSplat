"""Compare diff_gaussian_rasterization vs gsplat on synthetic Gaussians.

Verifies the gsplat port produces output numerically close to the original
diff_gaussian_rasterization renderer that stage-1 ZPressor was trained with.

Run:
    python smoke_rasterizer_compare.py
"""
import torch
from einops import rearrange, repeat
from math import isqrt

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gsplat import rasterization


def get_projection_matrix(near, far, fov_x, fov_y):
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


def render_diff_gaussian(extr, intr, near, far, hw, bg, means, covs, shs, opa):
    """Verbatim original mvsplat render_cuda body (diff_gaussian_rasterization)."""
    # scale_invariant
    scale = 1 / near
    extr = extr.clone()
    extr[..., :3, 3] = extr[..., :3, 3] * scale[:, None]
    covs = covs * (scale[:, None, None, None] ** 2)
    means = means * scale[:, None, None]
    near = near * scale
    far = far * scale

    _, _, _, n = shs.shape
    degree = isqrt(n) - 1
    shs_b = rearrange(shs, "b g xyz n -> b g n xyz").contiguous()
    b, _, _ = extr.shape
    h, w = hw

    # mvsplat fov from intrinsics (normalized: fx_norm = K[0,0])
    fov_x = 2 * torch.atan(0.5 / intr[:, 0, 0])
    fov_y = 2 * torch.atan(0.5 / intr[:, 1, 1])
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    P = get_projection_matrix(near, far, fov_x, fov_y)
    P = rearrange(P, "b i j -> b j i")
    V = rearrange(extr.inverse(), "b i j -> b j i")
    full_P = V @ P

    out = []
    for i in range(b):
        s = GaussianRasterizationSettings(
            image_height=h, image_width=w,
            tanfovx=tan_fov_x[i].item(), tanfovy=tan_fov_y[i].item(),
            bg=bg[i], scale_modifier=1.0,
            viewmatrix=V[i], projmatrix=full_P[i],
            sh_degree=degree, campos=extr[i, :3, 3],
            prefiltered=False, debug=False,
        )
        r = GaussianRasterizer(s)
        row, col = torch.triu_indices(3, 3)
        means2D = torch.zeros_like(means[i], requires_grad=True)
        ret = r(
            means3D=means[i], means2D=means2D,
            shs=shs_b[i], colors_precomp=None,
            opacities=opa[i, ..., None],
            cov3D_precomp=covs[i, :, row, col],
        )
        img = ret[0]  # installed version returns (image, radii, depth, alpha)
        out.append(img)
    return torch.stack(out)  # (b, 3, h, w)


def render_gsplat(extr, intr, near, far, hw, bg, means, covs, shs, opa, rots_xyzw, scs):
    """Mirrors my current cuda_splatting.render_cuda."""
    scale = 1 / near
    extr = extr.clone()
    extr[..., :3, 3] = extr[..., :3, 3] * scale[:, None]
    means = means * scale[:, None, None]
    scs = scs * scale[:, None, None]
    near = near * scale
    far = far * scale

    _, _, _, n = shs.shape
    degree = isqrt(n) - 1
    shs_b = rearrange(shs, "b g xyz n -> b g n xyz").contiguous()
    b, _, _ = extr.shape
    h, w = hw

    V = extr.inverse()
    K = intr.clone()
    K[:, 0] = K[:, 0] * w
    K[:, 1] = K[:, 1] * h

    rots_wxyz = rots_xyzw[..., [3, 0, 1, 2]]

    out = []
    for i in range(b):
        img, _alpha, _ = rasterization(
            means[i], rots_wxyz[i], scs[i], opa[i], shs_b[i],
            V[i][None], K[i][None], w, h,
            sh_degree=degree, render_mode="RGB+D", packed=False,
            near_plane=1e-10, backgrounds=bg[i].unsqueeze(0),
            radius_clip=0.0, rasterize_mode="classic",
        )
        img3, _depth = torch.split(img.squeeze(0).permute(2, 0, 1), [3, 1], dim=0)
        out.append(img3)
    return torch.stack(out)


def make_world_quat_from_extrinsics_and_local(c2w_rot, local_xyzw):
    """world_rot = c2w_rot @ local_rotmat -> back to xyzw quat."""
    i, j, k, r = local_xyzw.unbind(-1)
    two_s = 2 / ((local_xyzw * local_xyzw).sum(-1) + 1e-8)
    m = torch.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], -1).reshape(*local_xyzw.shape[:-1], 3, 3)
    world = c2w_rot @ m
    # rotmat -> xyzw (case-largest-trace)
    trace = world[..., 0, 0] + world[..., 1, 1] + world[..., 2, 2]
    cases = torch.stack([
        1 + trace,
        1 + world[..., 0, 0] - world[..., 1, 1] - world[..., 2, 2],
        1 - world[..., 0, 0] + world[..., 1, 1] - world[..., 2, 2],
        1 - world[..., 0, 0] - world[..., 1, 1] + world[..., 2, 2],
    ], -1)
    best = cases.argmax(-1)
    eps = 1e-8
    r0 = torch.sqrt(cases[..., 0].clamp_min(eps)) * 0.5
    s0 = 0.25 / r0.clamp_min(eps)
    q0 = torch.stack([
        (world[..., 2, 1] - world[..., 1, 2]) * s0,
        (world[..., 0, 2] - world[..., 2, 0]) * s0,
        (world[..., 1, 0] - world[..., 0, 1]) * s0,
        r0,
    ], -1)
    i1 = torch.sqrt(cases[..., 1].clamp_min(eps)) * 0.5
    s1 = 0.25 / i1.clamp_min(eps)
    q1 = torch.stack([
        i1,
        (world[..., 0, 1] + world[..., 1, 0]) * s1,
        (world[..., 0, 2] + world[..., 2, 0]) * s1,
        (world[..., 2, 1] - world[..., 1, 2]) * s1,
    ], -1)
    j2 = torch.sqrt(cases[..., 2].clamp_min(eps)) * 0.5
    s2 = 0.25 / j2.clamp_min(eps)
    q2 = torch.stack([
        (world[..., 0, 1] + world[..., 1, 0]) * s2,
        j2,
        (world[..., 1, 2] + world[..., 2, 1]) * s2,
        (world[..., 0, 2] - world[..., 2, 0]) * s2,
    ], -1)
    k3 = torch.sqrt(cases[..., 3].clamp_min(eps)) * 0.5
    s3 = 0.25 / k3.clamp_min(eps)
    q3 = torch.stack([
        (world[..., 0, 2] + world[..., 2, 0]) * s3,
        (world[..., 1, 2] + world[..., 2, 1]) * s3,
        k3,
        (world[..., 1, 0] - world[..., 0, 1]) * s3,
    ], -1)
    q = torch.zeros_like(q0)
    q = torch.where((best == 0).unsqueeze(-1), q0, q)
    q = torch.where((best == 1).unsqueeze(-1), q1, q)
    q = torch.where((best == 2).unsqueeze(-1), q2, q)
    q = torch.where((best == 3).unsqueeze(-1), q3, q)
    return q


def main():
    torch.manual_seed(0)
    device = "cuda"
    B = 1
    H, W = 256, 256
    G = 4096

    # Synthetic stage-1-style Gaussians: random world means in [-1, 1].
    means = torch.rand(B, G, 3, device=device) * 2 - 1
    means[..., 2] += 3.0  # in front of camera

    # Local rotations xyzw (unit quats with small perturbation).
    local_q = torch.randn(B, G, 4, device=device)
    local_q = local_q / local_q.norm(dim=-1, keepdim=True)

    # Local scales in world units (small).
    scales = torch.rand(B, G, 3, device=device) * 0.02 + 0.005

    # Build covariance in world space: R_world @ S @ S^T @ R_world^T.
    # For comparison we treat extrinsics=identity, so world_rot = local_rot.
    c2w = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    world_q = make_world_quat_from_extrinsics_and_local(c2w[:, :3, :3].unsqueeze(1), local_q)
    # rotmat from xyzw
    i, j, k, r = world_q.unbind(-1)
    two_s = 2 / ((world_q * world_q).sum(-1) + 1e-8)
    R = torch.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], -1).reshape(B, G, 3, 3)
    S = torch.diag_embed(scales)
    covs = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

    # SH (degree 0, 1 coef) random colors.
    d_sh = 1
    shs = torch.rand(B, G, 3, d_sh, device=device)

    opa = torch.rand(B, G, device=device) * 0.8 + 0.1

    # Render setup.
    extr = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    intr = torch.zeros(B, 3, 3, device=device)
    intr[:, 0, 0] = 0.5  # fx_norm
    intr[:, 1, 1] = 0.5
    intr[:, 0, 2] = 0.5
    intr[:, 1, 2] = 0.5
    intr[:, 2, 2] = 1.0
    near = torch.full((B,), 1.0, device=device)
    far = torch.full((B,), 100.0, device=device)
    bg = torch.zeros(B, 3, device=device)

    # Run both.
    img_dgr = render_diff_gaussian(extr.clone(), intr, near.clone(), far.clone(),
                                    (H, W), bg, means.clone(), covs.clone(),
                                    shs.clone(), opa.clone())
    img_gs = render_gsplat(extr.clone(), intr, near.clone(), far.clone(),
                            (H, W), bg, means.clone(), covs.clone(),
                            shs.clone(), opa.clone(), world_q.clone(), scales.clone())

    print(f"diff_gaussian_rasterization: {tuple(img_dgr.shape)}  range [{img_dgr.min():.4f}, {img_dgr.max():.4f}]")
    print(f"gsplat:                       {tuple(img_gs.shape)}  range [{img_gs.min():.4f}, {img_gs.max():.4f}]")

    diff = (img_dgr - img_gs).abs()
    print()
    print(f"max abs diff:  {diff.max().item():.6e}")
    print(f"mean abs diff: {diff.mean().item():.6e}")
    print(f"median abs diff: {diff.median().item():.6e}")
    print(f"# pixels diff>1e-3: {(diff > 1e-3).float().mean().item() * 100:.2f}% of total")
    print(f"# pixels diff>1e-2: {(diff > 1e-2).float().mean().item() * 100:.2f}% of total")

    psnr_self = -10 * (diff.pow(2).mean()).log10()
    print(f"effective PSNR vs gsplat:  {psnr_self.item():.2f} dB  (higher = closer)")


if __name__ == "__main__":
    main()

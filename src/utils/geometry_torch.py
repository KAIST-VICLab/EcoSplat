from typing import *
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

def view_plane_uv(width: int, height: int, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    u = torch.linspace(0, width-1, width, dtype=dtype, device=device)
    v = torch.linspace(0, height-1, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def covariance_to_scale_rotation(covariance: torch.Tensor):
    """
    Computes scale and rotation (as a quaternion) from a 3x3 covariance matrix.
    This version is vectorized to avoid loops and .item() calls.

    Args:
        covariance: A tensor of shape (..., 3, 3) representing covariance matrices.

    Returns:
        A tuple of (scales, rotations_quat) where:
        - scales: A tensor of shape (..., 3)
        - rotations: A tensor of shape (..., 4) representing quaternions.
    """
    # Eigendecomposition for symmetric matrices
    # b, v, n, _ = covariance.shape
    # covariance = covariance.view(b, v, n, 3, 3)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    # Ensure eigenvalues are non-negative for sqrt
    eigenvalues = torch.clamp(eigenvalues, min=0)

    # Scales are the square root of the eigenvalues
    scales = torch.sqrt(eigenvalues)

    # The eigenvectors form the rotation matrix
    rotation_matrix = eigenvectors

    # --- [핵심 수정] for 루프와 .item() 제거 ---
    # Check determinants for all matrices in the batch at once
    det = torch.linalg.det(rotation_matrix)
    neg_det_mask = det < 0 # Shape: (...,)

    # If any determinants are negative, fix them
    if torch.any(neg_det_mask):
        # Find the index of the smallest eigenvalue for each matrix
        min_eig_idx = torch.argmin(eigenvalues, dim=-1)

        # Create a multiplier tensor, default is 1.0
        multiplier = torch.ones_like(rotation_matrix)

        # For matrices with negative determinants, set the multiplier for the
        # column corresponding to the smallest eigenvector to -1.0
        # This uses advanced indexing to target only the necessary elements.
        
        # Get batch and num_gaussians dimensions for indexing
        b_dims = neg_det_mask.shape
        # Create indices for all elements that need flipping
        flip_indices = neg_det_mask.nonzero(as_tuple=True)

        # Select the min_eig_idx only for those that need flipping
        min_eig_idx_to_flip = min_eig_idx[flip_indices]

        # Apply the flip
        if len(b_dims) == 2: # Batch, Num_Gaussians
            multiplier[flip_indices[0], flip_indices[1], :, min_eig_idx_to_flip] = -1.0
        elif len(b_dims) == 1: # Num_Gaussians only
            multiplier[flip_indices[0], :, min_eig_idx_to_flip] = -1.0
        else: # Handle other shapes if necessary, this is a simplified example
            # Fallback to a loop for > 2D batch shapes if needed, but the principle is the same.
            # This logic should cover most cases like (B, N)
            pass

        # Apply the flip by element-wise multiplication
        rotation_matrix = rotation_matrix * multiplier
    # ----------------------------------------------------

    # Convert the proper rotation matrix to a quaternion.
    # Using a robust library function is recommended in production.
    # from pytorch3d.transforms import matrix_to_quaternion
    # rotations_quat = matrix_to_quaternion(rotation_matrix)
    
    # Manual implementation for demonstration:
    q = torch.empty(rotation_matrix.shape[:-2] + (4,), device=rotation_matrix.device, dtype=rotation_matrix.dtype)
    t = rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]
    
    q[..., 0] = 0.5 * torch.sqrt(torch.clamp(1.0 + t, min=0)) # w
    q[..., 1] = 0.5 * torch.sign(rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]) * torch.sqrt(torch.clamp(1.0 + rotation_matrix[..., 0, 0] - rotation_matrix[..., 1, 1] - rotation_matrix[..., 2, 2], min=0)) # x
    q[..., 2] = 0.5 * torch.sign(rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]) * torch.sqrt(torch.clamp(1.0 - rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] - rotation_matrix[..., 2, 2], min=0)) # y
    q[..., 3] = 0.5 * torch.sign(rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]) * torch.sqrt(torch.clamp(1.0 - rotation_matrix[..., 0, 0] - rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2], min=0)) # z

    # Normalize the quaternion
    rotations_quat = torch.nn.functional.normalize(q, dim=-1)

    return scales, rotations_quat

def visualization(grad_map: torch.Tensor, type: str):
    
    for v in range(grad_map.shape[0]):
        grad_v = grad_map[v]
        
        min_val = torch.min(grad_v)
        max_val = torch.max(grad_v)
        
        if max_val - min_val:
            normalized_grad = (grad_v - min_val) / (max_val - min_val)

        unit8_grad = (normalized_grad * 255).to(torch.uint8)
        numpy_grad = unit8_grad.cpu().numpy()
        
        image = Image.fromarray(numpy_grad, mode='L')
        image.save(f"{type}_{v}.png")


def _cam2world_to_world2cam(Tcw):  # cam2world -> world2cam (빠르고 안전한 역)
    R = Tcw[..., :3, :3]
    t = Tcw[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    W2C = torch.zeros_like(Tcw)
    W2C[..., :3, :3] = Rt
    W2C[..., :3, 3:4] = -Rt @ t
    W2C[..., 3, 3] = 1.0
    return W2C

def _project_to_view(Xw, Tcw_view, K_view, H, W):
    """Xw:(b,n,3), Tcw_view:(b,4,4) cam2world, K_view:(b,3,3) -> grid:(b,n,1,2), z_cam:(b,n,1), inb:(b,n,1)"""
    W2C = _cam2world_to_world2cam(Tcw_view)         # world->cam
    Xw_h = torch.cat([Xw, torch.ones_like(Xw[..., :1])], dim=-1)  # (b,n,4)
    Xc_h = (W2C @ Xw_h.transpose(1,2)).transpose(1,2)            # (b,n,4)
    Xc   = Xc_h[..., :3]                                         # (b,n,3)
    z    = Xc[..., 2:3]                                          # (b,n,1)

    x = (K_view @ Xc.transpose(1,2)).transpose(1,2)              # (b,n,3)
    uv = x[..., :2] / (x[..., 2:3] + 1e-8)
    u, v = uv[..., :1], uv[..., 1:2]
    grid_x = 2.0 * (u / (W-1)) - 1.0
    grid_y = 2.0 * (v / (H-1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)                 # (b,n,1,2)
    inb  = (grid_x>=-1)&(grid_x<=1)&(grid_y>=-1)&(grid_y<=1)     # (b,n,1)
    return grid, z, inb.float()

def _ensure_pixel_K(K_view: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """
    Ensure intrinsics K are in pixel units (fx, fy, cx, cy in pixels).
    If they look normalized (~0..1), de-normalize by (W,H).
    K_view: (b,3,3)
    """
    Kp = K_view.clone()
    fx = Kp[..., 0, 0]
    fy = Kp[..., 1, 1]
    needs_denorm = (fx.abs() < 4) | (fy.abs() < 4)  # heuristic
    if needs_denorm.any():
        Kp[..., 0, :] = Kp[..., 0, :] * W
        Kp[..., 1, :] = Kp[..., 1, :] * H
    return Kp

def _estimate_affine_per_view_to_ref(
    pts_all_bvr13: torch.Tensor,          # (b, v, r, 1, 3)
    extrinsics_bv44: torch.Tensor,        # (b, v, 4, 4) cam2world
    intrinsics_bv33: torch.Tensor,        # (b, v, 3, 3) (pixel K or normalized K)
    depths_per_view_bvhw: torch.Tensor,   # (b, v, H, W) depth z-buffer per view
    H: int,
    W: int,
    ref_idx: int = 0,
    min_samples: int = 256,
    tau_vis: float = 0.001,
    clamp_alpha: tuple[float, float] = (0.9, 1.1),
    clamp_beta: tuple[float, float]  = (-0.02, 0.02),
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate per-view affine depth alignment (alpha, beta) w.r.t. a reference view.
    Uses only overlapping (co-visible) regions between view vi and ref.

    Returns:
      alphas: (b, v, 1)
      betas : (b, v, 1)
    Notes:
      - For the reference view (vi == ref_idx), we force alpha=1, beta=0.
      - Requires an external `_project_to_view(Xw, Tcw_view, K_view, H, W)` that:
          inputs: Xw (b,n,3), Tcw_view (b,4,4 cam2world), K_view (b,3,3 pixels)
          outputs: grid (b,n,1,2 in [-1,1]), z_cam (b,n,1), inb (b,n,1)
    """
    b, v, r, _, _ = pts_all_bvr13.shape
    device = pts_all_bvr13.device
    dtype  = pts_all_bvr13.dtype

    Xw = pts_all_bvr13.squeeze(-2)  # (b, v, r, 3)

    # Ensure intrinsics are in pixel units
    K_ref   = _ensure_pixel_K(intrinsics_bv33[:, ref_idx], W, H)  # (b,3,3)
    Tcw_ref = extrinsics_bv44[:, ref_idx]                         # (b,4,4)
    zbuf_ref= depths_per_view_bvhw[:, ref_idx:ref_idx+1]          # (b,1,H,W)

    alphas = torch.ones(b, v, 1, device=device, dtype=dtype)
    betas  = torch.zeros(b, v, 1, device=device, dtype=dtype)

    for vi in range(v):
        if vi == ref_idx:
            # Reference view defines the scale/offset axis
            alphas[:, vi, 0] = 1.0
            betas [:, vi, 0] = 0.0
            continue

        # vi intrinsics in pixels
        K_v   = _ensure_pixel_K(intrinsics_bv33[:, vi], W, H)     # (b,3,3)
        Tcw_v = extrinsics_bv44[:, vi]                            # (b,4,4)

        # 1) Project vi's world points to ref -> z_hat_ref & grid_ref
        grid_ref, zhat_ref, inb_ref = _project_to_view(Xw[:, vi], Tcw_ref, K_ref, H, W)  # (b,r,1,2),(b,r,1),(b,r,1)
        zbuf_ref_samp = F.grid_sample(
            zbuf_ref, grid_ref, mode="bilinear", align_corners=True
        ).squeeze(1).squeeze(-1)  # (b,r)
        vis_ref = (zhat_ref.squeeze(-1) <= zbuf_ref_samp + tau_vis)  # (b,r)

        # 2) Project vi's world points to itself -> z_hat_vi & grid_vi (self-consistency)
        grid_vi, zhat_vi, inb_vi = _project_to_view(Xw[:, vi], Tcw_v, K_v, H, W)  # (b,r,1,2),(b,r,1),(b,r,1)
        zbuf_vi_samp = F.grid_sample(
            depths_per_view_bvhw[:, vi:vi+1], grid_vi, mode="bilinear", align_corners=True
        ).squeeze(1).squeeze(-1)  # (b,r)
        vis_vi = (zhat_vi.squeeze(-1) <= zbuf_vi_samp + tau_vis)      # (b,r)

        # 3) Overlap/co-visible mask: in-bounds & visible in both ref and vi
        overlap = (inb_ref.squeeze(-1) > 0.5) & (vis_ref > 0.5) \
                & (inb_vi.squeeze(-1)  > 0.5) & (vis_vi  > 0.5)        # (b,r)

        # 4) Weighted least-squares (here simple LS) on overlap: alpha*z_vi - beta ≈ zbuf_ref
        z_v = zhat_vi.squeeze(-1)  # (b, r)  # vi 카메라 기준 z

        for bi in range(b):
            # (a) inlier 선택: overlap ∧ finite ∧ z>0
            sel = overlap[bi] \
                & torch.isfinite(z_v[bi]) & torch.isfinite(zbuf_ref_samp[bi]) \
                & (z_v[bi] > 0)

            n = int(sel.sum().item())
            if n < min_samples:
                # 샘플이 너무 적으면 기본값 유지
                continue

            zv = z_v[bi][sel]            # (n,)
            zr = zbuf_ref_samp[bi][sel]  # (n,)

            # (b) 수치 안정화를 위한 중심화/표준화
            mu  = zv.mean()
            std = zv.std().clamp_min(1e-6)
            zv_n = (zv - mu) / std

            # (c) 허버 IRLS + 리지 정칙화
            #     정규화 공간에서 zr ≈ a_n * zv_n + b_n
            #     복원: α = a_n / std, β = b_n - α*mu
            A    = torch.stack([zv_n, torch.ones_like(zv_n)], dim=1).to(torch.float64)  # (n,2)
            bvec = zr.to(torch.float64)

            # 초기 가중치(모두 1)
            w = torch.ones_like(bvec, dtype=torch.float64)

            # 리지 강도: α는 약간, β는 더 약하게 묶기 (폭주 방지)
            lam_a, lam_b = 1e-3, 1e-4

            # 허버 임계값(상대형 추천): 깊이가 큰 곳에서 보간/수치오차가 커지므로 비례 허용
            delta = (0.02 * bvec.median().abs().clamp_min(1e-3)).to(torch.float64)

            x = None
            for _ in range(3):  # 2~3회면 충분
                # (가중) 정상방정식 풀기
                # (A^T W A + λI) x = A^T W b
                AtWA = A.t().mm((w[:, None] * A))
                Reg  = torch.diag(torch.tensor([lam_a, lam_b], dtype=torch.float64, device=A.device))
                AtWb = A.t().mv(w * bvec)
                x    = torch.linalg.solve(AtWA + Reg, AtWb)  # (2,)
                # 잔차와 허버 가중치 갱신
                pred = A.mv(x)
                res  = (bvec - pred).abs()
                w    = torch.where(res <= delta, torch.ones_like(res), (delta / (res + 1e-12)))

            a_n, b_n = x[0], x[1]                          # 정규화 공간 계수
            a  = (a_n / std.to(torch.float64))             # α 복원
            b0 = (b_n - a * mu.to(torch.float64))          # β 복원

            # (d) 안정화: α>0, 클램프
            a  = torch.clamp(a, clamp_alpha[0], clamp_alpha[1])
            b0 = torch.clamp(b0, clamp_beta [0],  clamp_beta [1])

            # (e) 기록
            alphas[bi, vi, 0] = a.to(dtype)
            betas [bi, vi, 0] = b0.to(dtype)

    return alphas, betas

def rgb_to_grayscale(image):
    """
    (3, H, W) 또는 (H, W, 3) RGB 이미지를 그레이스케일로 변환
    """
    if image.ndimension() == 3 and image.shape[-1] == 3:  # (H, W, 3)인 경우
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
    elif image.ndimension() == 3 and image.shape[0] == 3:  # (3, H, W)인 경우
        r, g, b = image[0], image[1], image[2]
    else:
        raise ValueError("Input image must be (H, W, 3) or (3, H, W)")

    grayscale = 0.299 * r + 0.587 * g + 0.114 * b  # 표준 Y 변환식
    return grayscale

def compute_frequency_high_freq_score(image):
    """
    RGB 이미지를 입력받아 고주파 성분 비율로 블러 특성 추출
    Args:
        image (torch.Tensor): (3, H, W) 또는 (H, W, 3) 형태 RGB 이미지
    Returns:
        torch.Tensor: 고주파 성분 비율 (Blur Feature)
    """
    gray_image = rgb_to_grayscale(image)

    f = torch.fft.fft2(gray_image)
    fshift = torch.fft.fftshift(f)
    magnitude_spectrum = torch.abs(fshift)

    h, w = magnitude_spectrum.shape
    center_size = 60  # 중앙 저주파 영역 크기

    center = (slice(h//2 - center_size//2, h//2 + center_size//2),
              slice(w//2 - center_size//2, w//2 + center_size//2))

    total_energy = magnitude_spectrum.sum()
    low_freq_energy = magnitude_spectrum[center].sum()
    high_freq_energy = total_energy - low_freq_energy

    high_freq_ratio = high_freq_energy / total_energy
    return high_freq_ratio
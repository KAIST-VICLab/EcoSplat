import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def matrix_to_quaternion(
    matrix: Float[Tensor, "*batch 3 3"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 4"]:
    """Inverse of `quaternion_to_matrix`. Returns xyzw quaternion."""
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cases = torch.stack(
        [
            1 + trace,
            1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ],
        dim=-1,
    )
    # Pick largest-magnitude case for numerical stability.
    best = cases.argmax(dim=-1)
    q = torch.zeros(*m.shape[:-2], 4, dtype=m.dtype, device=m.device)

    # case 0: r largest
    mask0 = best == 0
    r0 = torch.sqrt(cases[..., 0].clamp_min(eps)) * 0.5
    s0 = 0.25 / r0.clamp_min(eps)
    q0 = torch.stack(
        [
            (m[..., 2, 1] - m[..., 1, 2]) * s0,
            (m[..., 0, 2] - m[..., 2, 0]) * s0,
            (m[..., 1, 0] - m[..., 0, 1]) * s0,
            r0,
        ],
        dim=-1,
    )

    # case 1: i largest
    i1 = torch.sqrt(cases[..., 1].clamp_min(eps)) * 0.5
    s1 = 0.25 / i1.clamp_min(eps)
    q1 = torch.stack(
        [
            i1,
            (m[..., 0, 1] + m[..., 1, 0]) * s1,
            (m[..., 0, 2] + m[..., 2, 0]) * s1,
            (m[..., 2, 1] - m[..., 1, 2]) * s1,
        ],
        dim=-1,
    )

    # case 2: j largest
    j2 = torch.sqrt(cases[..., 2].clamp_min(eps)) * 0.5
    s2 = 0.25 / j2.clamp_min(eps)
    q2 = torch.stack(
        [
            (m[..., 0, 1] + m[..., 1, 0]) * s2,
            j2,
            (m[..., 1, 2] + m[..., 2, 1]) * s2,
            (m[..., 0, 2] - m[..., 2, 0]) * s2,
        ],
        dim=-1,
    )

    # case 3: k largest
    k3 = torch.sqrt(cases[..., 3].clamp_min(eps)) * 0.5
    s3 = 0.25 / k3.clamp_min(eps)
    q3 = torch.stack(
        [
            (m[..., 0, 2] + m[..., 2, 0]) * s3,
            (m[..., 1, 2] + m[..., 2, 1]) * s3,
            k3,
            (m[..., 1, 0] - m[..., 0, 1]) * s3,
        ],
        dim=-1,
    )

    q = torch.where(mask0.unsqueeze(-1), q0, q)
    q = torch.where((best == 1).unsqueeze(-1), q1, q)
    q = torch.where((best == 2).unsqueeze(-1), q2, q)
    q = torch.where((best == 3).unsqueeze(-1), q3, q)
    return q


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )

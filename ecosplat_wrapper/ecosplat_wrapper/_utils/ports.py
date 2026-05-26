"""Verbatim ports of SPFSplat helpers used by EcoSplat IGF mask generation.

Sources (from the SPFSplat-based EcoSplat encoder):
- get_grad_map, fast_similarity_chunks, init_generator, do_nothing,
  mps_gather_workaround, bipartite_soft_matching_random2d: encoder_spfsplat.py
  (added by the EcoSplat adaptation).
- get_normal_map: src/utils/point.py
- normalized_view_plane_uv, view_plane_uv: src/utils/geometry_torch.py
"""

from typing import Callable, Tuple

import torch
from torch import nn


def get_grad_map(input):
    pad_f_right = nn.ReplicationPad2d((0, 1, 0, 0))
    pad_f_bottom = nn.ReplicationPad2d((0, 0, 0, 1))

    pad_input_right = pad_f_right(input)
    pad_input_bottom = pad_f_bottom(input)
    grad_x = pad_input_right[:, :, :, 1:] - pad_input_right[:, :, :, :-1]
    grad_y = pad_input_bottom[:, :, 1:, :] - pad_input_bottom[:, :, :-1, :]

    grad_norm = torch.stack([grad_x.norm(dim=1), grad_y.norm(dim=1)], dim=1).norm(dim=1)
    return grad_norm


def get_normal_map(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    B, H, W = depth_map.shape
    assert intrinsic.shape == (B, 3, 3), "Intrinsic matrix must be Bx3x3"
    assert (intrinsic[:, 0, 1] == 0).all() and (intrinsic[:, 1, 0] == 0).all(), "Intrinsic matrix must have zero skew"

    fu = intrinsic[:, 0, 0] * W
    fv = intrinsic[:, 1, 1] * H
    cu = intrinsic[:, 0, 2] * W
    cv = intrinsic[:, 1, 2] * H

    u = torch.arange(W, device=depth_map.device)[None, None, :].expand(B, H, W)
    v = torch.arange(H, device=depth_map.device)[None, :, None].expand(B, H, W)

    x_cam = (u - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v - cv[:, None, None]) * depth_map / fv[:, None, None]
    z_cam = depth_map

    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(dtype=torch.float32)

    output = torch.zeros_like(cam_coords)
    dy = cam_coords[:, 2:, 1:-1] - cam_coords[:, :-2, 1:-1]
    dx = cam_coords[:, 1:-1, 2:] - cam_coords[:, 1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[:, 1:-1, 1:-1, :] = normal_map

    return output


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
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
    u = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    v = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


@torch.jit.script
def fast_similarity_chunks(a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, num_src, C = a.shape
    original_dtype = a.dtype
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]
        scores_chunk = (a_chunk[:, :, None] - b_bf16[:, None, :]).norm(dim=-1, p=2)
        chunk_max_bf16, chunk_idx = torch.min(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def init_generator(device: torch.device, fallback: torch.Generator = None):
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1),
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(
    metric: torch.Tensor,
    w: int, h: int, sx: int, sy: int, r: int, p: int,
    no_rand: bool = False,
    generator: torch.Generator = None,
    merge_mask: torch.Tensor = None,
    pixel_metric: torch.Tensor = None,
) -> Tuple[Callable, Callable, Callable]:
    """Verbatim from encoder_spfsplat.py:162-365. Returns (merge, visualize, unmerge)."""
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)

        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))

        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        idx_buffer_view = (idx_buffer_view[None].repeat(B, 1, 1) - 1) * merge_mask.type(torch.int64)

        if (hsy * sy) < h or (wsx * sx) < w:
            raise NotImplementedError
        else:
            idx_buffer = idx_buffer_view

        idx_buffer = idx_buffer.reshape(B, -1, 1)

        rand_idx = idx_buffer.argsort(dim=1)
        num_dst = (idx_buffer == -2).sum(-1).sum(-1)[0]
        num_src = (idx_buffer == -1).sum(-1).sum(-1)[0]
        num_protect = (idx_buffer == 0).sum(-1).sum(-1)[0]

        del idx_buffer, idx_buffer_view

        a_idx = rand_idx[:, num_dst:num_dst + num_src, :]
        b_idx = rand_idx[:, :num_dst, :]
        c_idx = rand_idx[:, num_dst + num_src:, :]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            protected = gather(x, dim=1, index=c_idx.expand(B, num_protect, C))
            return src, dst, protected, a_idx, b_idx, c_idx

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b, protected, _, _, _ = split(metric)
        r = a.shape[1]

        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

        node_max, node_idx = fast_similarity_chunks(a, b, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst, protected, _, _, _ = split(x)
        n, t1, c = src.shape
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst, protected], dim=1)

    def visualize(x: torch.Tensor) -> torch.Tensor:
        src, dst, protected, a_idx_, b_idx_, c_idx_ = split(x)
        n, t1, c = src.shape
        n, t2, c = dst.shape
        chunk_size_ = 100000
        vis_list = []
        for i, chunk_i in enumerate(range(0, t2, chunk_size_)):
            if i % 10 != 0:
                continue
            curr_x = x.clone()
            chunk_dst_i = (torch.arange(t2)[None] % chunk_size_ + chunk_size_).int()
            dst_i = torch.zeros_like(chunk_dst_i)
            dst_i[:, chunk_i:chunk_i + chunk_size_] = chunk_dst_i[:, chunk_i:chunk_i + chunk_size_]
            r_value = (dst_i * 123457) % 256
            g_value = (dst_i * 234567) % 256
            b_value = (dst_i * 345678) % 256
            dst_color = torch.clamp(torch.stack([r_value, g_value, b_value], dim=-1) / 255.0, 0.0, 1.0).to(b_idx_.device)
            curr_x.scatter_(dim=-2, index=b_idx_.expand(-1, -1, 3), src=dst_color.expand(n, -1, -1))
            src_color = torch.zeros_like(src)
            src_color_from_dst = gather(dst_color.expand(n, -1, -1), dim=-2, index=dst_idx.expand(-1, -1, 3))
            src_color = src_color.scatter_reduce(-2, src_idx.expand(n, r, c), src_color_from_dst, reduce='mean')
            curr_x.scatter_(dim=-2, index=a_idx_.expand(-1, -1, 3), src=src_color.expand(n, -1, -1))
            vis_list.append(curr_x)
        return vis_list

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst, protected_ = x[..., :unm_len, :], x[..., unm_len:unm_len + num_dst, :], x[..., unm_len + num_dst:, :]
        assert protected_.shape[1] == num_protect
        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=c_idx.expand(B, num_protect, c), src=protected_)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)
        return out

    return merge, visualize, unmerge

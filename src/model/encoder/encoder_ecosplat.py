from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Callable
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import math
import random
from collections import OrderedDict

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory, camera_head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim, normalize_image
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...misc.cam_utils import camera_normalization, convert_pose_to_4x4, depth_projector
from ...geometry.camera_emb import get_plucker_embedding
from .heads.pose_head import PoseHeadCfg
from ...misc.intrinsics_utils import estimate_intrinsics
from src.utils.point import get_normal_map
from src.utils.geometry_torch import view_plane_uv, normalized_view_plane_uv, compute_frequency_high_freq_score
import cv2
import numpy as np
from src.model.encoder.common.gaussians import build_covariance

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderEcoSplatCfg:
    name: Literal["ecosplat"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pose_head: PoseHeadCfg

    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pose_make_baseline_1: bool = True
    pose_make_relative: bool = True
    pose_head_type: str = 'mlp'
    estimating_focal: bool = False
    estimating_pose: bool = True

    primitive_ratio: float = 1.0

def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat

def get_grad_map(input):
    pad_f_right = torch.nn.ReplicationPad2d((0, 1, 0, 0))
    pad_f_bottom = torch.nn.ReplicationPad2d((0, 0, 0, 1))

    pad_input_right = pad_f_right(input)
    pad_input_bottom = pad_f_bottom(input)
    grad_x = pad_input_right[:, :, :, 1:] - pad_input_right[:, :, :, :-1]
    grad_y = pad_input_bottom[:, :, 1:, :] - pad_input_bottom[:, :, :-1, :]

    grad_norm = torch.stack([grad_x.norm(dim=1), grad_y.norm(dim=1)],dim=1).norm(dim=1)
    return grad_norm

@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = (a_chunk[:,:,None]-b_bf16[:,None,:]).norm(dim=-1, p=2)
        chunk_max_bf16, chunk_idx = torch.min(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int, p:int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None, merge_mask: torch.Tensor = None, pixel_metric: torch.Tensor = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)
        idx_buffer_view = (idx_buffer_view[None].repeat(B,1,1) - 1) * merge_mask.type(torch.int64)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            raise NotImplementedError
        else:
            idx_buffer = idx_buffer_view

        idx_buffer = idx_buffer.reshape(B, -1, 1)

        # We set dst tokens to be -2 and src to be -1 and protect to be 0, so an argsort gives us dst|src|pro indices
        rand_idx = idx_buffer.argsort(dim=1)
        num_dst = (idx_buffer==-2).sum(-1).sum(-1)[0]
        num_src = (idx_buffer==-1).sum(-1).sum(-1)[0]
        num_protect = (idx_buffer==0).sum(-1).sum(-1)[0]   # same for all batch

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        a_idx = rand_idx[:, num_dst:num_dst+num_src, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst
        c_idx = rand_idx[:, num_dst+num_src:, :] # protect

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            protected = gather(x, dim=1, index=c_idx.expand(B, num_protect, C))
            # return src, dst, protected
            return src, dst, protected, a_idx, b_idx, c_idx

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b, protected, _, _, _ = split(metric)
        r = a.shape[1]

        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

        node_max, node_idx = fast_similarity_chunks(a, b, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst, protected,_,_,_ = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst, protected], dim=1)

    def visualize(x: torch.Tensor) -> torch.Tensor:
        # input x channel = 3 init with all zereos
        src, dst, protected, a_idx, b_idx, c_idx = split(x)

        n, t1, c = src.shape
        n, t2, c = dst.shape

        chunk_size = 100000
        vis_list = []

        for i, chunk_i in enumerate(range(0, t2, chunk_size)):
            if i % 10 != 0:
                continue
            curr_x = x.clone()
            chunk_dst_i = (torch.arange(t2)[None] % chunk_size + chunk_size).int()
            dst_i = torch.zeros_like(chunk_dst_i)

            dst_i[:,chunk_i:chunk_i+chunk_size] = chunk_dst_i[:,chunk_i:chunk_i+chunk_size]

            r_value = (dst_i * 123457) % 256
            g_value = (dst_i * 234567) % 256
            b_value = (dst_i * 345678) % 256
            dst_color = torch.clamp(torch.stack([r_value,g_value,b_value], dim=-1) / 255.0, 0.0, 1.0).to(b_idx.device)
            curr_x.scatter_(dim=-2, index=b_idx.expand(-1, -1, 3), src=dst_color.expand(n,-1,-1))

            src_color = torch.zeros_like(src)
            src_color_from_dst = gather(dst_color.expand(n,-1,-1), dim=-2, index=dst_idx.expand(-1, -1, 3))

            # unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src_color = src_color.scatter_reduce(-2, src_idx.expand(n, r, c), src_color_from_dst, reduce='mean')
            curr_x.scatter_(dim=-2, index=a_idx.expand(-1, -1, 3), src=src_color.expand(n,-1,-1))
            vis_list.append(curr_x)
        return vis_list

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst, protected = x[..., :unm_len, :], x[..., unm_len:unm_len+num_dst, :], x[..., unm_len+num_dst:, :]
        assert protected.shape[1] == num_protect

        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=c_idx.expand(B, num_protect, c), src=protected)

        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, visualize, unmerge

import torch.nn.init as init
def _init_weights(module):
    # Initialize convolutional weights to zero
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            init.zeros_(m.weight)  # Initialize weights to 0
            if m.bias is not None:  # Initialize biases to 0 (optional)
                init.zeros_(m.bias)
                
class EncoderEcoSplat(Encoder[EncoderEcoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderEcoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type
       
        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                            depth_mode=('exp', -inf, inf), conf_mode=None,)
 
            
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        if self.cfg.estimating_pose:
            self.set_pose_head(cfg, cfg.pose_head_type)
            
        stage1_statedict = torch.load(cfg.pretrained_weights)['state_dict']
        weights_to_load = OrderedDict({})

        prefix = 'encoder.'
        prefix_len = len(prefix)

        for k, v in stage1_statedict.items():
            if k.startswith(prefix):
                trimmed_k = k[prefix_len:]
                weights_to_load[trimmed_k] = v

        self.load_state_dict(weights_to_load)
        
        self.merge_gaussian_param_head = deepcopy(self.gaussian_param_head)
        self.merge_gaussian_param_head2 = deepcopy(self.gaussian_param_head2)

        self.is_all_layer = False
        if self.is_all_layer:          
            self.merge_rate = nn.Sequential(
                nn.Conv2d(3, 256, 7, 1, 3),
                nn.ReLU(),
                nn.Conv2d(256, 768, 3, 1, 1),
                nn.ReLU(),
            )
        else:
            self.merge_rate = nn.Sequential(
                nn.Conv2d(3, 256, 7, 1, 3),
                nn.ReLU(),
            )
                
        # freeze all stage1 param
        for name, param in self.named_parameters():
            if "merge_" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)


    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)

        elif 'dpt' in head_type:
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
        
   
    def set_pose_head(self, cfg, head_type='mlp'):
        self.pose_head = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)
        self.pose_head2 = camera_head_factory(head_type, 'pose', self.backbone, cfg.pose_head)


    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)
    
    def build_covariance_from_feats(self, scales, rotations):
        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)
        
        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        return build_covariance(scales, rotations)

    def get_kld_cov(self, mu_merge, mu, cov): # mu*, mu_i, sigma_i
        mu_diff = (mu - mu_merge)
        mu_cov = torch.matmul(mu_diff.unsqueeze(-1), mu_diff.unsqueeze(-2))
        return cov + mu_cov

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        primitive_ratio: float = 1.0,
        visualization_dump: Optional[dict] = None,
        target: Optional[dict] = None,
    ) :
        device = context["image"].device
        b, v_cxt, _, h, w = context["image"].shape
        distill_infos = {}

        high_freq_score_lists = []
        for i in range(context["image"].shape[1]):
            high_freq_score = compute_frequency_high_freq_score(context["image"][0][i])
            high_freq_score_lists.append(high_freq_score)
        high_freq_scores = torch.stack(high_freq_score_lists, dim=0).to(device)  # (v_cxt,)
        high_freq_score_tensor = F.softmax(high_freq_scores/0.3, dim=0)
        
        if target is not None:
            v_tgt = target["image"].shape[1]
            context_target = {
                "image": normalize_image(torch.cat([context["image"], target["image"]], dim=1)),
                "intrinsics": torch.cat([context["intrinsics"], target["intrinsics"]], dim=1),
            }
            # Encode the context and target images.
            out = self.backbone(context_target, target_num_views=v_tgt)
            dec_feat, dec_feat_w_tgt = out['dec_feat'], out['dec_feat_w_tgt']
        else:
            v_tgt = 0
            context_input = {
                "image": normalize_image(context["image"]),
                "intrinsics": context["intrinsics"],
            }
            # Encode the context images.
            out = self.backbone(context_input)
            dec_feat = out['dec_feat']

        shape, images = out["shape"], out["images"]
        
        with torch.amp.autocast('cuda', enabled=False):
            all_mean_res, all_other_params, all_other_merge_params = [], [], []
            if self.cfg.estimating_pose:
                all_pose_params, all_pose_params_cwt = [], []

            # merge heads
            primitive_ratio = 0.4
            primitive_ratio_per_view = primitive_ratio * high_freq_score_tensor / high_freq_score_tensor.mean()
            # primitive_ratio_per_view = torch.clamp(primitive_ratio * high_freq_score_tensor / high_freq_score_tensor.mean(), 0.05, 0.95)
            if self.is_all_layer:
                rate_feat = self.merge_rate((torch.ones(b*v_cxt,3,h//self.patch_size,w//self.patch_size).type_as(context["image"])*primitive_ratio_per_view.view(-1, 1, 1, 1))).view(b,v_cxt,-1,h//self.patch_size,w//self.patch_size)
            else:
                rate_feat = self.merge_rate((torch.ones_like(context["image"])*primitive_ratio_per_view.view(1, -1, 1, 1, 1)).view(b*v_cxt,-1,h,w)).view(b,v_cxt,-1,h,w)

            # Pts3d head (context only)
            res1 = self._downstream_head(1, [tok[:, 0].float() for tok in dec_feat], shape[:, 0])['pts3d']
            all_mean_res.append(res1[:,None])
            res2 = self._downstream_head(2, [tok[:, 1:v_cxt].flatten(0,1).float() for tok in dec_feat], shape[:, 0])['pts3d']
            res2 = res2.view(b,v_cxt-1,h,w,-1)
            all_mean_res.append(res2)
            
            # Gaussian parameter head (context only)
            if 'dpt' in self.gs_params_head_type:
                GS_res1 = self.gaussian_param_head([tok[:, 0].float() for tok in dec_feat], images[:, 0, :3], shape[0, 0].cpu().tolist())
                GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                all_other_params.append(GS_res1[:,None])
                
                GS_res2 = self.gaussian_param_head2([tok[:, 1:v_cxt].flatten(0,1).float() for tok in dec_feat], images[:, 1:v_cxt, :3].flatten(0,1), shape[0, 0].cpu().tolist())
                GS_res2 = GS_res2.view(b,v_cxt-1,-1,h*w).permute(0,1,3,2)
                all_other_params.append(GS_res2)

            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")
            
            # Merged Gaussian parameter head (context only)
            if 'dpt' in self.gs_params_head_type:
                merged_GS_res1 = self.merge_gaussian_param_head([tok[:, 0].float() for tok in dec_feat], images[:, 0, :3], shape[0, 0].cpu().tolist(), rate_feat=rate_feat[:,0])
                merged_GS_res1 = rearrange(merged_GS_res1, "b d h w -> b (h w) d")
                all_other_merge_params.append(merged_GS_res1[:,None])
                
                merged_GS_res2 = self.merge_gaussian_param_head2([tok[:, 1:v_cxt].flatten(0,1).float() for tok in dec_feat], images[:, 1:v_cxt, :3].flatten(0,1), shape[0, 0].cpu().tolist(), rate_feat=rate_feat[:,1:v_cxt].flatten(0,1))
                merged_GS_res2 = merged_GS_res2.view(b,v_cxt-1,-1,h*w).permute(0,1,3,2)
                all_other_merge_params.append(merged_GS_res2)

            else:
                raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

            # Pose head
            if self.cfg.estimating_pose:
                # Context views
                pose_res1 = self.pose_head([tok[:, 0].float() for tok in dec_feat], shape[0, 0].cpu().tolist()) # (16, 9)
                all_pose_params.append(pose_res1[: ,None])
                
                pose_res2 = self.pose_head2([tok[:, 1:v_cxt].flatten(0,1).float() for tok in dec_feat], shape[0, 0].cpu().tolist()) # (16*(v-1), 9)
                pose_res2 = pose_res2.view(b,v_cxt-1,-1)
                all_pose_params.append(pose_res2)

                # Context + target views
                if target is not None:
                    pose_res1 = self.pose_head([tok[:, 0].float() for tok in dec_feat_w_tgt], shape[0, 0].cpu().tolist()) # (16, 9)
                    all_pose_params_cwt.append(pose_res1[: ,None])
                    pose_res2 = self.pose_head2([tok[:, 1:v_cxt + v_tgt].flatten(0,1).float() for tok in dec_feat_w_tgt], shape[0, 0].cpu().tolist()) # (16*(v-1), 9)
                    pose_res2 = pose_res2.view(b,v_cxt + v_tgt -1,-1)
                    all_pose_params_cwt.append(pose_res2)
            
        gaussians = torch.cat(all_other_params, dim=1) # [b, v, 65536, 83]
        merge_gaussians = torch.cat(all_other_merge_params, dim=1)
        
        if self.cfg.estimating_pose:
            poses_enc = torch.cat(all_pose_params, dim=1) # (b, v, 9)
            pred_extrinsics = self.process_pose(poses_enc, v_cxt) # (b, v, 4, 4)

            if target is not None:
                poses_enc_cwt = torch.cat(all_pose_params_cwt, dim=1) # (b, v + v2, 9)
                pred_extrinsics_cwt = self.process_pose(poses_enc_cwt, v_cxt) # (b, v + v2, 4, 4)

        pts_all = torch.cat(all_mean_res, dim=1) # [b, v, h, w, 3]
        pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz")
        
        context_extrinsics = pred_extrinsics[:, :v_cxt] if self.cfg.estimating_pose else context["extrinsics"]
        coords_per_view = self.process_depth(context_extrinsics, rearrange(pts_all, "b v (h w) xyz -> b v h w xyz", h=h, w=w)) # depth for each cam, (b, v, h, w)
        depths_per_view = coords_per_view[..., 2]
        

        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces) # for cfg.num_surfaces
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        gaussian_parameters = gaussians[..., 1:]
        
        merge_gaussians = rearrange(merge_gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces) # for cfg.num_surfaces
        merge_densities = merge_gaussians[..., 0].sigmoid().unsqueeze(-1)
        merge_gaussian_parameters = merge_gaussians[..., 1:]      

        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

        # pseudo merging
        normal_map = get_normal_map(depths_per_view.view(b*v_cxt,h,w), context["intrinsics"].view(b*v_cxt,3,3)).view(b,v_cxt,h,w,3)
        normal_grad = get_grad_map(normal_map.view(b*v_cxt,h,w,3).permute(0,3,1,2)).view(b,v_cxt,1,h,w).permute(0,1,3,4,2)
        img_grad = get_grad_map(context["image"].view(b*v_cxt,3,h,w)).view(b,v_cxt,1,h,w).permute(0,1,3,4,2)        
        grad_map = torch.cat([normal_grad, img_grad], dim=-1).mean(-1).view(b*v_cxt,h, w)
        
        sx,sy = 4, 4
        patch_grad_map = grad_map.view(b*v_cxt,h//sy,sy,w//sx,sx).transpose(2,3).reshape(b*v_cxt,h//sy,w//sx,sx*sy)
        grad_thes_list = []
        for i in range(len(patch_grad_map)):
            grad_thes_list.append(torch.quantile(patch_grad_map.sum(-1).flatten(1,2)[i], 1.0-primitive_ratio_per_view[i].item()))
        grad_thes = torch.stack(grad_thes_list, dim=0) 
        merge_mask = torch.le(patch_grad_map.sum(-1), grad_thes[:, None, None])
        full_merge_mask = merge_mask.unsqueeze(-1).expand(-1,-1,-1,sx*sy).view(b*v_cxt, h//sy, w//sx, sy, sx).transpose(2,3).reshape(b*v_cxt,h,w)

        norm_uv = normalized_view_plane_uv(w, h, dtype=pts_all.dtype, device=pts_all.device)
        norm_pts_all = pts_all.view(b,v_cxt,h*w,3).clone()
        min_xyz = norm_pts_all.min(dim=2, keepdim=True).values
        max_xyz = norm_pts_all.max(dim=2, keepdim=True).values
        norm_pts_all =  (norm_pts_all - min_xyz) / (max_xyz - min_xyz + 1e-8)
        agg_features = torch.cat([(pts_all.flatten(0,3)).view(b*v_cxt, h*w,-1), (norm_uv.flatten(0,1))[None].expand(b*v_cxt,-1,-1)], dim=-1)
        
        m, _, _ = bipartite_soft_matching_random2d(agg_features, w=w,h=h,sx=sx,sy=sy,r=int(agg_features.shape[1] * 0.5), p=int(agg_features.shape[1] * 0.2), generator=init_generator(agg_features.device), merge_mask=full_merge_mask)    
        merged_pts_all = m(pts_all.view(b*v_cxt,h*w,3))
        N_tokens = merged_pts_all.shape[1]
        curr_intrin = context["intrinsics"]
        projection = torch.zeros_like(depths_per_view).squeeze(-1)
        cam_pts = self.process_coords(context_extrinsics, merged_pts_all).view(b,v_cxt,N_tokens,3,1)
        pixel_pts = torch.matmul(curr_intrin[:,:,None], cam_pts).squeeze(-1)
        pixel_pts /= (pixel_pts[...,2:] +1e-8)
        
        valid_mask = (pixel_pts[...,0] >=0) & (pixel_pts[...,0] < 1) & (pixel_pts[...,1] >=0) & (pixel_pts[...,1] < 1) & (cam_pts.squeeze(-1)[...,2]>0)

        distill_infos["pixel_pts"] = 2*pixel_pts[...,:2].clone().flatten(0,1) -1.0

        pixel_pts[...,0] = torch.clamp(pixel_pts[...,0]*w, 0, w-1)
        pixel_pts[...,1] = torch.clamp(pixel_pts[...,1]*h, 0, h-1)
        pixel_pts = pixel_pts.long()
        
        flatten_index = pixel_pts[...,1]*w + pixel_pts[...,0]
        projection = projection.view(b,v_cxt,-1)

        projection.scatter_(2, flatten_index, torch.ones_like(flatten_index, dtype=projection.dtype)*valid_mask.type(projection.dtype))
        distill_infos["alpha"] = projection
            
        learn_merged_pts_all =  pts_all
        learn_merged_gaussian_parameters = merge_gaussian_parameters
        learn_merged_densities = merge_densities

        learn_merged_opacity = self.map_pdf_to_opacity(merge_densities, global_step).squeeze(-1).squeeze(-1)
        
        distill_infos["learn_alpha"] = learn_merged_opacity

        pts_list = []
        opa_list = []
        gs_param_list = []
        for i in range(len(primitive_ratio_per_view)):
            # _, top_opa_id = torch.topk(learn_merged_opacity[:, i], k=int(max(primitive_ratio_per_view[i], 1/16)*h*w), dim=-1)
            _, top_opa_id = torch.topk(learn_merged_opacity[:, i], k=int(primitive_ratio_per_view[i]*h*w), dim=-1)
            learn_merged_pts_all = torch.gather(pts_all[:, i], 1, top_opa_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,pts_all.shape[-1])).unsqueeze(1)
            learn_merged_gaussian_parameters = torch.gather(merge_gaussian_parameters[:, i], 1, top_opa_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,merge_gaussian_parameters.shape[-1])).unsqueeze(1)
            learn_merged_densities = torch.gather(merge_densities[:, i], 1, top_opa_id.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,merge_densities.shape[-1])).unsqueeze(1)

            pts_list.append(learn_merged_pts_all.unsqueeze(-2))
            opa_list.append(self.map_pdf_to_opacity(learn_merged_densities, global_step))
            gs_param_list.append(rearrange(learn_merged_gaussian_parameters, "b v r srf c -> b v r srf () c"))

        pts = torch.cat(pts_list, dim=2)
        opa = torch.cat(opa_list, dim=2)
        gs_param = torch.cat(gs_param_list, dim=2)

        gaussians = self.gaussian_adapter.forward(pts, opa, gs_param)

        # Convert the features and depths into Gaussians.
        ori_gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussian_parameters, "b v r srf c -> b v r srf () c"),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = depths_per_view
            
        if self.cfg.estimating_focal:
            intrinsics = estimate_intrinsics(rearrange(gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w).squeeze(-2), h, w)
            pred_intrinsics = intrinsics.unsqueeze(1).repeat(1, v_cxt, 1, 1)
            pred_intrinsics_cwt = intrinsics.unsqueeze(1).repeat(1, v_cxt+v_tgt, 1, 1)

        encoder_output = dict()

        encoder_output["gaussians"] = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.rotations,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                gaussians.scales,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            )
        )
        
        encoder_output["ori_gaussians"] = Gaussians(
            rearrange(
                ori_gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                ori_gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                ori_gaussians.rotations,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                ori_gaussians.scales,
                "b v r srf spp i  -> b (v r srf spp) i ",
            ),
            rearrange(
                ori_gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                ori_gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            )
        )

        if self.cfg.estimating_pose:
            encoder_output['extrinsics'] = {"c": pred_extrinsics}
            if target is not None:
                encoder_output['extrinsics']['cwt'] = pred_extrinsics_cwt

        if self.cfg.estimating_focal:
            encoder_output['intrinsics'] = {"c": pred_intrinsics}
            if target is not None:
                encoder_output['intrinsics']['cwt'] = pred_intrinsics_cwt
                
        encoder_output['distill_infos'] = distill_infos
        encoder_output['compression_ratio'] = primitive_ratio
        return encoder_output

    def process_pose(self, pose_enc, context_views):
        # pose_enc: (b v 9)
        b, v = pose_enc.shape[:2]
        poses = convert_pose_to_4x4(rearrange(pose_enc, "b v ... -> (b v) ..."))
        poses = rearrange(poses, "(b v) ... -> b v ...", b=b, v=v)

        if self.cfg.pose_make_baseline_1:
            a = poses[:, 0, :3, 3]  # [b, 3]
            b = poses[:, 1, :3, 3]  #  [b, 3]

            scale = (a - b).norm(dim=1, keepdim=True)  # [b, 1]

            poses[:, :, :3, 3] /= scale.unsqueeze(-1)

        if self.cfg.pose_make_relative:
            base_context_pose = poses[:,0] # [b, 4, 4]
            inv_base_context_pose = torch.inverse(base_context_pose)
            poses = inv_base_context_pose[:, None, :, :] @ poses # [b,1,4,4] @ [b,v,4,4]

        return poses      
    
    def process_depth(self, pose, pts3d):
        b, v, h, w, _ = pts3d.shape
        pts3d = rearrange(pts3d, "b v h w c -> (b v) (h w) c")
        pose = rearrange(pose, "b v ... -> (b v) ...")
        coords = depth_projector(pts3d, pose) # (bv, n, 2)
        coords = rearrange(coords, "(b v) (h w) c -> b v h w c", b=b, v=v, h=h, w=w, c=3)
        return coords.contiguous()
    
    def process_coords(self, pose, pts3d):
        pose = rearrange(pose, "b v ... -> (b v) ...")
        coords = depth_projector(pts3d, pose) # (bv, n, 2)
        return coords

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim

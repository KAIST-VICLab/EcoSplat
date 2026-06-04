from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding

from zpressor import ZPressor
from zpressor.utils import center_filter

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    use_cluster: bool
    cluster_num: int
    num_heads: int
    num_layers: int
    igf: Optional[dict] = None  # EcoSplat IGF stage-2 finetune cfg; None disables.
    stage1_weights_path: Optional[str] = None  # stage-1 ckpt loaded into encoder before IGF init.
    freeze_pretrained: bool = False  # True = stage2 SPFSplat style (freeze all except IGF).
    ib_distill_weight: float = 0.0  # >0 enables IB feature distillation: anchor the
    # ZPressor (zipmatch) compressed output to a frozen stage-1 copy. Soft alternative
    # to freeze_pretrained — preserves the bottleneck representation while letting the
    # rest of the backbone (esp. depth_predictor) adapt to the IGF objective.

class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            if cfg.use_cluster:
                num_views_epipolar = get_cfg().model.encoder.cluster_num
            else:
                num_views_epipolar = get_cfg().dataset.view_sampler.num_context_views
            self.epipolar_sampler = EpipolarSampler(
                num_views=num_views_epipolar,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
            use_cluster=cfg.use_cluster,
            cluster_embed_dim=cfg.d_feature,
            cluster_num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=False)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        num_views = get_cfg().model.encoder.cluster_num if cfg.use_cluster else get_cfg().dataset.view_sampler.num_context_views
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=num_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )

        # EcoSplat IGF stage-2: clones `to_gaussians`, adds 256-ch rate_embed,
        # projects rate_feat to `depth_unet_feat_dim` for shallow-add to refine_out
        # (mvsplat analog of SPFSplat DPT path_1 injection).
        self.igf = None
        self.igf_rate_proj = None
        self.last_igf = None
        if cfg.igf is not None:
            if cfg.stage1_weights_path:
                print("==> Load EcoSplat stage-1 weights for IGF: %s" % cfg.stage1_weights_path)
                stage1_sd = torch.load(cfg.stage1_weights_path, map_location="cpu")
                if isinstance(stage1_sd, dict) and "state_dict" in stage1_sd:
                    stage1_sd = stage1_sd["state_dict"]
                # Strip common Lightning prefixes (e.g. "encoder.").
                cleaned = OrderedDict()
                strip_prefixes = ("encoder.", "model.encoder.")
                for k, v in stage1_sd.items():
                    nk = k
                    for p in strip_prefixes:
                        if nk.startswith(p):
                            nk = nk[len(p):]
                            break
                    cleaned[nk] = v
                missing, unexpected = self.load_state_dict(cleaned, strict=False)
                if missing:
                    print(f"   [IGF] missing keys: {len(missing)} (first few: {list(missing)[:5]})")
                if unexpected:
                    print(f"   [IGF] unexpected keys: {len(unexpected)} (first few: {list(unexpected)[:5]})")

            # Clone BOTH `to_gaussians` (xy_offset/scale/rot/sh) and `to_disparity`
            # (delta_disp + density). Without the latter, merge-head opacity = stage-1
            # opacity → L_io BCE has no gradient signal and IGF never trains.
            from ecosplat_wrapper import IGFConfig, IGFModule
            self.igf = IGFModule(
                heads_to_clone=[
                    self.depth_predictor.to_gaussians,
                    self.depth_predictor.to_disparity,
                ],
                igf_cfg=IGFConfig(**dict(cfg.igf)),
            )
            # Project 256-ch rate_embed output down to refine_out channels for shallow-add.
            # Default Kaiming-uniform init (PyTorch nn.Conv2d default) — must be non-zero
            # so the merge head sees rate_feat from step 0; otherwise L_io has no gradient
            # signal and IGF never trains.
            self.igf_rate_proj = nn.Conv2d(256, cfg.depth_unet_feat_dim, kernel_size=1)

            # Freeze toggle: stage2 SPFSplat-style freezes everything except IGF.
            # Default off — train all params with lr-split via configure_optimizers.
            if cfg.freeze_pretrained:
                for p in self.parameters():
                    p.requires_grad = False
                for p in self.igf.parameters():
                    p.requires_grad = True
                for p in self.igf_rate_proj.parameters():
                    p.requires_grad = True
                print("[IGF] freeze_pretrained=True — only IGF + igf_rate_proj train.")

            # IB feature distillation: snapshot the just-loaded (stage-1) zipmatch as
            # a frozen reference. Soft alternative to freeze_pretrained that keeps the
            # ZPressor bottleneck intact while the rest of the backbone adapts.
            if cfg.ib_distill_weight > 0:
                self.backbone.enable_ib_distill()
                print(f"[IGF] IB feature distillation ON (weight={cfg.ib_distill_weight}).")

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

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = {
                "extrinsics": context["extrinsics"],
            }
        trans_features, cnn_features, center_views = self.backbone(
            context["image"],
            cluster_num=self.cfg.cluster_num,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        if self.cfg.use_cluster:
            images = center_filter(context["image"], center_views)
            extrinsics = center_filter(context["extrinsics"], center_views)
            intrinsics = center_filter(context["intrinsics"], center_views)
            near = center_filter(context["near"], center_views)
            far = center_filter(context["far"], center_views)

        else:
            extrinsics = context["extrinsics"]
            intrinsics = context["intrinsics"]
            near = context["near"]
            far = context["far"]
            images = context["image"]

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(images, "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        # IGF stage-2: sample protect_rate + build rate_feat for merge-head.
        igf_active = self.igf is not None
        rate_feat_proj = None
        merge_head = None
        protect_rate = None
        if igf_active:
            override = context.get("protect_rate") if isinstance(context, dict) else None
            protect_rate = self.igf.get_rho(global_step, self.training, override=override)
            v_use = images.shape[1]
            rho_3ch = torch.ones(
                v_use * b, 3, h, w, device=device, dtype=images.dtype
            ) * protect_rate
            rate_feat = self.igf.rate_embed(rho_3ch)  # (vb, 256, h, w)
            assert rate_feat.shape[1] == 256, (
                f"rate_feat must have 256 channels (got {rate_feat.shape[1]})."
            )
            rate_feat_proj = self.igf_rate_proj(rate_feat)  # (vb, depth_unet_feat_dim, h, w)
            merge_head = self.igf.merge_heads[0]
            merge_disparity = self.igf.merge_heads[1]
        else:
            merge_disparity = None

        depths, densities, raw_gaussians, merged_raw_gaussians, merged_densities_pred = self.depth_predictor(
            in_feats,
            intrinsics,
            extrinsics,
            near,
            far,
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            rate_feat_proj=rate_feat_proj,
            merge_head=merge_head,
            merge_disparity=merge_disparity,
        )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # IGF stage-2: build merged Gaussians from cloned-head raw output and stash
        # distill_infos for the IGF loss.
        merged_gaussians_flat = None
        if igf_active and merged_raw_gaussians is not None:
            merged_split = rearrange(
                merged_raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            merged_offset_xy = merged_split[..., :2].sigmoid()
            merged_xy_ray_grid, _ = sample_image_grid((h, w), device)
            merged_xy_ray_grid = rearrange(merged_xy_ray_grid, "h w xy -> (h w) () xy")
            merged_xy_ray = merged_xy_ray_grid + (merged_offset_xy - 0.5) * pixel_size

            # Use merge-head-predicted densities (NOT stage-1's) so L_io BCE has a
            # learnable gradient signal. merged_densities_pred is (b, v, h*w, srf, gpp).
            merged_densities = merged_densities_pred
            merged_opacities_per_pixel = self.map_pdf_to_opacity(
                merged_densities, global_step
            ).flatten(-3, -1)  # (b, v, h*w)

            # Top-k by per-pixel merged opacity (paper PLGC selection).
            # Floor protect_rate at 1/16 to avoid pathologically tiny renders.
            k_top = int(max(protect_rate, 1.0 / 16.0) * h * w)
            _, top_idx = torch.topk(merged_opacities_per_pixel, k=k_top, dim=-1)

            merged_params_full = merged_split[..., 2:]  # (b, v, h*w, srf, c)
            idx_xy = top_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, merged_xy_ray.shape[-2], merged_xy_ray.shape[-1]
            )
            xy_topk = torch.gather(merged_xy_ray, 2, idx_xy)
            idx_d = top_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, depths.shape[-2], depths.shape[-1]
            )
            depths_topk = torch.gather(depths, 2, idx_d)
            dens_topk = torch.gather(merged_densities, 2, idx_d)
            idx_p = top_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, merged_params_full.shape[-2], merged_params_full.shape[-1]
            )
            params_topk = torch.gather(merged_params_full, 2, idx_p)

            merged_gaussians = self.gaussian_adapter.forward(
                rearrange(extrinsics, "b v i j -> b v () () () i j"),
                rearrange(intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_topk, "b v r srf xy -> b v r srf () xy"),
                depths_topk,
                self.map_pdf_to_opacity(dens_topk, global_step) / gpp,
                rearrange(params_topk, "b v r srf c -> b v r srf () c"),
                (h, w),
            )

            # Flatten merged Gaussians.
            merged_gaussians_flat = Gaussians(
                rearrange(merged_gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(merged_gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(merged_gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                rearrange(merged_gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
                rotations=rearrange(merged_gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"),
                scales=rearrange(merged_gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
            )

            # Pseudo-GT (importance mask) is loss-only: only model_wrapper's
            # training_step reads `last_igf`. Skip it at inference — `test_step`
            # renders `merged_gaussians_flat` directly and never touches it.
            if self.training:
                # Stash side-channel for model_wrapper to pick up.
                ori_flat = Gaussians(
                    rearrange(gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                    rearrange(gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                    rearrange(gaussians.harmonics, "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"),
                    rearrange(gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
                    rotations=rearrange(gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"),
                    scales=rearrange(gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                )
                v_use = gaussians.means.shape[1]
                means_pix = gaussians.means.reshape(b, v_use, h, w, 3)
                cov_pix = gaussians.covariances.reshape(b, v_use, h, w, 3, 3)
                distill_infos = self.igf.compute_distill(
                    ori_gaussians={"means_pix": means_pix, "cov_pix": cov_pix},
                    image=images,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    protect_rate=protect_rate,
                )
                distill_infos["pred_opacity"] = merged_opacities_per_pixel
                self.last_igf = {
                    "ori_gaussians": ori_flat,
                    "distill_infos": distill_infos,
                    "protect_rate": protect_rate,
                }
            else:
                self.last_igf = None
        else:
            self.last_igf = None

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        # Stage-2 (IGF): return merged Gaussians for rendering; stage-1 ori_gaussians
        # is already stashed on self.last_igf for the distill loss.
        if merged_gaussians_flat is not None:
            return merged_gaussians_flat

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            rotations=rearrange(
                gaussians.rotations,
                "b v r srf spp xyzw -> b (v r srf spp) xyzw",
            ),
            scales=rearrange(
                gaussians.scales,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None

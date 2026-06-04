import copy

import torch
import torch.nn.functional as F
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from ..costvolume.conversions import depth_to_relative_disparity
from ....geometry.epipolar_lines import get_depth


from zpressor import ZPressor
from zpressor.utils import center_filter


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
        use_cluster=False,
        cluster_embed_dim=128,
        cluster_num_heads=1,
        num_layers=6,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels
        # Table 3: w/o cross-view attention
        self.no_cross_attn = no_cross_attn
        # Table B: w/ Epipolar Transformer
        self.use_epipolar_trans = use_epipolar_trans
        self.use_cluster = use_cluster
        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        if use_cluster:
            self.zipmatch = ZPressor(
                embed_dim=cluster_embed_dim,
                num_heads=cluster_num_heads,
                num_layers=num_layers,
            )

        # IB feature distillation (opt-in). When enabled, `ib_distill_ref` is a
        # frozen stage-1 copy of `zipmatch`; the forward pass anchors the live
        # compressed output to the reference (preserving the ZPressor bottleneck
        # representation during IGF fine-tuning). `ib_distill_loss` is the per-step
        # MSE side-channel read by the ModelWrapper. Both inert unless enabled.
        self.ib_distill_ref = None
        self.ib_distill_loss = None

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def enable_ib_distill(self):
        """Snapshot the current `zipmatch` as a frozen reference for IB feature
        distillation. Call AFTER stage-1 weights are loaded so the reference IS
        the converged stage-1 bottleneck. No-op if clustering is disabled.

        The reference is NOT meant to be saved/restored — it is recreated from
        stage-1 by the host whenever distillation is enabled; it is excluded from
        this module's state_dict (see `state_dict` override below) so training
        checkpoints stay the same size as a normal run.
        """
        if not self.use_cluster:
            return
        self.ib_distill_ref = copy.deepcopy(self.zipmatch)
        self.ib_distill_ref.eval()
        for p in self.ib_distill_ref.parameters():
            p.requires_grad_(False)

    def state_dict(self, *args, **kwargs):
        # Keep the frozen distillation reference out of checkpoints — it is a
        # transient copy of stage-1 `zipmatch`, re-created via enable_ib_distill().
        # PyTorch's recursive state_dict calls this child with a `prefix=` kwarg, so
        # ref keys appear as "<prefix>ib_distill_ref.*"; the substring match below
        # is prefix-agnostic and also covers the no-prefix (direct) call.
        sd = super().state_dict(*args, **kwargs)
        for k in [k for k in sd if "ib_distill_ref." in k]:
            del sd[k]
        return sd

    def forward(
        self,
        images,
        cluster_num=2,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features

        cur_features_list = [x[0] for x in features_list]

        cnn_features = torch.stack(cur_features_list, dim=1)

        if self.use_cluster:
            zipmatch_input = cnn_features  # pre-compression (B, V, C, H, W)
            cnn_features, center_views = self.zipmatch(
                zipmatch_input,
                extrinsics=epipolar_kwargs["extrinsics"],
                cls_token=None,
                cluster_num=cluster_num,
            )
            # IB feature distillation: anchor the live compressed output to the
            # frozen stage-1 zipmatch on the SAME input. Anchor selection (FPS on
            # camera positions) is parameter-free, so the reference output is
            # shape-aligned with the live one. Training-only; stashed for the host.
            if self.training and self.ib_distill_ref is not None:
                with torch.no_grad():
                    cnn_ref, _ = self.ib_distill_ref(
                        zipmatch_input,
                        extrinsics=epipolar_kwargs["extrinsics"],
                        cls_token=None,
                        cluster_num=cluster_num,
                    )
                self.ib_distill_loss = F.mse_loss(cnn_features, cnn_ref)
            else:
                self.ib_distill_loss = None
            images = center_filter(images, center_views)
            # cur_feature->cur_features_list
            cur_features_list = torch.unbind(cnn_features, dim=1)
            if self.use_epipolar_trans:
                extrinsics = center_filter(epipolar_kwargs["extrinsics"], center_views)
                intrinsics = center_filter(epipolar_kwargs["intrinsics"], center_views)
                near = center_filter(epipolar_kwargs["near"], center_views)
                far = center_filter(epipolar_kwargs["far"], center_views)
        else:
            if self.use_epipolar_trans:
                extrinsics = epipolar_kwargs["extrinsics"]
                intrinsics = epipolar_kwargs["intrinsics"]
                near = epipolar_kwargs["near"]
                far = epipolar_kwargs["far"]

        # if return_cnn_features:
        #     cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if self.use_epipolar_trans:
            # NOTE: Epipolar Transformer, only for ablation used
            # we only abalate Epipolar Transformer under 2 views setting
            assert (
                epipolar_kwargs is not None
            ), "must provide camera params to apply epipolar transformer"
            assert len(cur_features_list) == 2, "only use 2 views input for Epipolar Transformer ablation"
            feature0, feature1 = cur_features_list
            epipolar_sampler = epipolar_kwargs["epipolar_sampler"]
            depth_encoding = epipolar_kwargs["depth_encoding"]

            features = torch.stack((feature0, feature1), dim=1)  # [B, V, C, H, W]
            # Get the samples used for epipolar attention.
            sampling = epipolar_sampler.forward(
                features, extrinsics, intrinsics, near, far
            )
            # similar to pixelsplat, use camera distance as position encoding
            # Compute positionally encoded depths for the features.
            collect = epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )

            # Clip the depths. This is necessary for edge cases where the context views
            # are extremely close together (or possibly oriented the same way).
            depths = depths.maximum(near[..., None, None, None])
            depths = depths.minimum(far[..., None, None, None])
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            depths = depth_encoding(depths[..., None])
            target = sampling.features + depths
            source = features

            features = self.transformer((source, target), attn_type="epipolar")
        else:
            # add position to features
            cur_features_list = feature_add_position_list(
                cur_features_list, attn_splits, self.feature_channels)

            # Transformer
            cur_features_list = self.transformer(
                cur_features_list, attn_num_splits=attn_splits)

            features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if return_cnn_features and self.use_cluster:
            out_lists = [features, cnn_features, center_views]
        elif return_cnn_features:
            out_lists = [features, cnn_features, None]
        elif self.use_cluster:
            out_lists = [features, None, center_views]
        else:
            out_lists = [features, None, None]

        return out_lists

"""IGFModule smoke tests."""

import torch
import torch.nn.functional as F
from torch import nn

from ecosplat_wrapper import (
    IGFConfig,
    IGFModule,
    LossImportanceOpacity,
    LossImportanceOpacityCfg,
    LossImportanceOpacityCfgWrapper,
)


def test_igf_module_construction_clones_heads_and_builds_stage2_rate_embed():
    head = nn.Conv2d(8, 16, 3, padding=1)

    igf = IGFModule(heads_to_clone=[head])

    assert isinstance(igf.merge_heads, nn.ModuleList)
    assert len(igf.merge_heads) == 1
    assert id(igf.merge_heads[0]) != id(head)
    assert igf.merge_heads[0].weight.requires_grad is True

    # Stage-2 rate_embed: fresh Conv2d(3->256, k=7) + ReLU, Kaiming-uniform init.
    assert isinstance(igf.rate_embed, nn.Sequential)
    conv = igf.rate_embed[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 3
    assert conv.out_channels == 256
    assert conv.kernel_size == (7, 7)
    assert isinstance(igf.rate_embed[1], nn.ReLU)
    # Kaiming-uniform init → weights non-zero.
    assert not torch.all(conv.weight == 0)


def test_igf_module_image_embed_kwarg_is_ignored():
    head = nn.Conv2d(8, 16, 3, padding=1)
    image_embed = nn.Conv2d(3, 8, 1)
    image_embed.weight.data.fill_(0.7)
    # image_embed retained for backwards compat but not used.
    igf = IGFModule(heads_to_clone=head, image_embed=image_embed)
    # rate_embed is fresh Conv2d(3->256), not derived from image_embed.
    assert igf.rate_embed[0].out_channels == 256


def test_igf_get_rho_inference_default_and_override():
    igf = IGFModule(
        heads_to_clone=nn.Linear(4, 4),
        image_embed=nn.Conv2d(3, 4, 1),
    )
    # eval: returns cfg.inference_rho (default 0.4)
    assert igf.get_rho(0, training=False) == 0.4
    # override: caller-supplied value wins
    assert igf.get_rho(0, training=False, override=0.7) == 0.7
    assert igf.get_rho(0, training=True, override=0.2) == 0.2
    # custom inference_rho via cfg
    igf2 = IGFModule(
        heads_to_clone=nn.Linear(4, 4),
        image_embed=nn.Conv2d(3, 4, 1),
        igf_cfg=IGFConfig(inference_rho=0.6),
    )
    assert igf2.get_rho(0, training=False) == 0.6


def test_igf_sample_rho_in_range():
    igf = IGFModule(
        heads_to_clone=nn.Linear(4, 4),
        image_embed=nn.Conv2d(3, 4, 1),
    )
    for _ in range(20):
        rho = igf.sample_rho(0)
        assert 0.85 - 1e-6 <= rho <= 0.95 + 1e-6


def test_igf_compute_distill_smoke():
    B, V, H, W = 1, 2, 16, 16
    head = nn.Conv2d(8, 16, 3, padding=1)
    image_embed = nn.Conv2d(3, 8, 1)
    igf = IGFModule(heads_to_clone=[head], image_embed=image_embed)

    image = torch.rand(B, V, 3, H, W)
    intrinsics = torch.eye(3).expand(B, V, 3, 3).clone()
    extrinsics = torch.eye(4).expand(B, V, 4, 4).clone()
    pts3d = torch.randn(B, V, H, W, 3)
    cov_feat = torch.eye(3).expand(B, V, H, W, 3, 3).clone()

    ori_gs = {"means_pix": pts3d, "cov_pix": cov_feat}
    out = igf.compute_distill(
        ori_gaussians=ori_gs,
        image=image,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        protect_rate=0.9,
    )
    assert set(out.keys()) >= {"importance_mask", "kld_cov", "valid_mask", "pixel_pts"}


def test_igf_compute_loss_matches_direct_loss():
    head = nn.Linear(4, 4)
    image_embed = nn.Conv2d(3, 4, 1)
    igf = IGFModule(heads_to_clone=[head], image_embed=image_embed)

    pred = torch.rand(2, 4, 16).clamp(1e-3, 1 - 1e-3)
    gt = torch.rand(2, 4, 16).clamp(1e-3, 1 - 1e-3)
    rendered = torch.rand(2, 4, 16).clamp(1e-3, 1 - 1e-3)

    class FakeOut:
        pass
    out = FakeOut()
    out.alpha = rendered
    distill = {"importance_mask": gt, "pred_opacity": pred}

    got = igf.compute_loss(distill, out, out)
    direct = LossImportanceOpacity(
        LossImportanceOpacityCfgWrapper(
            importance_opacity=LossImportanceOpacityCfg(weight=0.1)
        )
    )(distill, out, out)
    assert torch.allclose(got, direct, atol=1e-7)


def test_igf_rate_embed_output_is_256_channels_at_input_resolution():
    igf = IGFModule(heads_to_clone=nn.Linear(4, 4))
    B, V, H, W = 1, 2, 32, 32
    rho_3ch = torch.ones(B * V, 3, H, W) * 0.4
    rate_feat = igf.rate_embed(rho_3ch)
    assert rate_feat.shape == (B * V, 256, H, W)


if __name__ == "__main__":
    fns = [
        test_igf_module_construction_clones_heads_and_builds_stage2_rate_embed,
        test_igf_module_image_embed_kwarg_is_ignored,
        test_igf_get_rho_inference_default_and_override,
        test_igf_sample_rho_in_range,
        test_igf_compute_distill_smoke,
        test_igf_compute_loss_matches_direct_loss,
        test_igf_rate_embed_output_is_256_channels_at_input_resolution,
    ]
    for fn in fns:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            raise
        else:
            print(f"OK   {fn.__name__}")
    print("All IGFModule tests passed.")

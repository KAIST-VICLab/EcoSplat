"""Smoke tests for ecosplat_wrapper primitives. CPU-only, no host deps."""

import torch
from torch import nn

from ecosplat_wrapper import (
    ImportanceEmbedding,
    LossImportanceOpacity,
    LossImportanceOpacityCfg,
    LossImportanceOpacityCfgWrapper,
    inject_shallow_add,
    make_rate_embed,
    plgc_protect_rate,
    relock,
    wrap_for_igf,
)


def test_imports():
    pass  # all imports above succeeded if this runs


def test_wrap_for_igf_freeze_and_clone():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.param_head = nn.Linear(8, 8)
            self.bn = nn.BatchNorm2d(8)
            self.backbone = nn.Linear(4, 4)

    m = Toy()
    wrap_for_igf(m, ["param_head"], freeze_bn=True)

    assert hasattr(m, "merge_param_head")
    assert m.merge_param_head.weight.requires_grad is True
    assert m.merge_param_head.bias.requires_grad is True
    assert m.param_head.weight.requires_grad is False
    assert m.param_head.bias.requires_grad is False
    assert m.backbone.weight.requires_grad is False

    # freeze_bn: BN is in eval after .train() if not under a trainable prefix
    m.train()
    assert m.bn.training is False

    # post-wrap addition stays trainable by default
    m.late = nn.Linear(4, 4)
    assert m.late.weight.requires_grad is True


def test_relock_recovers_from_post_wrap_drift():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.param_head = nn.Linear(8, 8)

    m = Toy()
    wrap_for_igf(m, ["param_head"])
    # simulate accidental unfreeze
    for p in m.param_head.parameters():
        p.requires_grad = True
    relock(m)
    assert all(not p.requires_grad for p in m.param_head.parameters())
    assert all(p.requires_grad for p in m.merge_param_head.parameters())


def test_importance_embedding_zero_init_yields_zero():
    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_merger = nn.Conv2d(3, 16, 1)

    host = Host()
    emb = ImportanceEmbedding.from_color_projector(host, "input_merger", zero_init=True)

    image = torch.randn(2, 3, 3, 32, 32)  # B=2, V=3, C=3, H=32, W=32
    out = emb(0.3, image)
    assert out.shape == (2 * 3, 16, 32, 32)
    assert torch.all(out == 0)


def test_importance_embedding_shape_matches_color_skip():
    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            # PatchEmbed-like: stride=patch_size
            self.rgb_embed = nn.Conv2d(3, 64, kernel_size=8, stride=8)

    host = Host()
    emb = ImportanceEmbedding.from_color_projector(host, "rgb_embed", zero_init=True)
    image = torch.randn(1, 2, 3, 64, 64)
    out = emb(0.5, image)
    # rgb_embed on (B*V=2, 3, 64, 64) → (2, 64, 8, 8)
    assert out.shape == (2, 64, 8, 8)


def test_inject_shallow_add_is_identity_when_R_zero():
    feat = torch.randn(2, 16, 8, 8)
    R = torch.zeros_like(feat)
    out = inject_shallow_add(feat, R)
    assert torch.allclose(out, feat)


def test_make_rate_embed_zero_init_clears_weights():
    src = nn.Conv2d(3, 16, 7, padding=3)
    src.weight.data.fill_(0.5)
    src.bias.data.fill_(0.5)

    class Host(nn.Module):
        pass
    host = Host()
    host.proj = src

    rate = make_rate_embed(host, "proj", zero_init=True)
    assert torch.all(rate.weight == 0)
    assert torch.all(rate.bias == 0)
    # original untouched
    assert torch.all(src.weight == 0.5)


def test_plgc_decay_schedule():
    import random
    random.seed(0)
    # at step 0, k_min should be 0.85
    rates0 = [plgc_protect_rate(0) for _ in range(50)]
    assert min(rates0) >= 0.85 - 1e-6 and max(rates0) <= 0.95 + 1e-6
    # far past, k_min hits floor 0.05
    rates_far = [plgc_protect_rate(100_000) for _ in range(50)]
    assert min(rates_far) >= 0.05 - 1e-6 and max(rates_far) <= 0.95 + 1e-6


def test_loss_skips_when_no_mask():
    cfg = LossImportanceOpacityCfgWrapper(
        importance_opacity=LossImportanceOpacityCfg(weight=0.1)
    )
    loss_fn = LossImportanceOpacity(cfg)

    class FakeOut:
        alpha = torch.rand(1, 4, 8, 8).clamp(1e-3, 1 - 1e-3)
    out = FakeOut()
    val = loss_fn({}, out, out)
    assert val.item() == 0.0


def test_loss_arithmetic_matches_legacy():
    """Legacy: loss = cfg.weight * (0.1 * BCE_io + BCE_acc). Defaults reproduce it."""
    import torch.nn.functional as F

    cfg = LossImportanceOpacityCfgWrapper(
        importance_opacity=LossImportanceOpacityCfg(weight=0.1)
    )
    loss_fn = LossImportanceOpacity(cfg)

    pred = torch.rand(2, 4, 16).clamp(1e-3, 1 - 1e-3)
    gt = torch.rand(2, 4, 16).clamp(1e-3, 1 - 1e-3)
    rendered = torch.rand(2, 4, 16).clamp(1e-3, 1 - 1e-3)

    class FakeOut:
        pass
    out = FakeOut()
    out.alpha = rendered
    distill = {"importance_mask": gt, "pred_opacity": pred}
    got = loss_fn(distill, out, out)

    expected = 0.1 * (0.1 * F.binary_cross_entropy(pred, gt) + F.binary_cross_entropy(rendered, torch.ones_like(rendered)))
    assert torch.allclose(got, expected, atol=1e-7), f"{got=} vs {expected=}"


if __name__ == "__main__":
    fns = [
        test_imports,
        test_wrap_for_igf_freeze_and_clone,
        test_relock_recovers_from_post_wrap_drift,
        test_importance_embedding_zero_init_yields_zero,
        test_importance_embedding_shape_matches_color_skip,
        test_inject_shallow_add_is_identity_when_R_zero,
        test_make_rate_embed_zero_init_clears_weights,
        test_plgc_decay_schedule,
        test_loss_skips_when_no_mask,
        test_loss_arithmetic_matches_legacy,
    ]
    for fn in fns:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            raise
        else:
            print(f"OK   {fn.__name__}")
    print("All smoke tests passed.")

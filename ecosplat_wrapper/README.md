# ecosplat_wrapper

The **IGF (Importance-Guided Fusion) stage-2 training strategy** from
[EcoSplat](https://github.com/KAIST-VICLab/EcoSplat), packaged as a
host-agnostic add-on. It turns any feed-forward 3D Gaussian Splatting model into
an **efficiency-controllable** one: a single scalar — the *protect rate* κ —
trades the number of rendered Gaussian primitives against reconstruction quality
at inference time, **without changing the base architecture**.

This package is **vendored identically** into every EcoSplat training branch
(`eco_spfsplat`, `eco_zpressor`). It is intentionally self-contained so you can
drop it into a new base model with ~5 edits — see
[Integrating into a new host](#integrating-into-a-new-host).

> EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from
> Multi-view Images (CVPR 2026).
> [Paper](https://arxiv.org/abs/2512.18692) ·
> [Project page](https://kaist-viclab.github.io/ecosplat-site/)

## Install

```bash
cd ecosplat_wrapper
pip install -e .
```

Install it into the base model's existing environment. Extra deps beyond
PyTorch: `einops`, `jaxtyping`.

## How it works (one paragraph)

IGF is a **second training stage** on top of a converged base model. It clones
the base model's Gaussian-parameter head(s) into parallel **merge heads**,
conditions them on a scalar protect-rate κ via a learned **rate embedding**
(shallow-added into the head features), and supervises the merged Gaussians with
an **importance-aware opacity loss** `L_io`. A pseudo-ground-truth **importance
mask** Ω is built per batch by bipartite-soft-matching the low-variation regions
and projecting the merged 3D centers back onto each view. During training κ is
sampled by a **PLGC schedule** (progressively compacting over steps); at
inference you fix κ to dial primitive count up or down. The base model's original
heads stay intact, so κ → 1 recovers the base model.

## Public API

```python
from ecosplat_wrapper import (
    IGFConfig,                 # dataclass of all hyperparameters
    IGFModule,                 # nn.Module: merge heads + rate embed + L_io
    wrap_for_igf, relock,      # freeze helpers (clone heads in-place, freeze base/BN)
    generate_importance_mask,  # build Ω (pseudo-GT importance mask)
    plgc_protect_rate,         # PLGC κ sampler
    LossImportanceOpacity,     # L_io loss module
)
```

### `IGFConfig`

| field | default | meaning |
|---|---|---|
| `loss_weight` | `0.1` | outer scale on `L_io` |
| `io_weight` | `0.1` | λ on `BCE(predicted opacity, Ω)` (paper Eq. 9) |
| `acc_weight` | `1.0` | weight on `BCE(rendered alpha, 1)` accumulation term |
| `plgc_min0` / `plgc_max` | `0.85` / `0.95` | initial κ sampling range during training |
| `plgc_decay` / `plgc_decay_steps` | `0.05` / `1000` | lower bound decays by `plgc_decay` every `plgc_decay_steps` |
| `plgc_floor` | `0.05` | hard floor for the sampled lower bound |
| `zero_init_rate_embed` | `True` | zero-init the rate embedding (no-op at start) |
| `inference_rho` | `0.4` | fixed κ used at eval/inference |

`inference_rho` is the eval-time compaction knob (paper κ_i). The published
**inference** release exposes a related knob, `model.encoder.primitive_ratio`,
on its bespoke `encoder_ecosplat` — that one distributes the primitive budget
per view (high-frequency-weighted), whereas this wrapper applies a uniform rate.

### `IGFModule`

Construct it with the head(s) you want cloned; it deep-copies them into
`merge_heads` and builds a `3→256` conv `rate_embed`:

```python
igf = IGFModule(
    heads_to_clone=[encoder.gaussian_param_head, encoder.gaussian_param_head2],
    igf_cfg=IGFConfig(**dict(cfg.igf)),
)
```

Primitives the host calls:

- `igf.get_rho(global_step, training, override=None) -> float` — κ for this step
  (PLGC-sampled when `training`, else `inference_rho`, unless `override` is given).
- `igf.rate_embed(rho_3ch) -> feat` — encode a `(B·V, 3, H, W)` κ-broadcast into a
  256-channel feature map to shallow-add into head features.
- `igf.merge_heads[i](...)` — run the cloned head(s) on the rate-injected features.
- `igf.compute_distill(ori_gaussians=, image=, intrinsics=, extrinsics=, protect_rate=, depth=None) -> dict`
  — build Ω and the KLD-merged covariance (wraps `generate_importance_mask`).
- `igf.compute_loss(distill_infos, output, ori_output) -> Tensor` — `L_io`.

## Integrating into a new host

IGF needs **5 touch points** in your base model. The two shipped branches are
worked references — read them side by side:

| step | SPFSplat (`eco_spfsplat`) | ZPressor / MVSplat (`eco_zpressor`) |
|---|---|---|
| 1. config field | `src/model/encoder/encoder_spfsplat.py` | `mvsplat/src/model/encoder/encoder_costvolume.py` |
| 2. build module | same file, `__init__` | same file, `__init__` (+ `stage1_weights_path`, `freeze_pretrained`) |
| 3. rate inject | `src/model/encoder/heads/dpt_gs_head.py` (`path_1 = path_1 + rate_feat`) | `.../costvolume/depth_predictor_multiview.py` (via `igf_rate_proj`) |
| 4. merge heads | `encoder_spfsplat.py` forward | `encoder_costvolume.py` forward |
| 5. loss | `src/model/model_wrapper.py` | `mvsplat/src/model/model_wrapper.py` |

**1. Config field** — add to your encoder config. `None` disables IGF (stage-1 /
baseline); a dict enables stage-2 (empty `{}` = `IGFConfig` defaults).

```python
igf: Optional[dict] = None
```

**2. Build the module** in your encoder `__init__`:

```python
self.igf = None
if cfg.igf is not None:
    from ecosplat_wrapper import IGFConfig, IGFModule
    self.igf = IGFModule(
        heads_to_clone=[self.<gaussian_head>],   # head(s) producing Gaussian params
        igf_cfg=IGFConfig(**dict(cfg.igf)),
    )
    for p in self.igf.parameters():
        p.requires_grad = True
# Optional: load a converged stage-1 checkpoint first, then freeze the base so
# only IGF trains (see eco_zpressor's `stage1_weights_path` + `freeze_pretrained`,
# or use wrap_for_igf/relock below).
```

**3. Sample κ and inject the rate feature** in `forward`:

```python
igf_active = self.igf is not None
if igf_active:
    rho = self.igf.get_rho(global_step, self.training,
                           override=context.get("protect_rate"))
    rho_3ch = torch.ones(b * v, 3, h, w, device=device, dtype=img.dtype) * rho
    rate_feat = self.igf.rate_embed(rho_3ch).view(b, v, -1, h, w)
```

Then **shallow-add** `rate_feat` into the per-pixel features that feed your
Gaussian head (paper Eq. 3 / Fig. 6c). If the channel counts differ, project
first (ZPressor uses a `1×1` `igf_rate_proj: 256 → feat_dim`).

**4. Run the merge head(s)** on the rate-injected features to produce the merged
Gaussians, building the output through your usual Gaussian adapter.

**5. Distill + loss** — stash `distill_infos = self.igf.compute_distill(...)` in
the encoder, then add the loss in your training step / model wrapper:

```python
if getattr(self.encoder, "igf", None) is not None:
    total_loss = total_loss + self.encoder.igf.compute_loss(distill_infos, output, None)
```

### `compute_distill` tensor conventions

- `image (B,V,3,H,W)`, `intrinsics (B,V,3,3)` **normalized**, `extrinsics (B,V,4,4)`
  **camera-to-world** (pixelSplat / OpenCV convention).
- `ori_gaussians` may expose `.means` / `.covariances`, or be a dict with
  `means_pix` / `cov_pix`, or `pts3d` / `cov_feat`.
- `depth (B,V,H,W)` is optional — derived from `extrinsics` when omitted.

### Two ways to freeze for stage-2

- **`IGFModule`** (used by both shipped branches): the module *owns* the cloned
  merge heads; you set `requires_grad` on the base yourself.
- **`wrap_for_igf(model, head_attrs=[...])` / `relock(model)`**: clones the named
  heads onto the model in-place as `merge_<name>`, sets `requires_grad=True` only
  for the `merge_` prefixes, and (optionally) freezes BatchNorm in eval mode —
  handy when you want the merge heads to live on the model rather than in a
  side-module.

## Tests

```bash
cd ecosplat_wrapper
pytest tests/        # test_igf_module.py, test_smoke.py
```

## Keeping the vendored copies in sync

`ecosplat_wrapper/` is duplicated across training branches and **must stay
byte-identical**. To check:

```bash
git diff eco_spfsplat eco_zpressor -- ecosplat_wrapper
# (empty output = in sync)
```

When you change the wrapper on one branch, copy the folder verbatim to the other
before pushing.

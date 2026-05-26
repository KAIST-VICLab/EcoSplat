# Integrating `ecosplat_wrapper` into ZPressor (MVSplat baseline)

This documents exactly which MVSplat files we changed to add EcoSplat's IGF
efficiency control. For the base-agnostic API and the generic recipe, see
[`ecosplat_wrapper/README.md`](ecosplat_wrapper/README.md). IGF is toggled by the
encoder's `igf` config — when it is unset, every change below is inert and the
model behaves like stock ZPressor-MVSplat.

```bash
pip install -e ecosplat_wrapper
```

Paths are relative to the repo root; the MVSplat baseline lives under `mvsplat/`.

## Setup: config + module

**Config switches** — `mvsplat/src/model/encoder/encoder_costvolume.py:68-70`
```python
igf: Optional[dict] = None                  # IGF stage-2 cfg; None disables
stage1_weights_path: Optional[str] = None   # load a stage-1 ckpt before IGF init
freeze_pretrained: bool = False             # True = freeze all except IGF (SPFSplat-style)
```

**Build the module + rate projection** (encoder `__init__`) — `encoder_costvolume.py:145-199`
```python
if cfg.igf is not None:
    if cfg.stage1_weights_path:                       # load converged ZPressor stage-1 weights
        self.load_state_dict(torch.load(cfg.stage1_weights_path, ...), strict=False)
    from ecosplat_wrapper import IGFConfig, IGFModule
    self.igf = IGFModule(
        heads_to_clone=[self.depth_predictor.to_gaussians,
                        self.depth_predictor.to_disparity],
        igf_cfg=IGFConfig(**dict(cfg.igf)),
    )
    self.igf_rate_proj = nn.Conv2d(256, cfg.depth_unet_feat_dim, kernel_size=1)
    if cfg.freeze_pretrained:                          # only IGF + igf_rate_proj train
        ...
```
The `to_gaussians` and `to_disparity` heads are cloned into `igf.merge_heads`.
Because the cost-volume features aren't 256-ch, we add `igf_rate_proj` (1×1 conv)
to map the wrapper's 256-ch `rate_embed` output down to `depth_unet_feat_dim`.

## Forward pass

**1. Sample κ, encode + project the rate feature** — `encoder_costvolume.py:270-303`
```python
igf_active = self.igf is not None
if igf_active:
    protect_rate    = self.igf.get_rho(global_step, self.training,
                                       override=context.get("protect_rate"))
    rate_feat       = self.igf.rate_embed(rho_3ch)          # (vb, 256, h, w)
    rate_feat_proj  = self.igf_rate_proj(rate_feat)         # (vb, depth_unet_feat_dim, h, w)
    merge_head      = self.igf.merge_heads[0]               # cloned to_gaussians
    merge_disparity = self.igf.merge_heads[1]               # cloned to_disparity
# ...passed into self.depth_predictor(...)
```

**2. Shallow-add + run the cloned heads** (depth predictor) — `mvsplat/src/model/encoder/costvolume/depth_predictor_multiview.py:358-436`
```python
# merged GAUSSIANS — shallow-add κ into refine_out, then run the cloned `to_gaussians`
refine_out_m = refine_out
if merge_head is not None:
    if rate_feat_proj is not None:
        refine_out_m = refine_out_m + rate_feat_proj             # IGF shallow-add (Eq. 3 / Fig. 6c)
    merged_in = torch.cat([refine_out_m, images, proj_feat_in_fullres], dim=1)
    merged_raw_gaussians = merge_head(merged_in)                 # cloned `to_gaussians`

# merged DENSITY/opacity — cloned `to_disparity` on the same rate-injected features
if merge_disparity is not None and not self.wo_depth_refine:
    merged_delta_disps_density = merge_disparity(refine_out_m)   # -> merged opacity (feeds L_io)
```
`merge_head` (cloned `to_gaussians`) yields the merged Gaussian *params*; `merge_disparity`
(cloned `to_disparity`) yields the merged *density/opacity*. Both consume the same
rate-injected `refine_out_m`. The encoder stashes a `self.last_igf` side-channel
(original Gaussians, distill infos, protect_rate) for the loss.

## Training

**3. L_io loss + protect-rate logging** (model wrapper) — `mvsplat/src/model/model_wrapper.py:141-188`
```python
igf_active = getattr(self.encoder, "igf", None) is not None
output = self.encoder(..., render_alpha=igf_active)              # need rendered alpha for L_io
last_igf = getattr(self.encoder, "last_igf", None)
if last_igf is not None:
    ori_output = self.decoder(last_igf["ori_gaussians"], ...)    # render stage-1 gaussians (L_acc)
    igf_loss = self.encoder.igf.compute_loss(last_igf["distill_infos"], output, ori_output)
    total_loss = total_loss + igf_loss
```

**4. Higher LR for IGF params** (optimizer) — `mvsplat/src/model/model_wrapper.py:54,673-691`
```python
igf_lr_multiplier: float = 10.0
# params whose name contains ".igf." or "igf_rate_proj"  ->  lr * igf_lr_multiplier; rest -> base lr
```

## Files changed
| File | Forward / Training | What we added |
| --- | --- | --- |
| `mvsplat/src/model/encoder/encoder_costvolume.py` | both | `igf`/`stage1_weights_path`/`freeze_pretrained` cfg, `IGFModule` + `igf_rate_proj` init, κ + rate_feat, merge-head selection, `last_igf` side-channel |
| `mvsplat/src/model/encoder/costvolume/depth_predictor_multiview.py` | forward | `rate_feat_proj` shallow-add + cloned `to_gaussians` / `to_disparity` |
| `mvsplat/src/model/model_wrapper.py` | training | `render_alpha`, `compute_loss` → `L_io`, `igf_lr_multiplier` param-group split |

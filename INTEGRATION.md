# Integrating `ecosplat_wrapper` into SPFSplat

This documents exactly which SPFSplat files we changed to add EcoSplat's IGF
efficiency control. For the base-agnostic API and the generic recipe, see
[`ecosplat_wrapper/README.md`](ecosplat_wrapper/README.md). IGF is toggled by the
encoder's `igf` config — when it is unset, every change below is inert and the
model behaves like stock SPFSplat.

```bash
pip install -e ecosplat_wrapper
```

All paths are relative to the repo root.

## Setup: config + module

**Config switch** — `src/model/encoder/encoder_spfsplat.py:63`
```python
igf: Optional[dict] = None   # ecosplat IGF stage-2 finetune cfg; None disables
```
A dict (even empty `{}`) enables IGF and is passed to `IGFConfig(**cfg.igf)`.

**Build the module** (encoder `__init__`) — `encoder_spfsplat.py:109-132`
```python
self.igf = None
if cfg.igf is not None:
    # load the converged stage-1 SPFSplat checkpoint (strip the "encoder." prefix)
    stage1 = torch.load(cfg.pretrained_weights, map_location='cpu')['state_dict']
    self.load_state_dict({k[len('encoder.'):]: v for k, v in stage1.items()
                          if k.startswith('encoder.')})
    from ecosplat_wrapper import IGFConfig, IGFModule
    self.igf = IGFModule(
        heads_to_clone=[self.gaussian_param_head, self.gaussian_param_head2],
        igf_cfg=IGFConfig(**dict(cfg.igf)),
    )
    for p in self.igf.parameters():
        p.requires_grad = True
```
When IGF is enabled the encoder first loads the stage-1 checkpoint from
`cfg.pretrained_weights`, then deep-clones both Gaussian-parameter heads into
`igf.merge_heads` and builds the `3→256` `rate_embed`. (`:205-208` asserts `igf`
is set when training — stage-2 is the only training mode here.)

## Forward pass

**1. Sample κ, build the rate feature** — `encoder_spfsplat.py:232-241`
```python
igf_active = self.igf is not None
if igf_active:
    override = context.get("protect_rate")
    protect_rate = self.igf.get_rho(global_step, self.training, override=override)
    rho_3ch = torch.ones(b * v_cxt, 3, h, w, ...) * protect_rate
    rate_feat = self.igf.rate_embed(rho_3ch).view(b, v_cxt, -1, h, w)   # 256-ch
```
Train: PLGC-sampled κ. Eval: `cfg.inference_rho` (default 0.4) or a
`context["protect_rate"]` override.

**2. Shallow-add the rate feature into the head** — `src/model/encoder/heads/dpt_gs_head.py:41,74-75`
```python
def forward(self, encoder_tokens, imgs, image_size=None, conf=None, rate_feat=None):
    ...
    if rate_feat is not None:
        path_1 = path_1 + rate_feat        # IGF Shallow Add (paper Eq. 3 / Fig. 6c)
```
We added a `rate_feat` kwarg to the DPT GS head and add it into the first
upsampling path.

**3. Run the cloned merge heads** — `encoder_spfsplat.py:262-278`
```python
GS_res1 = self.gaussian_param_head(...)                       # original (stage-1) head
if igf_active:
    GS_res1_m = self.igf.merge_heads[0](..., rate_feat=rate_feat[:, 0])
    GS_res2_m = self.igf.merge_heads[1](..., rate_feat=rate_feat[:, i])
```
The original heads still run; the cloned heads run on the same tokens but
rate-conditioned, producing the merged Gaussians.

**4. Build merged Gaussians + importance mask** — `encoder_spfsplat.py:337-425`
```python
if igf_active:
    # merged Gaussians from the cloned-head params; top-k keeps k = int(max(protect_rate, 1/16)*h*w)
    encoder_output["gaussians"] = _flatten_gaussians(merged_gaussians)   # the set that gets rendered
    if self.training:                                                    # distill is training-only
        distill_infos = self.igf.compute_distill(
            ori_gaussians={"means_pix": means_pix, "cov_pix": cov_pix},
            image=context["image"], intrinsics=context["intrinsics"],
            extrinsics=context_extrinsics, protect_rate=protect_rate, depth=depths_per_view)
        distill_infos["pred_opacity"] = merged_opacities_per_pixel
        encoder_output["distill_infos"] = distill_infos
        encoder_output["protect_rate"]  = protect_rate
```

## Training

**5. Add the L_io loss** (model wrapper) — `src/model/model_wrapper.py:224,270-277`
```python
distill_infos = encoder_output.get("distill_infos")
...
if distill_infos is not None and getattr(self.encoder, "igf", None) is not None:
    igf_loss = self.encoder.igf.compute_loss(distill_infos, output, None)
    self.log("loss/igf", igf_loss)
    total_loss = total_loss + igf_loss
```
The rendered `output.alpha` and the importance mask in `distill_infos` feed the
importance-aware opacity loss; nothing else in the training loop changes.

## Files changed
| File | Forward / Training | What we added |
| --- | --- | --- |
| `src/model/encoder/encoder_spfsplat.py` | both | `igf` cfg, `IGFModule` init, κ + `rate_feat`, merge-head calls, `compute_distill` |
| `src/model/encoder/heads/dpt_gs_head.py` | forward | `rate_feat` kwarg + shallow-add |
| `src/model/model_wrapper.py` | training | `compute_loss` → `L_io` added to the total loss |

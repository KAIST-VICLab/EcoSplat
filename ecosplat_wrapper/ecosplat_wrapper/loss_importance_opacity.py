from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LossImportanceOpacityCfg:
    weight: float                # outer scale (matches existing chamfer_distance.yaml weight: 0.1)
    io_weight: float = 0.1       # λ_io inner factor on BCE(α̃, Ω) — paper Eq. 9
    acc_weight: float = 1.0      # weight on BCE(rendered_alpha, 1) accumulation term


@dataclass
class LossImportanceOpacityCfgWrapper:
    importance_opacity: LossImportanceOpacityCfg


class LossImportanceOpacity(nn.Module):
    """Importance-aware Opacity Loss (paper Eq. 9, L_io).

    BCE between predicted per-pixel opacities {α̃_{i,j}} and the pseudo-GT
    importance mask Ω_i, plus the rendered-alpha-to-1 BCE term that the
    SPFSplat adaptation pairs with it. Default factors `io_weight=0.1`,
    `acc_weight=1.0` reproduce the prior `LossChamferDistance.forward`
    arithmetic exactly: `cfg.weight * (0.1 * l_io + 1.0 * l_acc)`.

    Cfg extraction follows SPFSplat's `Loss` pattern so the host's loss
    registry can instantiate this class identically to existing losses.
    """

    cfg: LossImportanceOpacityCfg
    name: str

    def __init__(self, cfg_wrapper: LossImportanceOpacityCfgWrapper) -> None:
        super().__init__()
        (field,) = fields(type(cfg_wrapper))
        self.cfg = getattr(cfg_wrapper, field.name)
        self.name = field.name

    def forward(self, distill_infos: dict, output, ori_output) -> Tensor:
        # Accept paper-faithful keys (`importance_mask`, `pred_opacity`) or the
        # legacy SPFSplat-adapted keys (`alpha`, `learn_alpha`) for migration.
        gt_mask = distill_infos.get("importance_mask")
        if gt_mask is None:
            gt_mask = distill_infos.get("alpha")
        pred_op = distill_infos.get("pred_opacity")
        if pred_op is None:
            pred_op = distill_infos.get("learn_alpha")

        if gt_mask is None or pred_op is None:
            return torch.tensor(0.0, device=output.alpha.device)

        gt_mask = gt_mask.detach()
        rendered = output.alpha

        l_io = F.binary_cross_entropy(pred_op, gt_mask)
        l_acc = F.binary_cross_entropy(rendered, torch.ones_like(rendered))
        inner = self.cfg.io_weight * l_io + self.cfg.acc_weight * l_acc
        return self.cfg.weight * torch.nan_to_num(inner, nan=0.0)

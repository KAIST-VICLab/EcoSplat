from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_reproj import LossReproj, LossReprojCfgWrapper
from .loss_chamfer_distance import LossChamferDistance, LossChamferDistanceCfgWrapper


LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossReprojCfgWrapper: LossReproj,
    LossChamferDistanceCfgWrapper: LossChamferDistance,

}

LossCfgWrapper =  LossLpipsCfgWrapper | LossMseCfgWrapper  | LossReprojCfgWrapper | LossChamferDistanceCfgWrapper   


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]

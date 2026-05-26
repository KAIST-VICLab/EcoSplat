from .igf_config import IGFConfig
from .igf_module import IGFModule
from .loss_importance_opacity import (
    LossImportanceOpacity,
    LossImportanceOpacityCfg,
    LossImportanceOpacityCfgWrapper,
)
from .mask import generate_importance_mask
from .plgc import plgc_protect_rate
from .wrap import relock, wrap_for_igf

# Legacy helpers (kept for hosts already wired to the older API).
from .importance_embedding import (
    ImportanceEmbedding,
    assert_rate_matches_color,
    inject_shallow_add,
    make_rate_embed,
)

__all__ = [
    "IGFConfig",
    "IGFModule",
    "wrap_for_igf",
    "relock",
    "plgc_protect_rate",
    "LossImportanceOpacity",
    "LossImportanceOpacityCfg",
    "LossImportanceOpacityCfgWrapper",
    "generate_importance_mask",
    "ImportanceEmbedding",
    "make_rate_embed",
    "inject_shallow_add",
    "assert_rate_matches_color",
]

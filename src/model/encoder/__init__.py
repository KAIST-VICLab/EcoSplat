from typing import Optional

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer
from .encoder_ecosplat import EncoderEcoSplatCfg, EncoderEcoSplat


ENCODERS = {
    "ecosplat": (EncoderEcoSplat, None),
}

EncoderCfg = EncoderEcoSplatCfg 

def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer

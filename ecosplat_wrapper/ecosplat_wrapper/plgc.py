import random


def plgc_protect_rate(
    global_step: int,
    *,
    k_min0: float = 0.85,
    floor: float = 0.05,
    k_max: float = 0.95,
    decay: float = 0.05,
    decay_steps: int = 1000,
) -> float:
    """Progressive Learning on Gaussian Compaction sampler (paper Eq. 11).

    Identical arithmetic to encoder_spfsplat.py:567-572.
    """
    k_min = max(k_min0 - (global_step // decay_steps) * decay, floor)
    return random.uniform(k_min, k_max)

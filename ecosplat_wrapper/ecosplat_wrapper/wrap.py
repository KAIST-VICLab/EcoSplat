import copy
from typing import Iterable

from torch import nn


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def _matches_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(p in name for p in prefixes)


def _apply_bn_eval(model: nn.Module, trainable_prefixes: Iterable[str]) -> None:
    for mname, m in model.named_modules():
        if _matches_prefix(mname, trainable_prefixes):
            continue
        if isinstance(m, _BN_TYPES):
            m.eval()


def _patch_train(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> None:
    """Override `model.train(mode)` so frozen BN stays in eval mode after each call."""
    if getattr(model, "_igf_train_patched", False):
        model._igf_trainable_prefixes = trainable_prefixes  # refresh
        return
    orig_train = model.train

    def patched_train(mode: bool = True):
        orig_train(mode)
        if mode:
            _apply_bn_eval(model, model._igf_trainable_prefixes)
        return model

    model._igf_trainable_prefixes = trainable_prefixes
    model.train = patched_train
    model._igf_train_patched = True


def wrap_for_igf(
    model: nn.Module,
    head_attrs: list[str],
    trainable_prefixes: tuple[str, ...] = ("merge_",),
    freeze_bn: bool = True,
) -> nn.Module:
    for name in head_attrs:
        if not hasattr(model, name):
            raise AttributeError(
                f"{type(model).__name__} has no submodule '{name}' to clone"
            )
        setattr(model, f"merge_{name}", copy.deepcopy(getattr(model, name)))

    for pname, p in model.named_parameters():
        p.requires_grad = _matches_prefix(pname, trainable_prefixes)

    if freeze_bn:
        _apply_bn_eval(model, trainable_prefixes)
        _patch_train(model, trainable_prefixes)
    return model


def relock(
    model: nn.Module,
    trainable_prefixes: tuple[str, ...] = ("merge_",),
    freeze_bn: bool = True,
) -> nn.Module:
    for pname, p in model.named_parameters():
        p.requires_grad = _matches_prefix(pname, trainable_prefixes)
    if freeze_bn:
        _apply_bn_eval(model, trainable_prefixes)
        _patch_train(model, trainable_prefixes)
    return model

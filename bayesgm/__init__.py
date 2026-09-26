"""Subset of the ``bayesgm`` package required by BayesNDE."""
from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "1.0.2+bayesnde"

if TYPE_CHECKING:
    from . import datasets, models
    from .datasets import Base_sampler, GMM_indep_sampler, Swiss_roll_sampler
    from .models import BGM, MNISTBGM

_SYMBOL_TO_MODULE = {
    "BGM": "bayesgm.models",
    "MNISTBGM": "bayesgm.models",
    "Base_sampler": "bayesgm.datasets",
    "Gaussian_sampler": "bayesgm.datasets",
    "GMM_indep_sampler": "bayesgm.datasets",
    "Swiss_roll_sampler": "bayesgm.datasets",
}

_MODULE_ATTRIBUTES = {
    "models": "bayesgm.models",
    "datasets": "bayesgm.datasets",
}

__all__ = [
    "BGM",
    "MNISTBGM",
    "Base_sampler",
    "Gaussian_sampler",
    "GMM_indep_sampler",
    "Swiss_roll_sampler",
]


def __getattr__(name):
    if name in _SYMBOL_TO_MODULE:
        module = import_module(_SYMBOL_TO_MODULE[name])
        return getattr(module, name)
    if name in _MODULE_ATTRIBUTES:
        return import_module(_MODULE_ATTRIBUTES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_MODULE_ATTRIBUTES))

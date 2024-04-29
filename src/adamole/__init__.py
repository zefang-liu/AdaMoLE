"""
AdaMoLE Initialization
"""
from .config import AdaMoleConfig
from .layer import AdaMoleLayer, LinearAdaMoleLayer
from .model import AdaMoleModel

__all__ = ["AdaMoleConfig", "AdaMoleLayer", "LinearAdaMoleLayer", "AdaMoleModel"]


def __getattr__(name):
    raise AttributeError(f"Module {__name__} has no attribute {name}.")

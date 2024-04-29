"""
MoLE Initialization
"""
from .config import MoleConfig
from .layer import MoleLayer, LinearMoleLayer
from .model import MoleModel

__all__ = ["MoleConfig", "MoleLayer", "LinearMoleLayer", "MoleModel"]


def __getattr__(name):
    raise AttributeError(f"Module {__name__} has no attribute {name}.")

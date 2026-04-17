"""TAAN-GRPO: Type-Aware Advantage Normalization for multi-task RL training."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("taan-grpo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

__all__ = ["__version__"]

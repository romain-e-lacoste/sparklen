# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Sparklen: statistical learning toolkit for high-dimensional Hawkes processes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparklen")
except PackageNotFoundError:  # source tree that was never installed
    __version__ = "0.0.0"

__all__ = ["__version__"]

"""Headless command line interface for FeAture Explorer.

The package only depends on the modelling code under ``BC/`` and ``SA/`` so it
can run on a Linux server without a display, PySide6 or pyradiomics.
"""

from HomeUI.VersionConstant import VERSION

__version__ = VERSION

__all__ = ["VERSION", "__version__"]

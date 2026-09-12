"""PACE core: typed storyboard specification, script breakdown, solved camera
geometry, Blender staging and self-auditing checks."""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    # One source of truth: the installed distribution's own metadata. A
    # literal here drifts from pyproject silently, and it did: 0.1.3 shipped
    # reporting 0.1.2.
    __version__ = _version("pace-core")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

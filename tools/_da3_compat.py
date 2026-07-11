"""Make `depth_anything_3` importable for CPU-only ONNX export.

Depth Anything 3 declares heavy optional dependencies (xformers, open3d, moviepy,
pycolmap, e3nn, evo, ...) that are only needed for its 3D/Gaussian/visualisation
export paths — none of which the ONNX depth export touches. Several have no wheels
on macOS arm64 (notably xformers), so a full `pip install -e .` fails there.

`ensure_da3_importable()` inserts lightweight stub modules for the optional deps
**only when they are not already importable**, so it is a no-op in a full install
(e.g. Linux CI with xformers present) and a rescue on a curated `--no-deps` install.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock

# Optional modules pulled in transitively by depth_anything_3.api's top-level imports
# (3D export / pose / visualisation) that the ONNX depth path never executes.
_OPTIONAL = [
    "moviepy", "moviepy.editor",
    "open3d", "open3d.visualization",
    "trimesh",
    "pycolmap",
    "e3nn", "e3nn.o3",
    "plyfile",
    "gsplat",
    "fastapi", "uvicorn",
    "evo", "evo.core", "evo.core.trajectory", "evo.core.lie_algebra", "evo.core.metrics",
    "pillow_heif",
    "xformers", "xformers.ops",
]


def _stub(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: MagicMock()  # type: ignore[attr-defined]
    mod.__path__ = []  # mark as package so submodule imports resolve
    sys.modules[name] = mod


def ensure_da3_importable() -> list[str]:
    """Stub any optional dep that is missing. Returns the list of stubbed names."""
    stubbed = []
    for name in _OPTIONAL:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue  # genuinely installed; leave it alone
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
        _stub(name)
        stubbed.append(name)
    return stubbed

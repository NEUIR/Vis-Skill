"""
Gemma-tree `lib` package.

`config` is a local module (points at the Gemma model + the Qwen-tree inputs).
Every other lib module is re-exported from the Qwen tree unchanged, so the
Gemma scripts speak the same API as the Qwen scripts without duplicating the
helpers.

If you `from lib.cells import load_state`, Python resolves `lib.cells` to the
Qwen file via the loader below, and that file in turn does
`from lib.config import ...` which resolves to THIS package's config.py --
so paths stay Gemma-side while helpers stay Qwen-side.
"""
from __future__ import annotations
import importlib.util
import os
import sys

_QWEN_LIB = os.environ.get(
    "MECH_QWEN_LIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "alfworld-qwen3.5-9b", "mechanism", "lib"),
)

_MODS = ("cells", "prompts_bridge", "stats", "position_utils")


def _install_shim(name):
    src = os.path.join(_QWEN_LIB, f"{name}.py")
    if not os.path.isfile(src):
        return
    spec = importlib.util.spec_from_file_location(f"lib.{name}", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"lib.{name}"] = mod
    spec.loader.exec_module(mod)


# Submodules are registered eagerly so `from lib.cells import ...` works whether
# the caller imports the package first or the submodule directly.
for _n in _MODS:
    try:
        _install_shim(_n)
    except Exception:
        # A submodule failing to import here is fine unless the caller actually
        # asks for it -- deferring the raise to the import site gives clearer
        # tracebacks than swallowing every exception up top.
        pass

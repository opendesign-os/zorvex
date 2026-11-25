"""Load what zorvex.json names as ``path/to/file.py:name`` or ``package.module:name``.

One loader for the auth file, the custom HTTP app and the graphs.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from zorvex_api.config import config_dir


def load_object(spec: str, *, module_name: str) -> Any:
    """The object *spec* names.

    A path ending in ``.py`` or holding a ``/`` is a file, resolved from the
    config's directory and imported as *module_name*; anything else is an
    importable module. Raises ImportError when *spec* is malformed or names
    nothing; errors raised by the module's own code propagate.
    """
    path, sep, name = spec.rpartition(":")
    if not sep or not path or not name:
        raise ImportError(f"{spec!r} is not 'path/to/file.py:name' or 'package.module:name'")
    is_file = path.endswith(".py") or "/" in path or "\\" in path
    module = _import_file(Path(path), module_name) if is_file else importlib.import_module(path)
    try:
        return getattr(module, name)
    except AttributeError:
        raise ImportError(f"{path} has no attribute {name!r}") from None


def _import_file(path: Path, module_name: str) -> ModuleType:
    file_path = path if path.is_absolute() else (config_dir() / path).resolve()
    if not file_path.is_file():
        raise ImportError(f"{file_path} is not a file")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs, so its dataclasses, pydantic models and pickled objects can find it.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module

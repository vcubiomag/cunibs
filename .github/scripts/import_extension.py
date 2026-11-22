"""Load the installed solver extension without importing cunibs, which needs CuPy."""

import importlib.util
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

package = importlib.util.find_spec("cunibs")
assert package is not None and package.origin is not None
solver = Path(package.origin).parent / "solver"
(path,) = [p for s in EXTENSION_SUFFIXES if (p := solver / f"_solver_ext{s}").exists()]
spec = importlib.util.spec_from_file_location("_solver_ext", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"loaded {path.name}: {len(dir(module))} names")

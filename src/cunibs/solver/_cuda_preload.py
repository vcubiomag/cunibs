from __future__ import annotations

from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib

CUDA_LIBNAMES = ("cudart",)


def _preload() -> dict[str, str]:
    failures: dict[str, str] = {}
    for libname in CUDA_LIBNAMES:
        try:
            load_nvidia_dynamic_lib(libname)
        except (DynamicLibNotFoundError, OSError) as exc:
            failures[libname] = str(exc)
    return failures


_FAILURES = _preload()


def describe_failure(exc: ImportError) -> str:
    if not _FAILURES:
        return f"cunibs could not load its solver extension: {exc}"

    detail = "".join(f"\n  {name}: {reason}" for name, reason in _FAILURES.items())
    return (
        f"cunibs could not load its solver extension: {exc}\n"
        "Install the CUDA toolkit with `pip install 'cuda-toolkit[cudart]'`, or point "
        f"CUDA_HOME at an existing CUDA 13 toolkit.{detail}"
    )

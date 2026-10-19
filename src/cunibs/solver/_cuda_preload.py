from __future__ import annotations

from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib


def _preload() -> str | None:
    try:
        load_nvidia_dynamic_lib("cudart")
    except (DynamicLibNotFoundError, OSError) as exc:
        return str(exc)
    return None


_FAILURE = _preload()


def describe_failure(exc: ImportError) -> str:
    if _FAILURE is None:
        return f"cunibs could not load its solver extension: {exc}"
    return (
        f"cunibs could not load its solver extension: {exc}\n"
        "Install the CUDA toolkit with `pip install 'cuda-toolkit[cudart]'`, or point "
        f"CUDA_HOME at an existing CUDA 13 toolkit.\n  cudart: {_FAILURE}"
    )

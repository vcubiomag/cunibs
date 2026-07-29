"""Report the CUDA toolkit that cuda.pathfinder resolves, as CMake KEY=VALUE lines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def emit(key: str, value: str) -> None:
    print(f"{key}={value}")


def emit_path(key: str, value: str | Path) -> None:
    emit(key, Path(value).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libs", required=True, help="comma-separated, in dependency order")
    parser.add_argument("--emit-lib-paths", action="store_true")
    args = parser.parse_args()

    try:
        from cuda.pathfinder import (
            find_nvidia_binary_utility,
            load_nvidia_dynamic_lib,
            locate_nvidia_header_directory,
        )
    except ImportError as exc:
        print(f"cuda.pathfinder is not importable: {exc}", file=sys.stderr)
        return 2

    nvcc = find_nvidia_binary_utility("nvcc")
    if not nvcc:
        print(
            "cuda.pathfinder could not locate nvcc. Install the CUDA toolkit wheels\n"
            "(pip install 'cuda-toolkit[nvcc]'), or configure with\n"
            "-DCUNIBS_USE_SYSTEM_CUDA=ON to use a system toolkit instead.",
            file=sys.stderr,
        )
        return 3
    emit_path("NVCC", nvcc)

    headers = locate_nvidia_header_directory("cudart")
    if headers is None or not headers.abs_path:
        print("cuda.pathfinder could not locate the cudart headers.", file=sys.stderr)
        return 4
    emit_path("INCLUDE_DIR", headers.abs_path)

    for libname in (name for name in args.libs.split(",") if name):
        loaded = load_nvidia_dynamic_lib(libname)
        emit(f"FOUND_VIA_{libname}", loaded.found_via)
        if not args.emit_lib_paths:
            continue
        if not loaded.abs_path:
            print(
                f"{libname} was loaded but reports no path on disk "
                f"(found_via={loaded.found_via}, "
                f"already_loaded={loaded.was_already_loaded_from_elsewhere}).",
                file=sys.stderr,
            )
            return 5
        emit_path(f"LIB_{libname}", loaded.abs_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

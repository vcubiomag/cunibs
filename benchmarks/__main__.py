"""Reproduce the measurements behind the cuNIBS and harmonic-recovery manuscripts.

python -m benchmarks list
python -m benchmarks run sphere single_placement --data <dataset>
python -m benchmarks run all --data <dataset> --quick
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from importlib import import_module
from pathlib import Path

# Run order for `all`: a benchmark that reads another's result comes after it.
BENCHMARKS = (
    "sphere",
    "interface_mms",
    "solver_tolerance",
    "coil_dadt",
    "harmonic_constraint",
    "simnibs_timing",
    "simnibs_onlinefem",
    "simnibs_throughput",
    "field_agreement",
    "field_maps",
    "equal_accuracy",
    "single_placement",
    "sweep_throughput",
    "mesh_scaling",
    "coil_cost",
    "ablations",
    "timing_repeatability",
    "adm_crosscode",
    "adm_vs_forward",
    "adm_throughput",
    "adm_grid_spacing",
    "uq_cost",
    "uq_exactness",
    "uq_cohort",
    "determinism",
    "cross_hardware",
    "convergence_robustness",
    "interface_convergence",
    "recovery_in_practice",
    "patch_population",
    "interface_contrast",
    "jump_residual",
    "interface_case",
    "error_in_context",
)


def _default_simnibs_python() -> str | None:
    if path := os.environ.get("SIMNIBS_PYTHON"):
        return path
    pixi = Path.home() / ".pixi/envs/simnibs/bin/python3.12"
    return str(pixi) if pixi.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="name and summary of every benchmark")
    run = sub.add_parser("run", help="run benchmarks and write their results")
    run.add_argument("names", nargs="+", help="benchmark names, or 'all'")
    run.add_argument("--data", type=Path, required=True, help="the extracted dataset")
    run.add_argument("--quick", action="store_true", help="a few subjects and repeats")
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    run.add_argument(
        "--simnibs-python",
        default=_default_simnibs_python(),
        help="SimNIBS 4.6's Python; '' skips SimNIBS (default: $SIMNIBS_PYTHON, "
        "then ~/.pixi/envs/simnibs/bin/python3.12)",
    )
    run.add_argument(
        "--simnibs-exchange",
        type=Path,
        help="leave SimNIBS requests here to answer on another host",
    )
    args = ap.parse_args()

    if args.command == "list":
        for name in BENCHMARKS:
            doc = import_module(f"benchmarks.{name}").__doc__ or ""
            print(f"{name:24s} {doc.strip().splitlines()[0] if doc.strip() else ''}")
        return 0

    names = list(BENCHMARKS) if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown:
        ap.error(f"unknown benchmark(s): {', '.join(unknown)}")

    # numpy's sorts intermittently return a wrong permutation on large arrays under the
    # free-threaded build, so host-side analysis runs with the GIL on.
    if sysconfig.get_config_var("Py_GIL_DISABLED") and os.environ.get("PYTHON_GIL") != "1":
        os.execve(
            sys.executable,
            [sys.executable, "-m", "benchmarks", *sys.argv[1:]],
            os.environ | {"PYTHON_GIL": "1"},
        )

    if len(names) > 1:
        failed = []
        options = [a for a in sys.argv[2:] if a not in args.names]
        for name in names:
            print(f"== {name}", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "benchmarks", "run", name, *options], check=False
            )
            if proc.returncode:
                failed.append(name)
        if failed:
            print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1 if failed else 0

    from benchmarks import _run
    from benchmarks._data import Dataset
    from benchmarks._simnibs import Simnibs

    ctx = _run.Context(
        data=Dataset(args.data),
        quick=args.quick,
        device=args.device,
        out=args.out,
        simnibs=Simnibs(
            args.simnibs_python or None, args.simnibs_exchange, args.out / ".simnibs"
        ),
    )
    print(_run.run(names[0], ctx))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run one benchmark and write its result, with a record of the machine it ran on.

Results land in ``<out>/<machine>/<benchmark>.json``. The machine tag names the GPU and the
CUDA major version, so results from different cards sit side by side. A benchmark that measures
SimNIBS is tagged by the CPU that ran it instead.

Nothing here refuses to write. Anything that could have disturbed a timing (another process on
the GPU, a busy GPU at the start, or clocks slowed by power or heat at the start or end of the
run) is listed under ``warnings``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks._data import Dataset
    from benchmarks._simnibs import Simnibs

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Context:
    data: Dataset
    quick: bool
    device: int
    out: Path
    simnibs: Simnibs
    tag: str = ""

    def _values(self, path: Path) -> dict:
        record = json.loads(path.read_text())
        if record["quick"] and not self.quick:
            raise RuntimeError(f"{path} is a --quick result; rerun it before this benchmark")
        return record["values"]

    def result(self, name: str) -> dict | None:
        """The values another benchmark wrote on this machine, if it has run."""
        path = self.out / self.tag / f"{name}.json"
        return self._values(path) if path.exists() else None

    def latest(self, name: str) -> dict | None:
        """The most recent result of ``name`` from any machine, with ``machine`` set to its tag.

        SimNIBS timings are filed under the CPU that ran them, so a GPU benchmark finds them here.
        """
        found = list(self.out.glob(f"*/{name}.json"))
        if not found:
            return None
        newest = max(found, key=lambda p: p.stat().st_mtime)
        return self._values(newest) | {"machine": newest.parent.name}

    def artifact(self, filename: str) -> Path:
        """A path beside this machine's results, for arrays too large for the JSON."""
        path = self.out / self.tag / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _slug(name: str, drop: set[str], stop: re.Pattern, sep: str = "") -> str:
    kept = []
    for token in re.split(r"[^a-z0-9]+", name.lower()):
        if stop.search(token):
            break
        if token and token not in drop:
            kept.append(token)
    return sep.join(kept)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cpu_tag(name: str | None = None) -> str:
    name = re.sub(r"\s\d+-core.*|\s(cpu\s)?@.*", "", name or cpu_name(), flags=re.IGNORECASE)
    return _slug(
        name, {"amd", "intel", "r", "tm", "core", "processor", "cpu"}, re.compile("^with$"), "-"
    )


# A desktop idles at a few percent. A game never appears among compute processes, so this is the
# only trace of one.
BUSY_PERCENT = 20
SLOWDOWNS = (
    "sw_power_cap",
    "hw_slowdown",
    "sw_thermal_slowdown",
    "hw_thermal_slowdown",
    "hw_power_brake_slowdown",
)


def gpu_state(device: int) -> dict:
    """The driver, clock, load and slowdowns of ``device``, or nothing when NVML is unavailable."""
    from cuda.core import Device, system
    from cuda.core.system.exceptions import NvmlError

    try:
        nvml = Device(device).to_system_device()
        return {
            "driver": ".".join(f"{v:02d}" for v in system.get_kernel_mode_driver_version()),
            "sm_mhz": nvml.get_clock("sm").get_current_mhz(),
            "busy_percent": nvml.utilization.gpu,
            "slowed": [r.value for r in nvml.current_clock_event_reasons if r in SLOWDOWNS],
            "others_mib": [
                p.used_gpu_memory / 2**20
                for p in nvml.compute_running_processes
                if p.pid != os.getpid()
            ],
        }
    except (NvmlError, RuntimeError):
        return {}


def gpu_tag(device: int) -> str:
    import cupy as cp

    name = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
    slug = _slug(name, {"nvidia", "geforce"}, re.compile(r"gb$|^sxm|^pcie|^hbm"))
    # One model ships with several memory sizes, which run different problem sizes.
    if memory := re.search(r"(\d+)GB", name):
        slug += f"-{memory[1]}gb"
    return f"{slug}-cu{cp.cuda.runtime.runtimeGetVersion() // 1000}"


def cunibs_version() -> str:
    import cunibs

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return version("cunibs")
    modified = ", modified" if git("status", "--porcelain", "--", "src") else ""
    # An installed wheel can come from any commit, so the checkout's says nothing about its code.
    wheel = "" if Path(cunibs.__file__).resolve().is_relative_to(REPO / "src") else ", wheel"
    return f"{version('cunibs')} ({commit}{modified}{wheel})"


def _warnings(before: dict, after: dict) -> list[str]:
    out = []
    busy = before.get("busy_percent", 0)
    if busy >= BUSY_PERCENT:
        out.append(f"the GPU was {busy}% busy before the run started")
    for when, state in (("start", before), ("end", after)):
        if state.get("others_mib"):
            out.append(
                f"{len(state['others_mib'])} other processes held "
                f"{sum(state['others_mib']):.0f} MiB of the GPU at the {when}"
            )
        if state.get("slowed"):
            out.append(f"clocks were slowed at the {when}: {', '.join(state['slowed'])}")
    return out


def run(name: str, ctx: Context) -> Path | str:
    """Run ``name`` and write its result. Returns the result's path, or why nothing was written.

    A benchmark returns ``None`` when it has nothing to report without SimNIBS.
    """
    import cupy as cp

    module = import_module(f"benchmarks.{name}")
    cp.cuda.Device(ctx.device).use()
    ctx.tag = gpu_tag(ctx.device)
    before = gpu_state(ctx.device)
    t0 = time.perf_counter()

    values = module.run(ctx)

    elapsed = time.perf_counter() - t0
    if ctx.simnibs.pending:
        return (
            f"waiting for {ctx.simnibs.pending} SimNIBS answers: run `python "
            f"benchmarks/simnibs_child.py serve {ctx.simnibs.exchange}` in SimNIBS's Python, "
            "then run this again"
        )
    if values is None:
        return "skipped: needs SimNIBS (--simnibs-python or --simnibs-exchange)"
    warnings = _warnings(before, gpu_state(ctx.device))
    if ctx.simnibs.unanswered:
        warnings.append("SimNIBS was not available; the values that need it are absent")
    record = {
        "benchmark": name,
        "date": datetime.now(UTC).date().isoformat(),
        "quick": ctx.quick,
        "elapsed_s": round(elapsed, 1),
        "system": {
            "gpu": cp.cuda.runtime.getDeviceProperties(ctx.device)["name"].decode(),
            "gpu_clock_mhz": before.get("sm_mhz"),
            "driver": before.get("driver"),
            "cpu": cpu_name(),
            "cunibs": cunibs_version(),
        },
        "warnings": warnings,
        "values": values,
    }
    tag = module.tag(values) if hasattr(module, "tag") else ctx.tag
    path = ctx.out / tag / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path

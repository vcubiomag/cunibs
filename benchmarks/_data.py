"""The dataset, and the coil placements every benchmark derives from it.

A dataset directory holds ``subjects/<id>.h5`` (a head mesh with its EEG caps) and
``ladder/<id>_s<NNN>.h5`` (one subject remeshed with every element length scaled by NNN/100).

A target is a 10-10 electrode, the coil 4 mm off the scalp above it, and its handle at 45 degrees
from posterior, turned about the inward normal. Every placement here is a function of the files
and those constants, so any machine reproduces it exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

COIL = "MagVenture_C-B70"
DIDT = 1e6
DISTANCE_MM = 4.0
HANDLE_ANGLE_DEG = 45.0
TARGETS = ("C3", "F3", "FCz", "P3")
# Every ROI quantity is read over the gray matter within ROI_RADIUS_MM of a point ROI_DEPTH_MM
# below the target's scalp projection, along the inward normal.
ROI_DEPTH_MM = 20.0
ROI_RADIUS_MM = 10.0

# Only the handle's direction matters; a long arm keeps it well defined where the scalp is flat.
_HANDLE_ARM_MM = 60.0


@dataclass(frozen=True)
class Rung:
    subject: str
    scale: float
    path: Path


class Dataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not (self.root / "subjects").is_dir():
            raise FileNotFoundError(f"{self.root} has no subjects/ directory")

    @cached_property
    def subjects(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "subjects").glob("*.h5"))

    def mesh(self, subject: str) -> Path:
        return self.root / "subjects" / f"{subject}.h5"

    @cached_property
    def ladder(self) -> list[Rung]:
        """Coarsest first."""
        matches = [
            (re.fullmatch(r"(.+)_s(\d{3})", p.stem), p)
            for p in (self.root / "ladder").glob("*.h5")
        ]
        rungs = [Rung(m[1], int(m[2]) / 100, p) for m, p in matches if m]
        return sorted(rungs, key=lambda r: -r.scale)

    @property
    def reference_subject(self) -> str:
        """The subject the ladder remeshes, which the single-subject benchmarks also use."""
        return self.ladder[0].subject if self.ladder else self.subjects[0]


def handle_points(subject, centers_mm, angle_deg: float = HANDLE_ANGLE_DEG) -> np.ndarray:
    """Handle points at ``angle_deg`` from posterior, about each center's inward scalp normal."""
    from cunibs.head import project_to_scalp

    origins, inward = project_to_scalp(subject, centers_mm)
    posterior = np.array([0.0, -1.0, 0.0])
    base = posterior - (inward @ posterior)[:, None] * inward
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    theta = np.radians(angle_deg)
    rotated = base * np.cos(theta) + np.cross(inward, base) * np.sin(theta)
    return origins + _HANDLE_ARM_MM * rotated


def placements(subject, mesh_path: Path, labels=TARGETS, angles_deg=(HANDLE_ANGLE_DEG,)):
    """One :class:`~cunibs.Placement` per (label, angle), labels outermost."""
    from cunibs import Placement
    from cunibs.head import read_cap

    cap = read_cap(mesh_path)
    centers = np.array([cap[label] for label in labels])
    out = []
    for center, *handles in zip(
        centers, *(handle_points(subject, centers, a) for a in angles_deg), strict=True
    ):
        out += [Placement(center, h, DISTANCE_MM) for h in handles]
    return out


def roi_centers(subject, placements) -> np.ndarray:
    from cunibs.head import project_to_scalp

    origins, inward = project_to_scalp(subject, np.array([p.center_mm for p in placements]))
    return origins + ROI_DEPTH_MM * inward


def rois(subject, placements) -> list:
    """Each placement's ROI, resolved on ``subject``'s mesh."""
    return [subject.roi(c, radius_mm=ROI_RADIUS_MM) for c in roi_centers(subject, placements)]


def frames(subject, placements) -> np.ndarray:
    """The ``(P, 4, 4)`` coil-to-head affines in mm that the solver builds from ``placements``."""
    import cupy as cp

    from cunibs.fem import compute_coil_transforms

    return cp.asnumpy(
        compute_coil_transforms(
            subject.context,
            np.array([p.center_mm for p in placements]),
            np.array([p.handle_mm for p in placements]),
            np.array([p.distance_mm for p in placements]),
        )
    )

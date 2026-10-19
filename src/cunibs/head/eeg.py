"""Read the EEG electrode positions CHARM writes beside a head mesh.

CHARM emits ``m2m_<id>/eeg_positions/<cap>.csv`` next to the ``.msh`` it meshes, in the same
subject coordinates as the mesh, so a 10-20 label resolves to a scalp point with no registration
step in between.

Rows are ``<kind>,x,y,z,<label>``. ``kind`` is not always ``Electrode``: every cap file carries Cz
as a ``ReferenceElectrode`` row, so filtering on ``Electrode`` alone drops the vertex silently.
:func:`read_cap` reads both kinds; ``Fiducial`` rows (Nz, Iz, LPA, RPA) are anatomical landmarks
rather than electrode sites and come back from :func:`read_fiducials` instead.

A mesh written by :meth:`~cunibs.HeadMesh.save` with ``eeg_positions`` carries every cap file
verbatim, so each reader here also accepts that ``.h5`` file and picks the cap by name.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

from cunibs.mesh import EEG_POSITIONS_GROUP, is_hdf5

# Both are electrode sites at scalp positions; which one a label lands in is a property of the cap
# montage, not of the position.
_ELECTRODE_KINDS = frozenset({"Electrode", "ReferenceElectrode"})
_FIDUCIAL_KIND = "Fiducial"

# What CHARM names the directory, and the cap SimNIBS treats as its default 10-10 montage.
EEG_POSITIONS_DIR = "eeg_positions"
DEFAULT_CAP = "EEG10-10_UI_Jurak_2007"


def _cap_text(path: Path, cap: str) -> str:
    try:
        if not is_hdf5(path):
            return path.read_text()
        with h5py.File(path, "r") as f:
            caps = f.get(EEG_POSITIONS_GROUP, {})
            if cap not in caps:
                available = ", ".join(sorted(caps)) or "none"
                raise ValueError(f"{path}: carries no {cap} EEG positions (has: {available})")
            return caps[cap][()].decode()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read EEG positions: {exc}") from exc


def _read_rows(
    path: Path, cap: str, kinds: frozenset[str] | set[str]
) -> dict[str, npt.NDArray[np.float64]]:
    text = _cap_text(path, cap)

    out: dict[str, npt.NDArray[np.float64]] = {}
    for line in text.splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 5 or parts[0] not in kinds:
            continue
        try:
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        # Later rows win, matching how a duplicated label would be read by anything else.
        out[parts[4]] = xyz
    return out


def read_cap(path: str | Path, cap: str = DEFAULT_CAP) -> dict[str, npt.NDArray[np.float64]]:
    """Electrode label -> scalp position in mm, from one CHARM cap CSV or a saved ``.h5`` mesh.

    ``cap`` names the cap to read from a mesh file and is ignored for a CSV. Includes
    ``ReferenceElectrode`` rows, which is where ``Cz`` lives in every cap file CHARM writes.
    Fiducials are excluded; see :func:`read_fiducials`.
    """
    positions = _read_rows(Path(path), cap, _ELECTRODE_KINDS)
    if not positions:
        raise ValueError(f"{path}: no electrode rows found; is this a CHARM eeg_positions CSV?")
    return positions


def read_fiducials(
    path: str | Path, cap: str = DEFAULT_CAP
) -> dict[str, npt.NDArray[np.float64]]:
    """Anatomical landmark label -> position in mm (``Nz``, ``Iz``, ``LPA``, ``RPA``)."""
    return _read_rows(Path(path), cap, {_FIDUCIAL_KIND})


def find_cap(m2m_dir: str | Path, cap: str = DEFAULT_CAP) -> Path:
    """Locate one cap CSV inside a CHARM ``m2m_<id>`` directory."""
    root = Path(m2m_dir) / EEG_POSITIONS_DIR
    path = root / (cap if cap.endswith(".csv") else f"{cap}.csv")
    if not path.exists():
        available = sorted(p.stem for p in root.glob("*.csv")) if root.is_dir() else []
        raise FileNotFoundError(
            f"{path} does not exist."
            + (f" Available caps: {', '.join(available)}." if available else "")
        )
    return path


def read_subject_cap(
    subject: str | Path, cap: str = DEFAULT_CAP
) -> dict[str, npt.NDArray[np.float64]]:
    """Electrode positions for one subject, by cap name, from its ``m2m_<id>`` or ``.h5`` mesh."""
    subject = Path(subject)
    return read_cap(subject if subject.is_file() else find_cap(subject, cap), cap)

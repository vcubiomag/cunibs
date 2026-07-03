from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt


_COILS_ROOT = Path(__file__).resolve().parent / "coils"
DEYMED_50BF = _COILS_ROOT / "Deymed_50BF.h5"
DEYMED_70BF = _COILS_ROOT / "Deymed_70BF.h5"
DEYMED_120BFV = _COILS_ROOT / "Deymed_120BFV.h5"
MAGMORE_PMD25_DECENTRAL = _COILS_ROOT / "MagMore_PMD25-decentral.h5"
MAGMORE_PMD45_EEG = _COILS_ROOT / "MagMore_PMD45-EEG.h5"
MAGMORE_PMD70 = _COILS_ROOT / "MagMore_PMD70.h5"
MAGSTIM_D70 = _COILS_ROOT / "MagStim_D70.h5"
MAGSTIM_DCC = _COILS_ROOT / "MagStim_DCC.h5"
MAGVENTURE_C_100 = _COILS_ROOT / "MagVenture_C-100.h5"
MAGVENTURE_C_B60 = _COILS_ROOT / "MagVenture_C-B60.h5"
MAGVENTURE_C_B70 = _COILS_ROOT / "MagVenture_C-B70.h5"
MAGVENTURE_COOL_B35 = _COILS_ROOT / "MagVenture_Cool-B35.h5"
MAGVENTURE_COOL_B65 = _COILS_ROOT / "MagVenture_Cool-B65.h5"
MAGVENTURE_COOL_B70 = _COILS_ROOT / "MagVenture_Cool-B70.h5"
MAGVENTURE_COOL_D_B80 = _COILS_ROOT / "MagVenture_Cool-D-B80.h5"
MAGVENTURE_MC_125 = _COILS_ROOT / "MagVenture_MC-125.h5"
MAGVENTURE_MC_125_NEW = _COILS_ROOT / "MagVenture_MC-125_new.h5"
MAGVENTURE_MC_B70 = _COILS_ROOT / "MagVenture_MC-B70.h5"
MAGVENTURE_MCF_75 = _COILS_ROOT / "MagVenture_MCF-75.h5"
MAGVENTURE_MCF_B65 = _COILS_ROOT / "MagVenture_MCF-B65.h5"
MAGVENTURE_MCF_B65_NEW = _COILS_ROOT / "MagVenture_MCF-B65_new.h5"
MAGVENTURE_MC_B65_HO8 = _COILS_ROOT / "MagVenture_MC_B65_HO8.h5"
MAGVENTURE_MMC_140_II = _COILS_ROOT / "MagVenture_MMC-140-II.h5"
MAGVENTURE_MRI_B91 = _COILS_ROOT / "MagVenture_MRI-B91.h5"
MAGVENTURE_MST_TWIN = _COILS_ROOT / "MagVenture_MST_Twin.h5"


def _decode_attr(value: object) -> str | int | float:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


@dataclass(frozen=True)
class Coil:
    """Store magnetic dipoles and coil metadata.

    Positions use metres. Moments use A·m².
    """

    positions_m: npt.NDArray[np.float64]
    moments: npt.NDArray[np.float64]
    name: str = ""
    didt_max: float | None = None
    metadata: dict[str, str | int | float] | None = None

    @classmethod
    def load(cls, source: str | Path) -> "Coil":
        """Load a coil from a bundled/encoded ``.h5`` file (see :func:`encode_ccd`)."""
        with h5py.File(Path(source), "r") as h5f:
            dset = h5f["dipoles"]
            data = np.asarray(dset, dtype=np.float64)
            attrs: dict[str, str | int | float] = {
                k: _decode_attr(v) for k, v in dset.attrs.items()
            }
        didt_raw = attrs.get("dIdtmax")
        return cls(
            positions_m=np.ascontiguousarray(data[:, 0:3]),
            moments=np.ascontiguousarray(data[:, 3:6]),
            name=str(attrs.get("coilname", Path(source).stem)),
            didt_max=float(didt_raw) if didt_raw is not None else None,
            metadata=attrs,
        )


def encode_ccd(ccd_file: Path, output_h5: Path) -> None:
    with open(ccd_file, "r") as f:
        meta_line = f.readline().strip().lstrip("#").strip()
        num_elements = int(f.readline().strip())
        data = np.loadtxt(f, dtype=np.float64, skiprows=1, max_rows=num_elements)

    parts = (p.split("=", 1) for p in meta_line.split(";") if "=" in p)
    metadata: dict[str, str | int] = {k.strip(): v.strip() for k, v in parts}

    metadata["num_elements"] = num_elements

    with h5py.File(output_h5, "w") as h5f:
        dset = h5f.create_dataset("dipoles", data=data, compression="gzip")
        dset.attrs.update(metadata)

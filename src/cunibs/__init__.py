"""GPU FEM solver for the TMS E-field."""

from importlib.metadata import version

from cunibs import adm, coil, metrics, uq
from cunibs.adm import ResolvedTarget, Target, resolve_target
from cunibs.coil import Coil
from cunibs.fem import RECOVERY_MODES, Recovery
from cunibs.mesh import HeadMesh, load_mesh
from cunibs.simulation import FieldResult, Placement, Subject
from cunibs.uq import ConductivityUQConfig, ConductivityUQResult, ConductivityUQSummary

__version__ = version("cunibs")

__all__ = [
    "RECOVERY_MODES",
    "Coil",
    "ConductivityUQConfig",
    "ConductivityUQResult",
    "ConductivityUQSummary",
    "FieldResult",
    "HeadMesh",
    "Placement",
    "Recovery",
    "ResolvedTarget",
    "Subject",
    "Target",
    "__version__",
    "adm",
    "coil",
    "load_mesh",
    "metrics",
    "resolve_target",
    "uq",
]

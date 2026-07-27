"""GPU FEM solver for the TMS E-field."""

from importlib.metadata import version

from cunibs import coil, metrics
from cunibs.coil import Coil
from cunibs.mesh import HeadMesh, load_mesh
from cunibs.simulation import FieldResult, Placement, Subject
from cunibs import adm, uq
from cunibs.adm import ResolvedTarget, Target, resolve_target
from cunibs.uq import ConductivityUQConfig, ConductivityUQResult, ConductivityUQSummary

__version__ = version("cunibs")

__all__ = [
    "__version__",
    "Subject",
    "Placement",
    "FieldResult",
    "Coil",
    "HeadMesh",
    "load_mesh",
    "Target",
    "ResolvedTarget",
    "resolve_target",
    "ConductivityUQConfig",
    "ConductivityUQResult",
    "ConductivityUQSummary",
    "coil",
    "metrics",
    "adm",
    "uq",
]

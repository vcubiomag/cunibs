"""GPU FEM solver for the TMS E-field."""

from importlib.metadata import version
from importlib.util import find_spec

if find_spec("cupy") is None or find_spec("cuda.bindings") is None:
    raise ImportError(
        "cuNIBS needs CuPy and cuda-bindings for your CUDA version. Install it with one of "
        "the extras cu12, cu13, cu12-local or cu13-local, e.g. pip install 'cunibs[cu13]'."
    )

from cunibs import adm, coil, head, metrics, uq
from cunibs.adm import ResolvedTarget, Target, resolve_target
from cunibs.coil import Coil, bundled_coils
from cunibs.fem import RECOVERY_MODES, Recovery
from cunibs.mesh import HeadMesh, MeshReport, load_mesh, validate_mesh
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
    "MeshReport",
    "Placement",
    "Recovery",
    "ResolvedTarget",
    "Subject",
    "Target",
    "__version__",
    "adm",
    "bundled_coils",
    "coil",
    "head",
    "load_mesh",
    "metrics",
    "resolve_target",
    "uq",
    "validate_mesh",
]

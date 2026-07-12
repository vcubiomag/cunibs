"""Auxiliary Dipole Method: fast reciprocity-based TMS coil-placement evaluation.

A few one-time adjoint solves (reusing the forward AMGx hierarchy) sample a reciprocity field on a
regular grid; thereafter the target E-field of any placement is a trilinear interpolation plus a
dipole sum.
"""

from cunibs.adm.evaluate import evaluate
from cunibs.adm.optimize import OptimizeResult, optimize
from cunibs.adm.reciprocity import ReciprocityField, build_reciprocity
from cunibs.adm.target import ResolvedTarget, Target, resolve_target

__all__ = [
    "Target",
    "ResolvedTarget",
    "resolve_target",
    "optimize",
    "OptimizeResult",
    "build_reciprocity",
    "ReciprocityField",
    "evaluate",
]

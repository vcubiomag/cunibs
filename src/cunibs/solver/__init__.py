# Must precede _solver_ext: loads the CUDA libraries the extension links against.
from cunibs.solver import _cuda_preload

try:
    from cunibs.solver._solver_ext import *
except ImportError as exc:
    raise ImportError(_cuda_preload.describe_failure(exc)) from exc

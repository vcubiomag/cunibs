"""Skip tests when no CUDA device is available."""

from __future__ import annotations

import pytest


def has_gpu() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(not has_gpu(), reason="no CUDA GPU available")

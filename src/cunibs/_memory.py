"""Scoping for the device memory a build of cached state allocates through CuPy."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar

import cupy as cp
import cupyx.scipy.sparse as csp

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

A = TypeVar("A", cp.ndarray, csp.csr_matrix)


class Arena:
    """The allocator that was current when a build began, for what the build keeps.

    Everything a build allocates is a temporary unless it is allocated inside :meth:`resident` or
    copied out with :meth:`keep`, both of which use the caller's allocator.
    """

    def __init__(self, caller: Callable[[int], cp.cuda.MemoryPointer]) -> None:
        self._caller = caller

    @contextmanager
    def resident(self) -> Iterator[None]:
        """Allocate what the build keeps directly from the caller's allocator."""
        with cp.cuda.using_allocator(self._caller):
            yield

    def keep(self, a: A) -> A:
        """A copy of ``a`` in the caller's allocator, for an array computed among temporaries."""
        with self.resident():
            return a.copy()


@contextmanager
def build_arena() -> Iterator[Arena]:
    """Run a build of long-lived device state with its temporaries outside the caller's pool.

    CuPy's pool serves a request by splitting the smallest free chunk that fits, and
    ``free_all_blocks`` only returns whole chunks. An array that outlives a build, allocated after
    one of the build's large temporaries was freed, can land in that temporary's chunk and keep all
    of it reserved for as long as the array lives.

    So the temporaries come from CUDA's stream-ordered allocator instead, the device mempool the
    extension's own scratch also uses. Each is returned with ``cudaFreeAsync`` when its array is
    dropped, including one that outlives the block, and the mempool hands unused memory back to
    the device at the next synchronisation. Nothing a build keeps shares a chunk with them.
    """
    arena = Arena(cp.cuda.get_allocator())
    with cp.cuda.using_allocator(cp.cuda.malloc_async):
        yield arena

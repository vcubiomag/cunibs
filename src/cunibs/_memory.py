"""Scoping for the device memory a build of cached state allocates through CuPy."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, NoReturn, TypeVar

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


_CUDA_ERROR_MEMORY_ALLOCATION = 2
_LIBRARY_OUT_OF_MEMORY = {
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUSOLVER_STATUS_ALLOC_FAILED",
    "CUSOLVER_STATUS_INTERNAL_ERROR",
}

_scratch: cp.cuda.MemoryPool | None = None
_scratch_lock = threading.Lock()


def scratch_pool() -> cp.cuda.MemoryPool:
    """The pool that per-call temporaries come from, shared by the whole process and never dropped.

    A block released after its pool is gone strands the pool's chunk, and a call's temporaries can
    outlive whatever owned a pool: another thread may still be using them, or may not yet have
    run the release of one it created.
    """
    global _scratch
    with _scratch_lock:
        if _scratch is None:
            _scratch = cp.cuda.MemoryPool()
        return _scratch


def release_scratch() -> None:
    """Return the scratch pool's free blocks to the device, if it was ever used."""
    with _scratch_lock:
        if _scratch is not None:
            _scratch.free_all_blocks()


def _reraise(error: cp.cuda.runtime.CUDARuntimeError, size: int) -> NoReturn:
    """Raise a failed allocation: only CuPy's pools raise a MemoryError themselves."""
    if error.status == _CUDA_ERROR_MEMORY_ALLOCATION:
        raise cp.cuda.memory.OutOfMemoryError(size, 0) from error
    raise error


def unpooled(size: int) -> cp.cuda.MemoryPointer:
    """An allocator giving each array its own device allocation, released when the array dies.

    For state that can outlive whatever built it: a block from a pool that has since been dropped
    strands the chunk it came from.
    """
    try:
        return cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0)
    except cp.cuda.runtime.CUDARuntimeError as error:
        _reraise(error, size)


def _stream_ordered(size: int) -> cp.cuda.MemoryPointer:
    try:
        return cp.cuda.malloc_async(size)
    except cp.cuda.runtime.CUDARuntimeError as error:
        _reraise(error, size)


@contextmanager
def library_memory_errors() -> Iterator[None]:
    """Report cuBLAS or cuSOLVER running out of device memory as the MemoryError it is.

    Each creates a handle on a thread's first use, whose allocations can fail on a nearly full
    device, where cuBLAS says so and cuSOLVER reports an internal error. Their exception types are
    RuntimeErrors matched by message, since importing them would load both libraries.
    """
    try:
        yield
    except RuntimeError as error:
        if str(error) in _LIBRARY_OUT_OF_MEMORY:
            raise MemoryError(f"the device ran out of memory: {error}") from error
        raise


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
    with library_memory_errors(), cp.cuda.using_allocator(_stream_ordered):
        yield arena

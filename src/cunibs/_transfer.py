"""Device-to-host copies that run behind the GPU work queued after them."""

from __future__ import annotations

import threading

import cupy as cp
import numpy as np

from cunibs.solver import host_copy_async

# Two slots, so one drains into its host array while the other fills.
_SLOT_BYTES = 64 << 20
_SLOTS = 2


class AsyncCopier:
    """Copies device arrays into new host arrays through a pinned ring, off the calling thread.

    A copy waits for the GPU work that produced its source, then moves it a slot at a time: the
    transfer runs on the copy engine and the slot drains into the host array on the CUDA driver's
    callback thread. Each slot has its own stream, so its next transfer waits in stream order for
    its last drain.

    Each caller keeps what its copies touch alive until its :meth:`settle` or :meth:`drain`, in the
    list it passes :meth:`copy`. Holding them here would let one thread release another's device
    arrays, and on a free-threaded build those then stay allocated until their creating thread
    next wakes.
    """

    def __init__(self) -> None:
        # Each slot is an allocation of its own, released with the copier. CuPy's pinned pool
        # would keep the page-locked memory for the life of the process.
        self._slots = [
            cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(_SLOT_BYTES), 0)
            for _ in range(_SLOTS)
        ]
        self._streams = [cp.cuda.Stream(non_blocking=True) for _ in range(_SLOTS)]
        self._turn = 0
        self._lock = threading.Lock()

    def copy(self, src: cp.ndarray, after: cp.cuda.Event, held: list[object]) -> np.ndarray:
        """A host array holding ``src`` once a later :meth:`fence` has been settled.

        Appends ``src`` and that array to ``held``, which must outlive the fence.
        """
        if not src.flags.c_contiguous:
            raise ValueError("AsyncCopier.copy needs a C-contiguous source")
        dst = np.empty(src.shape, dtype=src.dtype)
        held += (dst, src)
        with self._lock:
            for stream in self._streams:
                stream.wait_event(after)
            offset = 0
            while offset < src.nbytes:
                i = self._turn
                self._turn = (i + 1) % _SLOTS
                n = min(_SLOT_BYTES, src.nbytes - offset)
                stream, slot = self._streams[i], self._slots[i].ptr
                cp.cuda.runtime.memcpyAsync(
                    slot,
                    src.data.ptr + offset,
                    n,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )
                host_copy_async(slot, dst.ctypes.data + offset, n, stream.ptr)
                offset += n
        return dst

    def fence(self) -> list[cp.cuda.Event]:
        """Markers that are passed once every copy queued so far has landed."""
        with self._lock:
            events = []
            for stream in self._streams:
                event = cp.cuda.Event(disable_timing=True)
                event.record(stream)
                events.append(event)
            return events

    def settle(self, fence: list[cp.cuda.Event]) -> None:
        """Wait for ``fence``."""
        for event in fence:
            event.synchronize()

    def drain(self) -> None:
        """Wait for every queued copy."""
        self.settle(self.fence())

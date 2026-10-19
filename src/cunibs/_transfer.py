"""Device-to-host copies that run behind the GPU work queued after them."""

from __future__ import annotations

import threading
from typing import NamedTuple

import cupy as cp
import numpy as np

from cunibs.solver import host_copy_async

# Two slots, so one drains into its host array while the other fills.
_SLOT_BYTES = 64 << 20
_SLOTS = 2


class Fence(NamedTuple):
    """The point in the copy queue a set of events marks, and how many copies precede it."""

    events: list[cp.cuda.Event]
    copies: int


class AsyncCopier:
    """Copies device arrays into new host arrays through a pinned ring, off the calling thread.

    A copy waits for the GPU work that produced its source, then moves it a slot at a time: the
    transfer runs on the copy engine and the slot drains into the host array on the CUDA driver's
    callback thread. Each slot has its own stream, so its next transfer waits in stream order for
    its last drain.

    The copier keeps every source and destination alive until a :meth:`settle` or :meth:`drain`
    has seen its copy land, so dropping a result early can never free memory the callback thread
    is still writing.
    """

    def __init__(self) -> None:
        self._slots = [cp.cuda.alloc_pinned_memory(_SLOT_BYTES) for _ in range(_SLOTS)]
        self._streams = [cp.cuda.Stream(non_blocking=True) for _ in range(_SLOTS)]
        self._turn = 0
        self._lock = threading.Lock()
        # (sequence number, destination, source) for every copy not yet known to have landed.
        self._inflight: list[tuple[int, np.ndarray, cp.ndarray]] = []
        self._queued = 0

    def copy(self, src: cp.ndarray, after: cp.cuda.Event) -> np.ndarray:
        """A host array holding ``src`` once a later :meth:`fence` has been settled."""
        if not src.flags.c_contiguous:
            raise ValueError("AsyncCopier.copy needs a C-contiguous source")
        dst = np.empty(src.shape, dtype=src.dtype)
        with self._lock:
            self._queued += 1
            self._inflight.append((self._queued, dst, src))
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

    def fence(self) -> Fence:
        """A marker that is passed once every copy queued so far has landed."""
        with self._lock:
            events = []
            for stream in self._streams:
                event = cp.cuda.Event(disable_timing=True)
                event.record(stream)
                events.append(event)
            return Fence(events, self._queued)

    def settle(self, fence: Fence) -> None:
        """Wait for ``fence`` and release the buffers of the copies before it."""
        for event in fence.events:
            event.synchronize()
        with self._lock:
            self._inflight = [entry for entry in self._inflight if entry[0] > fence.copies]

    def drain(self) -> None:
        """Wait for every queued copy and release all of their buffers."""
        self.settle(self.fence())

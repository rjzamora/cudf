# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Memory-aware backpressure for I/O operations."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class MemoryGate:
    """
    A gate that can pause I/O operations based on memory pressure.

    This gate works in conjunction with a semaphore to provide both
    concurrency limiting (via semaphore) and dynamic memory-based
    backpressure (via this gate).

    The gate can be in two states:
    - Open: I/O operations can proceed
    - Closed: I/O operations are paused until the gate reopens

    Examples
    --------
    >>> gate = MemoryGate()
    >>> async def io_task():
    ...     async with gate:
    ...         # Perform I/O operation
    ...         await read_data()
    >>>
    >>> # From a memory monitor task:
    >>> if memory_usage > threshold:
    ...     gate.close()  # Pause all I/O
    >>> else:
    ...     gate.open()   # Resume I/O
    """

    def __init__(self):
        """Initialize the memory gate in open state."""
        self._event = asyncio.Event()
        self._event.set()  # Start in open state

    def open(self) -> None:
        """Open the gate, allowing I/O operations to proceed."""
        self._event.set()

    def close(self) -> None:
        """Close the gate, pausing all I/O operations."""
        self._event.clear()

    def is_open(self) -> bool:
        """Check if the gate is currently open."""
        return self._event.is_set()

    async def __aenter__(self):
        """Wait for the gate to be open before proceeding."""
        await self._event.wait()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context (no cleanup needed)."""
        return False


class MemoryMonitor:
    """
    Background task that monitors memory usage and controls a MemoryGate.

    Parameters
    ----------
    gate : MemoryGate
        The gate to control based on memory usage.
    get_memory_mb : Callable[[], float]
        Function that returns current memory usage in MB.
    high_threshold_mb : float
        When memory exceeds this threshold, close the gate.
    low_threshold_mb : float
        When memory falls below this threshold, open the gate.
    check_interval_s : float
        How often to check memory usage (in seconds).

    Examples
    --------
    >>> gate = MemoryGate()
    >>> monitor = MemoryMonitor(
    ...     gate=gate,
    ...     get_memory_mb=lambda: get_gpu_memory_mb(),
    ...     high_threshold_mb=8000,  # Pause I/O above 8GB
    ...     low_threshold_mb=6000,   # Resume I/O below 6GB
    ...     check_interval_s=0.1,
    ... )
    >>> async with monitor:
    ...     # I/O operations using the gate will be controlled
    ...     await perform_streaming_pipeline()
    """

    def __init__(
        self,
        gate: MemoryGate,
        get_memory_mb: Callable[[], float],
        high_threshold_mb: float,
        low_threshold_mb: float,
        check_interval_s: float = 0.1,
    ):
        """Initialize the memory monitor."""
        if low_threshold_mb >= high_threshold_mb:
            raise ValueError("low_threshold_mb must be less than high_threshold_mb")

        self.gate = gate
        self.get_memory_mb = get_memory_mb
        self.high_threshold_mb = high_threshold_mb
        self.low_threshold_mb = low_threshold_mb
        self.check_interval_s = check_interval_s
        self._task = None
        self._stop_event = asyncio.Event()

    async def _monitor_loop(self):
        """Background loop that monitors memory and controls the gate."""
        while not self._stop_event.is_set():
            try:
                current_memory = self.get_memory_mb()

                if current_memory > self.high_threshold_mb and self.gate.is_open():
                    # Close the gate when memory is too high
                    self.gate.close()
                elif current_memory < self.low_threshold_mb and not self.gate.is_open():
                    # Open the gate when memory is back to normal
                    self.gate.open()

                # Wait before next check
                await asyncio.sleep(self.check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception:
                # Don't let monitor crash the pipeline
                # TODO: Add logging here
                await asyncio.sleep(self.check_interval_s)

    async def start(self):
        """Start the memory monitor background task."""
        if self._task is not None:
            raise RuntimeError("MemoryMonitor is already running")
        self._stop_event.clear()
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the memory monitor background task."""
        if self._task is None:
            return
        self._stop_event.set()
        await self._task
        self._task = None

    async def __aenter__(self):
        """Start monitoring when entering context."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring when exiting context."""
        await self.stop()
        # Always open the gate when exiting to avoid deadlocks
        self.gate.open()
        return False



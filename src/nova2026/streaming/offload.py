"""Non-blocking task offloading for the consumer side.

The submitting loop stays the acquisition owner: ``submit`` hands a reference to
an already-independent item to a bounded queue and returns immediately, while
worker threads run the handler in their own context. This keeps pulling samples
independent from how long a handler takes.
"""

import math
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic, sleep

# Backpressure policies for a full queue.
OVERFLOW_POLICIES = ("drop_oldest", "drop_newest", "raise")


class TaskOffloader:
    """Run a handler on worker threads without blocking the submitting loop.

    Args:
        handler: Callable applied to each submitted item on a worker thread.
        workers: Worker threads. ``1`` reproduces a single consumer thread; a
            larger value runs a pool. Handlers must tolerate concurrent calls.
        capacity: Queued items before the overflow policy applies. Memory use is
            bounded by ``capacity`` times the item size.
        overflow: ``"drop_oldest"`` discards the oldest queued item to keep the
            freshest data, ``"drop_newest"`` discards the incoming item to keep
            history, ``"raise"`` propagates ``queue.Full`` to the caller.
        on_result: Optional callable receiving each handler return value; runs on
            a worker thread.
        on_error: Optional callable receiving each handler exception; runs on a
            worker thread.

    Notes:
        The submitting thread never runs the handler, so it can keep pulling.
        Handlers and callbacks must be thread-safe. Use ``raise_error()`` on the
        submitting thread to observe the first handler or callback failure.
        Items must be self-contained: a window copied out of a ring buffer
        qualifies, a view into a reused buffer does not.

    Attributes:
        submitted: Items accepted by ``submit``.
        completed: Handler calls that returned.
        failed: Handler or callback calls that raised.
        dropped: Items discarded by the overflow policy or after ``close``.
    """

    def __init__(
        self,
        handler: Callable[[object], object],
        *,
        workers: int = 1,
        capacity: int = 8,
        overflow: str = "drop_oldest",
        on_result: Callable[[object], object] | None = None,
        on_error: Callable[[BaseException], object] | None = None,
    ) -> None:
        """Validate the pool contract and start the worker threads."""

        if not callable(handler):
            raise ValueError("handler must be callable.")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        if overflow not in OVERFLOW_POLICIES:
            raise ValueError(f"overflow must be one of {OVERFLOW_POLICIES}.")

        self._handler = handler
        self._on_result = on_result
        self._on_error = on_error
        self._overflow = overflow
        # Bounded FIFO: items wait here until a worker is free.
        self._queue: Queue = Queue(maxsize=capacity)
        # Protects the counters and the first-error slot.
        self._lock = Lock()
        self._error: BaseException | None = None
        self._closed = False

        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.dropped = 0

        # One daemon thread per worker; they block on queue.get() when idle.
        self._workers = [
            Thread(target=self._run, name=f"nova-offload-{index}", daemon=True)
            for index in range(workers)
        ]
        for worker in self._workers:
            worker.start()

    @property
    def pending(self) -> int:
        """Items still waiting in the queue (a handler may hold one already)."""

        return self._queue.qsize()

    def submit(self, item: object) -> bool:
        """Queue one item and return immediately.

        Returns:
            ``True`` when the item was queued, ``False`` when it was dropped or
            the offloader is closed.

        Raises:
            queue.Full: If the queue is full and ``overflow="raise"``.
        """

        # Closed offloaders refuse new work.
        if self._closed:
            self._bump("dropped")
            return False

        try:
            self._queue.put_nowait(item)
        except Full:
            if self._overflow == "raise":
                raise
            if self._overflow == "drop_newest":
                # Keep history: reject the incoming item.
                self._bump("dropped")
                return False

            # drop_oldest: make room by evicting the oldest queued item...
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                pass
            self._bump("dropped")

            # ...then retry with the fresh item.
            try:
                self._queue.put_nowait(item)
            except Full:
                self._bump("dropped")
                return False

        self._bump("submitted")
        return True

    def raise_error(self) -> None:
        """Re-raise the first handler or callback failure on the calling thread."""

        with self._lock:
            error = self._error

        if error is not None:
            raise error

    def close(self, *, drain: bool = False, timeout: float = 5.0) -> None:
        """Stop accepting items and join the workers.

        Args:
            drain: Wait for the queued backlog before stopping. Items still
                queued after ``timeout`` are processed by the workers anyway
                unless they are stuck.
            timeout: Seconds allowed for draining and joining.

        Raises:
            RuntimeError: If a worker is still running after ``timeout``, which
                normally means a handler is stuck.
        """

        if self._closed:
            return
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive.")

        self._closed = True
        deadline = monotonic() + timeout

        # Optionally wait for the backlog to be consumed first.
        if drain:
            while self.pending and monotonic() < deadline:
                sleep(0.005)

        # Deliver one None sentinel per worker to make each loop exit.
        for _ in self._workers:
            while True:
                try:
                    self._queue.put(None, timeout=0.05)
                    break
                except Full:
                    if monotonic() >= deadline:
                        break

        # Join with the remaining budget; alive workers mean a stuck handler.
        for worker in self._workers:
            worker.join(timeout=max(0.0, deadline - monotonic()))

        if any(worker.is_alive() for worker in self._workers):
            raise RuntimeError("Offload workers did not stop within the timeout.")

    def _run(self) -> None:
        """Consume items until a sentinel arrives."""

        while True:
            item = self._queue.get()
            try:
                # None is the shutdown sentinel.
                if item is None:
                    return
                try:
                    result = self._handler(item)
                except BaseException as error:  # noqa: BLE001 - transported
                    self._record(error)
                    if self._on_error is not None:
                        self._guard(self._on_error, error)
                else:
                    self._bump("completed")
                    if self._on_result is not None:
                        self._guard(self._on_result, result)
            finally:
                # Always acknowledge so queue.join() never hangs.
                self._queue.task_done()

    def _guard(self, callback: Callable, value: object) -> None:
        """Run a user callback and record instead of losing its failure."""

        try:
            callback(value)
        except BaseException as error:  # noqa: BLE001 - transported
            self._record(error)

    def _record(self, error: BaseException) -> None:
        """Count a failure and keep the first one for ``raise_error()``."""

        self._bump("failed")
        with self._lock:
            if self._error is None:
                self._error = error

    def _bump(self, name: str) -> None:
        """Increment one counter under the shared lock."""

        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

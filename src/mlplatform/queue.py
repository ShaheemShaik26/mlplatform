from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from queue import PriorityQueue
from threading import Event, Thread
from typing import Any


@dataclass(order=True)
class QueueItem:
    priority: int
    job_id: str
    payload: dict[str, Any]


class AsyncJobQueue:
    def __init__(self) -> None:
        self._queue: PriorityQueue[QueueItem] = PriorityQueue()
        self._shutdown = Event()
        self._worker_thread: Thread | None = None
        self._handler: Callable[[str, dict[str, Any]], Any] | None = None

    def start(self, handler: Callable[[str, dict[str, Any]], Any]) -> None:
        self._handler = handler
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def submit(self, job_id: str, payload: dict[str, Any], priority: int = 0) -> None:
        self._queue.put(QueueItem(priority=-priority, job_id=job_id, payload=payload))

    def stop(self) -> None:
        self._shutdown.set()

    def _run_loop(self) -> None:
        asyncio.run(self._consume())

    async def _consume(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            if self._handler is None:
                continue
            result = self._handler(item.job_id, item.payload)
            if asyncio.iscoroutine(result):
                await result

import asyncio
import itertools

from vllm.entrypoints.openai.api_server import create_chat_completion  # type: ignore[import-not-found]
from fastapi import Request
from lmcache_vllm.batching.config import batch_settings
from lmcache_vllm.batching.models import QueuedChatCompletion
from lmcache_vllm.batching.utils import (
    parse_positive_int_header,
    parse_strategy_header,
    schedule_batch,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore[import-not-found]


class BatchScheduler:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._queue: list[QueuedChatCompletion] = []
        self._arrival_counter = itertools.count()
        self._timeout_handle: asyncio.TimerHandle | None = None

    async def submit(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        batch_size = parse_positive_int_header(
            raw_request,
            batch_settings.batch_size_header,
            batch_settings.default_batch_size,
        )
        batch_timeout_ms = parse_positive_int_header(
            raw_request,
            batch_settings.batch_timeout_header,
            batch_settings.default_batch_timeout_ms,
        )
        item = QueuedChatCompletion(
            request=request,
            raw_request=raw_request,
            future=future,
            arrival_order=next(self._arrival_counter),
            batch_size=batch_size,
            batch_timeout_s=batch_timeout_ms / 1000,
            strategy=parse_strategy_header(raw_request),
        )

        async with self._lock:
            self._queue.append(item)
            self._flush_ready_batch_locked(loop)

        return await future

    def _flush_ready_batch_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._queue:
            self._cancel_timeout_locked()
            return

        batch_size = self._queue[0].batch_size
        if len(self._queue) >= batch_size:
            self._cancel_timeout_locked()
            batch = self._pop_next_batch_locked(batch_size)
            loop.create_task(self._run_batch(batch))
            self._schedule_timeout_locked(loop)
            return

        self._schedule_timeout_locked(loop)

    def _pop_next_batch_locked(self, batch_size: int) -> list[QueuedChatCompletion]:
        batch = self._queue[:batch_size]
        del self._queue[:batch_size]
        return batch

    def _schedule_timeout_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._queue or self._timeout_handle is not None:
            return

        self._timeout_handle = loop.call_later(
            self._queue[0].batch_timeout_s,
            lambda: loop.create_task(self._flush_due_batch()),
        )

    def _cancel_timeout_locked(self) -> None:
        if self._timeout_handle is None:
            return

        self._timeout_handle.cancel()
        self._timeout_handle = None

    async def _flush_due_batch(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            self._timeout_handle = None
            if not self._queue:
                return

            batch_size = min(self._queue[0].batch_size, len(self._queue))
            batch = self._pop_next_batch_locked(batch_size)
            self._schedule_timeout_locked(loop)

        await self._run_batch(batch)

    async def _run_batch(self, batch: list[QueuedChatCompletion]) -> None:
        if not batch:
            return

        strategy = batch[0].strategy
        scheduled_batch = schedule_batch(batch, strategy)

        print(
            "v2 batched completion flush: "
            f"size={len(batch)} scheduler={strategy}",
            flush=True,
        )
        for item in scheduled_batch:
            if item.future.cancelled():
                continue

            try:
                response = await create_chat_completion(
                    item.request,
                    item.raw_request,
                )
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            else:
                if not item.future.done():
                    item.future.set_result(response)


batch_scheduler = BatchScheduler()

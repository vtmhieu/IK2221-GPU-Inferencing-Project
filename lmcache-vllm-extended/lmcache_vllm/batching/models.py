import asyncio
from dataclasses import dataclass

from fastapi import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore[import-not-found]


@dataclass
class QueuedChatCompletion:
    request: ChatCompletionRequest
    raw_request: Request
    future: asyncio.Future
    arrival_order: int
    batch_size: int
    batch_timeout_s: float
    strategy: str

from typing import Any

from fastapi import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore[import-not-found]

from lmcache_vllm.batching.config import batch_settings
from lmcache_vllm.batching.models import QueuedChatCompletion


def parse_positive_int_header(
    raw_request: Request,
    header_name: str,
    default: int,
) -> int:
    value = raw_request.headers.get(header_name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    return parsed if parsed > 0 else default


def parse_strategy_header(raw_request: Request) -> str:
    strategy = raw_request.headers.get(batch_settings.batch_strategy_header, "grouped")
    return strategy if strategy in batch_settings.valid_strategies else "grouped"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _context_key(item: QueuedChatCompletion) -> str:
    rag_context_id = getattr(item.raw_request.state, "rag_context_id", None)
    if rag_context_id:
        return str(rag_context_id)

    request = item.request
    for message in request.messages:
        if _message_role(message) == "user":
            content = _content_to_text(_message_content(message))
            return content.split(batch_settings.context_separator, 1)[-1]
    return request.model


def schedule_batch(
    batch: list[QueuedChatCompletion],
    strategy: str,
) -> list[QueuedChatCompletion]:
    if strategy == "none":
        return batch

    return sorted(
        batch,
        key=lambda item: (_context_key(item), item.arrival_order),
    )

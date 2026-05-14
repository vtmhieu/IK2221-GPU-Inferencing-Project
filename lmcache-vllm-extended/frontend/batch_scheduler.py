"""Simple batch scheduler for Task 2."""

from collections.abc import Sequence
from typing import TypeVar


RequestT = TypeVar("RequestT")
VALID_STRATEGIES = ("none", "grouped")


def _get_schedule_key(item: RequestT, key: str) -> str:
    return getattr(item, key)


def schedule_batch_with_positions(
    batch: Sequence[RequestT],
    strategy: str = "none",
    key: str = "context_id",
) -> list[tuple[int, RequestT]]:
    """Return scheduled ``(original_position, item)`` pairs for one batch."""
    if strategy not in VALID_STRATEGIES:
        valid = ", ".join(VALID_STRATEGIES)
        raise ValueError(f"Unknown scheduler strategy '{strategy}'. Use one of: {valid}")

    indexed_batch = list(enumerate(batch))
    if strategy == "none":
        return indexed_batch

    return sorted(indexed_batch, key=lambda pair: _get_schedule_key(pair[1], key))


def schedule_batch(
    batch: Sequence[RequestT],
    strategy: str = "none",
    key: str = "context_id",
) -> list[RequestT]:
    """Schedule one batch and return only the request items."""
    return [item for _, item in schedule_batch_with_positions(batch, strategy, key)]

from collections import OrderedDict
from threading import Lock
from typing import Any

from lmcache_vllm.rag.config import rag_settings


class RagMetricsStore:
    def __init__(self, max_entries: int):
        self._max_entries = max_entries
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def put(self, request_id: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            self._entries[request_id] = metadata
            self._entries.move_to_end(request_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            metadata = self._entries.get(request_id)
            if metadata is not None:
                self._entries.move_to_end(request_id)
            return metadata


rag_metrics_store = RagMetricsStore(rag_settings.max_metrics_entries)

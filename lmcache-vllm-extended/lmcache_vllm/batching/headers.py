from fastapi import Request

from lmcache_vllm.batching.config import batch_settings


def has_batch_headers(raw_request: Request) -> bool:
    return any(
        header_name in raw_request.headers
        for header_name in (
            batch_settings.batch_size_header,
            batch_settings.batch_timeout_header,
            batch_settings.batch_strategy_header,
        )
    )

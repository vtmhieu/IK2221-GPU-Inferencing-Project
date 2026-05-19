from fastapi import Request

from lmcache_vllm.rag.config import RagDocumentSet, rag_settings

TRUE_VALUES = {"1", "true", "yes", "on"}
VALID_DOCUMENT_SETS: set[RagDocumentSet] = {"base", "extended", "all"}


def has_rag_headers(raw_request: Request) -> bool:
    value = raw_request.headers.get(rag_settings.rag_enabled_header)
    return value is not None and value.lower() in TRUE_VALUES


def parse_rag_request_id(raw_request: Request) -> str | None:
    return raw_request.headers.get(rag_settings.rag_request_id_header)


def parse_document_set(raw_request: Request) -> RagDocumentSet:
    document_set = raw_request.headers.get(
        rag_settings.rag_document_set_header,
        rag_settings.default_document_set,
    )
    if document_set in VALID_DOCUMENT_SETS:
        return document_set
    return rag_settings.default_document_set

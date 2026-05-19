import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

RagDocumentSet = Literal["base", "extended", "all"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"


def _default_document_set() -> RagDocumentSet:
    value = os.getenv("LMCACHE_RAG_DOCSET", "base")
    if value == "extended":
        return "extended"
    if value == "all":
        return "all"
    return "base"


class RagSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    rag_enabled_header: str = "x-lmcache-rag"
    rag_request_id_header: str = "x-lmcache-rag-request-id"
    rag_document_set_header: str = "x-lmcache-rag-docset"
    context_separator: str = "###"
    assistant_ack: str = "Got it!"
    system_prompt: str = (
        "You are a helpful assistant. I will now give you a document and "
        "please answer my question afterwards based on the content in document"
    )
    embedding_model_name: str = os.getenv(
        "LMCACHE_RAG_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    embedding_device: str = os.getenv("LMCACHE_RAG_DEVICE", "cpu")
    max_embedding_tokens: int = int(os.getenv("LMCACHE_RAG_MAX_TOKENS", "512"))
    default_document_set: RagDocumentSet = _default_document_set()
    base_data_dir: Path = Path(
        os.getenv("LMCACHE_RAG_BASE_DATA_DIR", str(FRONTEND_ROOT / "data"))
    )
    extended_data_dir: Path = Path(
        os.getenv(
            "LMCACHE_RAG_EXTENDED_DATA_DIR",
            str(FRONTEND_ROOT / "additionaldata"),
        )
    )
    max_metrics_entries: int = int(os.getenv("LMCACHE_RAG_MAX_METRICS", "2000"))


rag_settings = RagSettings()

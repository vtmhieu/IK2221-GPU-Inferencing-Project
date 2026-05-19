from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    text: str
    source_path: str


@dataclass(frozen=True)
class RetrievalResult:
    document: DocumentRecord
    similarity: float
    retrieval_time_s: float
    document_count: int


@dataclass(frozen=True)
class EmbeddedDocuments:
    documents: list[DocumentRecord]
    embeddings: torch.Tensor

import time
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModel, AutoTokenizer

from lmcache_vllm.rag.config import RagDocumentSet, rag_settings
from lmcache_vllm.rag.models import DocumentRecord, EmbeddedDocuments, RetrievalResult
from lmcache_vllm.rag.utils import cosine_scores, mean_pool, normalize_embedding


class RagStore:
    def __init__(self):
        self._lock = Lock()
        self._tokenizer = None
        self._model = None
        self._embedded_documents: dict[RagDocumentSet, EmbeddedDocuments] = {}

    def retrieve(self, query: str, document_set: RagDocumentSet) -> RetrievalResult:
        start = time.perf_counter()
        embedded_documents = self._get_embedded_documents(document_set)
        query_embedding = self._embed_texts([query])
        scores = cosine_scores(query_embedding, embedded_documents.embeddings)
        best_index = int(torch.argmax(scores).item())
        retrieval_time_s = time.perf_counter() - start

        return RetrievalResult(
            document=embedded_documents.documents[best_index],
            similarity=float(scores[best_index].item()),
            retrieval_time_s=retrieval_time_s,
            document_count=len(embedded_documents.documents),
        )

    def _get_embedded_documents(self, document_set: RagDocumentSet) -> EmbeddedDocuments:
        with self._lock:
            embedded_documents = self._embedded_documents.get(document_set)
            if embedded_documents is not None:
                return embedded_documents

            documents = self._load_documents(document_set)
            if not documents:
                raise FileNotFoundError(
                    "No RAG documents found. Check LMCACHE_RAG_BASE_DATA_DIR "
                    "and LMCACHE_RAG_EXTENDED_DATA_DIR."
                )

            embeddings = self._embed_texts([document.text for document in documents])
            embedded_documents = EmbeddedDocuments(
                documents=documents,
                embeddings=embeddings,
            )
            self._embedded_documents[document_set] = embedded_documents
            return embedded_documents

    def _load_documents(self, document_set: RagDocumentSet) -> list[DocumentRecord]:
        directories = self._directories_for_set(document_set)
        records: list[DocumentRecord] = []
        seen_ids: set[str] = set()

        for directory in directories:
            if not directory.exists():
                continue

            for path in sorted(directory.glob("*.txt")):
                document_id = path.stem
                if document_id in seen_ids:
                    document_id = f"{directory.name}_{document_id}"
                seen_ids.add(document_id)
                records.append(
                    DocumentRecord(
                        document_id=document_id,
                        text=path.read_text(encoding="utf-8", errors="replace"),
                        source_path=str(path),
                    )
                )

        return records

    def _directories_for_set(self, document_set: RagDocumentSet) -> list[Path]:
        if document_set == "base":
            return [rag_settings.base_data_dir]
        if document_set == "extended":
            return [rag_settings.extended_data_dir]
        return [rag_settings.base_data_dir, rag_settings.extended_data_dir]

    def _load_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(rag_settings.embedding_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(rag_settings.embedding_model_name)
        self._model.to(rag_settings.embedding_device)
        self._model.eval()

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        self._load_model()
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=rag_settings.max_embedding_tokens,
            return_tensors="pt",
        )
        inputs = inputs.to(rag_settings.embedding_device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])

        return normalize_embedding(pooled)


rag_store = RagStore()

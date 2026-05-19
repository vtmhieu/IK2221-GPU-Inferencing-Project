from fastapi import Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore[import-not-found]

from lmcache_vllm.rag.config import rag_settings
from lmcache_vllm.rag.headers import parse_document_set, parse_rag_request_id
from lmcache_vllm.rag.metrics import rag_metrics_store
from lmcache_vllm.rag.store import rag_store
from lmcache_vllm.rag.utils import latest_user_message_text


def apply_rag_to_request(
    request: ChatCompletionRequest,
    raw_request: Request,
) -> ChatCompletionRequest:
    question = latest_user_message_text(request.messages)
    document_set = parse_document_set(raw_request)
    retrieval = rag_store.retrieve(question, document_set)

    raw_request.state.rag_context_id = retrieval.document.document_id
    raw_request.state.rag_similarity = retrieval.similarity
    raw_request.state.rag_retrieval_time_s = retrieval.retrieval_time_s
    raw_request.state.rag_document_count = retrieval.document_count

    request_id = parse_rag_request_id(raw_request)
    if request_id is not None:
        raw_request.state.rag_request_id = request_id
        rag_metrics_store.put(
            request_id,
            {
                "request_id": request_id,
                "predicted_context_id": retrieval.document.document_id,
                "similarity": round(retrieval.similarity, 6),
                "retrieval_time_s": round(retrieval.retrieval_time_s, 6),
                "document_count": retrieval.document_count,
                "document_set": document_set,
                "source_path": retrieval.document.source_path,
            },
        )

    request_data = request.model_dump()
    request_data["messages"] = _build_rag_messages(question, retrieval.document.text)
    return ChatCompletionRequest.model_validate(request_data)


def _build_rag_messages(question: str, context_text: str) -> list[dict[str, str]]:
    context_message = rag_settings.context_separator.join(
        [rag_settings.system_prompt, context_text]
    )
    return [
        {"role": "user", "content": context_message},
        {"role": "assistant", "content": rag_settings.assistant_ack},
        {"role": "user", "content": question},
    ]

import vllm.entrypoints.openai.api_server as base_api  # type: ignore[import-not-found]
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from lmcache_vllm.batching import batch_scheduler
from lmcache_vllm.batching.headers import has_batch_headers
from lmcache_vllm.rag import apply_rag_to_request, has_rag_headers
from lmcache_vllm.rag.metrics import rag_metrics_store
from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore[import-not-found]

# You should use the following file to implement all APIs you may require in the project.
# Note that the two important ones are already implemented here simply by calling the default v1 implementation in VLLM.
# You may need to modify these functions to enable pre-processing of requests, before running the inference.

extended_router = APIRouter()


@extended_router.get("/models")
async def show_available_models(request: Request):
    print("v2 models is called!")
    return await base_api.show_available_models(request)


@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    if has_rag_headers(raw_request):
        print("v2 completion is called with RAG")
        request = apply_rag_to_request(request, raw_request)

    if has_batch_headers(raw_request):
        print("v2 completion is called with batch headers")
        return await batch_scheduler.submit(request, raw_request)

    print("v2 completion is called")
    return await base_api.create_chat_completion(request, raw_request)


@extended_router.get("/rag/metrics/{request_id}")
async def get_rag_metrics(request_id: str):
    metadata = rag_metrics_store.get(request_id)
    if metadata is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No RAG metrics found for request_id '{request_id}'"},
        )
    return metadata

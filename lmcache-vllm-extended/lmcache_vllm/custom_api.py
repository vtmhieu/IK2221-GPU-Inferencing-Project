import vllm.entrypoints.openai.api_server as base_api  # type: ignore[import-not-found]
from fastapi import APIRouter, Request
from lmcache_vllm.batching import batch_scheduler
from vllm.entrypoints.openai.protocol import ChatCompletionRequest  # type: ignore[import-not-found]

# You should use the following file to implement all APIs you may require in the project.
# Note that the two important ones are already implemented here simply by calling the default v1 implementation in VLLM.
# You may need to modify these functions to enable pre-processing of requests, before running the inference.

extended_router = APIRouter()

@extended_router.get("/models")
@extended_router.get("/batch/models")
async def show_available_models(request: Request):
    print("v2 models is called!")
    return await base_api.show_available_models(request)


@extended_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    print("v2 completion is called")
    return await base_api.create_chat_completion(request, raw_request)


@extended_router.post("/batch/chat/completions")
@extended_router.post("/chat/completions/batched")
async def create_batched_chat_completion(request: ChatCompletionRequest,
                                         raw_request: Request):
    print("v2 batched completion is called")
    return await batch_scheduler.submit(request, raw_request)

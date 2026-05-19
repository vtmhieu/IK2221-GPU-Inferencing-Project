from typing import Any

import torch
import torch.nn.functional as F


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    return str(content)


def message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def latest_user_message_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if message_role(message) == "user":
            return content_to_text(message_content(message))
    return ""


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    return F.normalize(embedding.detach().cpu(), p=2, dim=-1)


def cosine_scores(query_embedding: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.matmul(document_embeddings, query_embedding.squeeze(0))

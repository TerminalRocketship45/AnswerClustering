from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from .llm_client import embed_texts, generate_text


def chunk_text(text: str, max_words: int = 120) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    while words:
        chunk = words[:max_words]
        chunks.append(" ".join(chunk))
        words = words[max_words:]
    return chunks


def build_document_store(documents: List[str], model_name: str) -> List[Dict[str, object]]:
    chunks: List[str] = []
    for document in documents:
        chunks.extend(chunk_text(document, max_words=120))
    embeddings = embed_texts(chunks, model=model_name)
    return [
        {"text": text, "embedding": embedding}
        for text, embedding in zip(chunks, embeddings)
    ]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_context(
    query: str,
    store: List[Dict[str, object]],
    model_name: str,
    top_k: int = 3,
) -> str:
    if not store:
        return ""
    query_embedding = embed_texts([query], model=model_name)[0]
    scored: List[tuple[float, str]] = []
    for item in store:
        similarity = _cosine_similarity(query_embedding, item["embedding"])
        scored.append((similarity, item["text"]))
    scored.sort(reverse=True, key=lambda pair: pair[0])
    return "\n---\n".join(text for _, text in scored[:top_k])


def search_fallback(query: str, model_name: str) -> str:
    prompt = (
        "You are a grounded research assistant. Based on general domain knowledge and publicly available information, "
        "provide a concise summary of facts that are relevant to evaluating the following batch of ideas. "
        "Do not invent specific internal data. "
        f"Query: {query}"
    )
    return generate_text(prompt, model=model_name, max_output_tokens=400)

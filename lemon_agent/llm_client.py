from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Iterable, List, Optional

from dotenv import load_dotenv

try:
    import google.generativeai as genai
    from google.generativeai import types
except ImportError as exc:
    raise ImportError(
        "The Lemon Agent requires google-generativeai. Install it with 'pip install google-generativeai'."
    ) from exc

load_dotenv()

API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_RETRIES = 3
BASE_BACKOFF = 1.5

_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"{API_KEY_ENV} is not set. Put your Gemini API key in .env or the environment."
        )

    if hasattr(genai, "configure"):
        genai.configure(api_key=api_key)
    _configured = True


def _is_rate_limit_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in ["rate limit", "rate_limit", "429", "too many requests", "quota", "unavailable"]
    )


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, dict) and "text" in response:
        return response["text"]
    return str(response)


def _call_llm(
    prompt: str,
    model: str,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 1600,
    temperature: float = 0.3,
) -> str:
    _ensure_configured()
    attempt = 0
    while True:
        attempt += 1
        try:
            model_obj = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction or "You are a rigorous decision-analysis expert.",
            )
            response = model_obj.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
            return _extract_text(response)
        except Exception as exc:
            if attempt >= MAX_RETRIES or not _is_rate_limit_error(exc):
                raise
            backoff = BASE_BACKOFF * (2 ** (attempt - 1))
            logging.warning("LLM rate limit or transient error, retrying in %.1fs: %s", backoff, exc)
            time.sleep(backoff)


def generate_text(
    prompt: str,
    model: str,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 1600,
    temperature: float = 0.3,
) -> str:
    return _call_llm(
        prompt,
        model=model,
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def parse_json_response(response_text: str) -> Any:
    code_block = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", response_text, re.S)
    if code_block:
        json_text = code_block.group(1).strip()
    else:
        start = response_text.find("[")
        end = response_text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"Could not locate JSON array in response:\n{response_text}")
        json_text = response_text[start : end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON output. Raw response:\n{response_text}"
        ) from exc


def generate_json(
    prompt: str,
    model: str,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 1600,
) -> Any:
    raw = _call_llm(prompt, model=model, system_instruction=system_instruction, max_output_tokens=max_output_tokens)
    try:
        return parse_json_response(raw)
    except ValueError as exc:
        logging.warning("JSON parse failed on first response, retrying once. Error: %s", exc)
        raw = _call_llm(prompt, model=model, system_instruction=system_instruction, max_output_tokens=max_output_tokens)
        return parse_json_response(raw)


def embed_texts(texts: Iterable[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    _ensure_configured()
    response = genai.embed_content(model=model, input=list(texts))
    data = None
    if isinstance(response, dict):
        data = response.get("data")
    elif hasattr(response, "data"):
        data = response.data
    else:
        data = response
    if not data:
        raise ValueError("Embedding response did not contain data.")

    embeddings: List[List[float]] = []
    for item in data:
        if isinstance(item, dict):
            embedding = item.get("embedding")
        else:
            embedding = getattr(item, "embedding", None)
        if embedding is None:
            raise ValueError("Embedding item missing embedding vector.")
        embeddings.append(list(embedding))
    return embeddings

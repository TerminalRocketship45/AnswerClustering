from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

from dotenv import load_dotenv

from .config import config

try:
    import google.generativeai as genai
    from google.generativeai import types
    _google_import_error: Optional[ImportError] = None
except ImportError as exc:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
    _google_import_error = exc

load_dotenv()

API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = config["model"]
EMBEDDING_MODEL = config["embeddings_model"]
MAX_RETRIES = config["retry_attempts"]
BASE_BACKOFF = config["retry_backoff_seconds"]

_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return

    if genai is None:
        raise ImportError(
            "The Lemon Agent requires google-generativeai. Install it with 'pip install google-generativeai'."
        ) from _google_import_error

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
    max_output_tokens: int = config["max_llm_tokens"],
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
    max_output_tokens: int = config["max_llm_tokens"],
    temperature: float = 0.3,
) -> str:
    return _call_llm(
        prompt,
        model=model,
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def _save_parse_failure(response_text: str, prompt: str, model: str) -> str:
    failures_dir = Path(__file__).resolve().parent / "parse_failures"
    failures_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    safe_model = re.sub(r"[^A-Za-z0-9_-]", "_", model)
    filename = f"{timestamp}_{safe_model}_json_parse_failure.json"
    failure_path = failures_dir / filename
    with open(failure_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model,
                "prompt": prompt,
                "response": response_text,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return str(failure_path)


def _try_parse_truncated_json(json_text: str) -> Optional[Any]:
    decoder = json.JSONDecoder()
    text = json_text.strip()
    if not text:
        return None

    for end in range(len(text), 0, -1):
        if text[end - 1] not in ']}"0123456789lsef':
            continue
        candidate = text[:end].rstrip()
        try:
            obj, index = decoder.raw_decode(candidate)
            if candidate[index:].strip() == "":
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _close_truncated_json(json_text: str) -> str:
    in_string = False
    escaped = False
    stack: list[str] = []

    for char in json_text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            if in_string:
                escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "[{":
            stack.append(char)
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()

    fixed = json_text
    if in_string:
        fixed += '"'
    for open_char in reversed(stack):
        fixed += "]" if open_char == "[" else "}"
    return fixed


def parse_json_response(response_text: str) -> Any:
    code_block = re.search(r"```json\s*(.*?)(?:\s*```|$)", response_text, re.S | re.I)
    if code_block:
        json_text = code_block.group(1).strip()
    else:
        start = response_text.find("[")
        brace_start = response_text.find("{")
        if start == -1 and brace_start == -1:
            raise ValueError(f"Could not locate JSON array in response:\n{response_text}")
        if start == -1:
            start = brace_start
        elif brace_start != -1:
            start = min(start, brace_start)
        json_text = response_text[start:].strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        closed_json = _close_truncated_json(json_text)
        try:
            return json.loads(closed_json)
        except json.JSONDecodeError:
            salvaged = _try_parse_truncated_json(closed_json)
            if salvaged is not None:
                return salvaged
            last_close = max(closed_json.rfind("]"), closed_json.rfind("}"))
            if last_close != -1:
                candidate = closed_json[: last_close + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Failed to parse JSON output. Raw response:\n{response_text}"
            )


def generate_json(
    prompt: str,
    model: str,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = config["max_llm_tokens"],
    return_raw: bool = False,
) -> Any:
    raw = _call_llm(prompt, model=model, system_instruction=system_instruction, max_output_tokens=max_output_tokens)
    try:
        parsed = parse_json_response(raw)
    except ValueError as exc:
        failure_path = _save_parse_failure(raw, prompt, model)
        raise ValueError(
            f"Failed to parse JSON output. Raw response saved to: {failure_path}"
        ) from exc
    return (raw, parsed) if return_raw else parsed


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

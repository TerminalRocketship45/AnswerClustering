"""
LLM client — wraps the Kimi API (Moonshot AI) via the OpenAI-compatible SDK.

Mirrors the Lisp helpers:
  (askLLM prompt)             -> sends a prompt, returns the raw text response
  (parse-llm-output response) -> extracts and parses the JSON block

Set KIMI_API_KEY in the environment before running:
  Windows PowerShell : $env:KIMI_API_KEY = "sk-..."
  Or hardcode below  : replace the os.environ.get(...) call with your key string
"""

import json
import os
import re
from typing import Any

from openai import OpenAI

KIMI_BASE_URL = "https://api.moonshot.ai/v1"

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client(api_key: str | None = None) -> OpenAI:
    global _client
    if _client is None:
        key = api_key or os.environ.get("KIMI_API_KEY")
        if not key:
            raise EnvironmentError(
                "KIMI_API_KEY environment variable is not set.\n"
                "Set it in PowerShell with:  $env:KIMI_API_KEY = 'sk-...'"
            )
        _client = OpenAI(api_key=key, base_url=KIMI_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# ask_llm  (Lisp: askLLM)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a rigorous decision-analysis critic specialising in identifying "
    "failure modes for proposed solutions. When asked, you reason carefully "
    "about each evaluation criterion and produce well-structured JSON output "
    "exactly matching the schema requested. Do not add commentary outside the "
    "JSON code block."
)


def ask_llm(
    prompt: str,
    model: str = "moonshot-v1-8k",
    api_key: str | None = None,
) -> str:
    """
    Send `prompt` to the Kimi API and return the raw text response.

    Parameters
    ----------
    prompt  : the user-facing prompt built by _build_prompt()
    model   : Kimi model ID — options: moonshot-v1-8k / moonshot-v1-32k /
              moonshot-v1-128k / moonshot-v1-auto
    api_key : optional key override; falls back to KIMI_API_KEY env var
    """
    client = _get_client(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# parse_llm_output  (Lisp: parse-llm-output)
# ---------------------------------------------------------------------------

def parse_llm_output(response: str) -> list[dict[str, Any]]:
    """
    Extract the JSON array from the LLM response and parse it.

    The model is instructed to wrap output in a ```json ... ``` fence.
    Falls back to locating the first '[' … ']' span if no fence is present.
    """
    fence_match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        start = response.find("[")
        end   = response.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(
                f"Could not locate a JSON array in the LLM response:\n{response}"
            )
        json_str = response[start : end + 1]

    return json.loads(json_str)

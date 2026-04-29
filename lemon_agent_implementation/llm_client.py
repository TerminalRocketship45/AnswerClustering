"""
LLM client — supports OpenAI, Kimi, and Gemini.

Provider selection (in priority order):
  1. Explicit:  set LLM_PROVIDER=openai | kimi | gemini
  2. Auto-detect from whichever API key is present:
       OPENAI_API_KEY  -> OpenAI
       KIMI_API_KEY    -> Kimi
       GEMINI_API_KEY  -> Gemini

PowerShell examples:
  $env:OPENAI_API_KEY  = "sk-..."          # uses OpenAI automatically
  $env:KIMI_API_KEY    = "sk-..."          # uses Kimi automatically
  $env:GEMINI_API_KEY  = "AIza..."         # uses Gemini automatically

  # Force a specific provider when multiple keys are set:
  $env:LLM_PROVIDER = "gemini"
  $env:GEMINI_API_KEY = "AIza..."
"""

import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

OPENAI = "openai"
KIMI   = "kimi"
GEMINI = "gemini"

KIMI_BASE_URL = "https://api.moonshot.ai/v1"

DEFAULT_MODELS = {
    OPENAI: "gpt-4o-mini",
    KIMI:   "moonshot-v1-8k",
    GEMINI: "gemini-2.5-flash-lite",
}

SYSTEM_PROMPT = (
    "You are a rigorous decision-analysis critic specialising in identifying "
    "failure modes for proposed solutions. When asked, you reason carefully "
    "about each evaluation criterion and produce well-structured JSON output "
    "exactly matching the schema requested. Do not add commentary outside the "
    "JSON code block."
)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str]:
    """
    Return (provider, api_key).

    Checks LLM_PROVIDER first for an explicit choice, then falls back to
    whichever API key env var is set.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if explicit == OPENAI:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise EnvironmentError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        return OPENAI, key

    if explicit == KIMI:
        key = os.environ.get("KIMI_API_KEY", "")
        if not key:
            raise EnvironmentError("LLM_PROVIDER=kimi but KIMI_API_KEY is not set.")
        return KIMI, key

    if explicit == GEMINI:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set.")
        return GEMINI, key

    if explicit and explicit not in (OPENAI, KIMI, GEMINI):
        raise EnvironmentError(
            f"Unknown LLM_PROVIDER '{explicit}'. Choose: openai, kimi, or gemini."
        )

    # Auto-detect from whichever key is present (first match wins)
    if os.environ.get("OPENAI_API_KEY"):
        return OPENAI, os.environ["OPENAI_API_KEY"]
    if os.environ.get("KIMI_API_KEY"):
        return KIMI, os.environ["KIMI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return GEMINI, os.environ["GEMINI_API_KEY"]

    raise EnvironmentError(
        "No API key found. Set one of:\n"
        "  $env:OPENAI_API_KEY  = 'sk-...'   (OpenAI)\n"
        "  $env:KIMI_API_KEY    = 'sk-...'   (Kimi)\n"
        "  $env:GEMINI_API_KEY  = 'AIza...'  (Gemini)\n"
        "Or set $env:LLM_PROVIDER to force a specific provider."
    )

# ---------------------------------------------------------------------------
# ask_llm  (Lisp: askLLM)
# ---------------------------------------------------------------------------

def ask_llm(prompt: str, model: str | None = None) -> str:
    """
    Send `prompt` to the active LLM provider and return the raw text response.

    The provider and API key are resolved automatically from environment
    variables — see module docstring for details.

    Parameters
    ----------
    prompt : the user-facing prompt built by _build_prompt()
    model  : optional model override; if omitted, the provider default is used
    """
    provider, api_key = _detect_provider()
    chosen_model = model or DEFAULT_MODELS[provider]

    print(f"[llm_client] provider={provider}  model={chosen_model}")

    if provider in (OPENAI, KIMI):
        from openai import OpenAI
        base_url = KIMI_BASE_URL if provider == KIMI else None
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.3,
        )
        return response.choices[0].message.content

    if provider == GEMINI:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=chosen_model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=4096,
            ),
            contents=prompt,
        )
        return response.text

    raise RuntimeError(f"Unhandled provider: {provider}")  # unreachable


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

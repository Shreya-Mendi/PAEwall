"""
Shared LLM client factory.

Prefers Duke OIT LLM proxy (OpenAI-compatible, no billing).
Falls back to Anthropic if ANTHROPIC_API_KEY is set.
Returns None if neither is configured — all callers fall back to rule-based logic.
"""

import os


DUKE_LLM_BASE_URL = "https://litellm.oit.duke.edu/v1"
DUKE_LLM_MODEL    = "GPT 4.1 Mini"


def get_openai_client():
    """Return an openai.OpenAI pointed at Duke LLM, or None."""
    key = os.environ.get("DUKE_LLM_API_KEY")
    if not key:
        return None, None
    try:
        from openai import OpenAI
        base = os.environ.get("DUKE_LLM_BASE_URL", DUKE_LLM_BASE_URL)
        model = os.environ.get("DUKE_LLM_MODEL", DUKE_LLM_MODEL)
        return OpenAI(api_key=key, base_url=base), model
    except ImportError:
        return None, None


def chat(messages: list[dict], system: str | None = None,
         max_tokens: int = 1024, json_mode: bool = False) -> str | None:
    """
    Single-turn chat call. Returns text response or None on failure.

    Uses Duke LLM when available; rule-based callers handle None.
    """
    client, model = get_openai_client()
    if client is None:
        return None

    all_msgs = []
    if system:
        all_msgs.append({"role": "system", "content": system})
    all_msgs.extend(messages)

    kwargs = dict(model=model, messages=all_msgs, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception:
        return None

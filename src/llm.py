"""
Multi-provider LLM client with an ORDERED FALLBACK CHAIN.

The pipeline's real LLM path (classify_intent, draft_reply, detect_safety, judge) routes through
this module. `chat()` tries each configured provider in priority order and returns the first
success; if a provider errors (rate limit, auth, timeout, empty output) it falls through to the
next. If every provider fails, it raises — and the pipeline then falls back to the transparent
rule path, which is always REPORTED in the output's "method" field (never silently hidden).

Priority chain (set keys in .env; any subset works):
  1. GROQ_API_KEY        -> Groq        (openai/gpt-oss-120b)      primary
  2. GEMINI_API_KEY      -> Google Gemini (gemini-2.0-flash)       fallback
  3. OPENROUTER_API_KEY  -> OpenRouter  (llama-3.3-70b :free)
  4. OPENAI_API_KEY      -> OpenAI      (gpt-4o-mini; paid)

`last_provider()` reports which provider actually produced the most recent response, so the UI /
method field shows the truth (e.g. groq quota out -> "llm:gemini").
"""
from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# (env_var, base_url, default_model, provider_name)  — order = fallback priority
_PROVIDERS = [
    ("GROQ_API_KEY",       "https://api.groq.com/openai/v1",                     "openai/gpt-oss-120b", "groq"),
    ("GEMINI_API_KEY",     "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.6-flash", "gemini"),
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",                       "meta-llama/llama-3.3-70b-instruct:free", "openrouter"),
    ("OPENAI_API_KEY",     None,                                                 "gpt-4o-mini", "openai"),
]

_LAST_PROVIDER: str | None = None


@lru_cache(maxsize=1)
def _chain():
    """Build the ordered list of (client, model, provider) for every provider with a key."""
    chain = []
    for env_var, base_url, default_model, provider in _PROVIDERS:
        key = os.getenv(env_var)
        if not key:
            continue
        from openai import OpenAI
        kwargs = {"api_key": key, "max_retries": 0, "timeout": 30.0}  # fail fast -> next provider
        if base_url:
            kwargs["base_url"] = base_url
        model = os.getenv("LLM_MODEL", default_model) if provider == "groq" else default_model
        chain.append((OpenAI(**kwargs), model, provider))
    return chain


def llm_available() -> bool:
    return len(_chain()) > 0


def provider_name() -> str | None:
    """Primary (first) configured provider — used for the header label."""
    ch = _chain()
    return ch[0][2] if ch else None


def provider_chain() -> list[str]:
    return [p for _, _, p in _chain()]


def last_provider() -> str | None:
    """Provider that produced the most recent successful response (for honest method reporting)."""
    return _LAST_PROVIDER


def _one_call(client, model, messages, temperature, max_tokens, json_mode):
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if "gpt-oss" in model:  # groq reasoning model: cap reasoning so it doesn't eat the token budget
        kwargs["reasoning_effort"] = "low"
    if json_mode:
        try:
            resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content
        except Exception:
            pass  # some models reject/empty strict json mode -> retry without it
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("empty completion")
    if json_mode and "{" not in content:
        # this provider returned prose instead of JSON -> treat as failure so chain tries next
        raise RuntimeError("json_mode response contained no JSON object")
    return content


def chat(messages, temperature: float = 0.0, max_tokens: int = 512, json_mode: bool = False) -> str:
    """Try each provider in the chain until one succeeds. Raises if all fail / none configured."""
    global _LAST_PROVIDER
    chain = _chain()
    if not chain:
        raise RuntimeError("No LLM provider key found. Set GROQ_API_KEY and/or GEMINI_API_KEY in .env.")
    last_err = None
    for client, model, provider in chain:
        try:
            out = _one_call(client, model, messages, temperature, max_tokens, json_mode)
            _LAST_PROVIDER = provider
            return out
        except Exception as e:
            last_err = e
            continue  # provider failed (rate limit / auth / timeout) -> next in chain
    raise RuntimeError(f"all LLM providers failed; last error: {type(last_err).__name__}: {last_err}")


if __name__ == "__main__":
    if llm_available():
        print("Provider chain:", " -> ".join(provider_chain()))
        print("Reply:", chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=10))
        print("Answered by:", last_provider())
    else:
        print("No LLM key configured. Pipeline will use the rule-based baseline (mode='rules').")

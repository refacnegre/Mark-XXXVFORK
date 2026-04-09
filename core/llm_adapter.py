import json
import re
from typing import Any

from core.minimax_client import MiniMaxClient
from memory.config_manager import get_minimax_key

DEFAULT_MODEL = "MiniMax-M2.5"


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"```(?:json|python)?", "", text).strip().rstrip("`").strip()
    return text


def _client(model: str = DEFAULT_MODEL) -> MiniMaxClient:
    key = get_minimax_key()
    if not key:
        raise RuntimeError("minimax_api_key not found in config/api_keys.json")
    return MiniMaxClient(api_key=key, model=model)


def complete_text(
    prompt: str,
    *,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    data = _client(model=model).chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("MiniMax returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                text_parts.append(p.get("text", ""))
        content = "\n".join(t for t in text_parts if t)

    text = _clean(str(content or ""))
    if not text:
        raise RuntimeError("MiniMax returned empty content")
    return text


def complete_json(
    prompt: str,
    *,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    text = complete_text(
        prompt,
        system_instruction=system_instruction,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return json.loads(_clean(text))

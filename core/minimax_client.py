import json
from typing import Any

import requests


class MiniMaxClient:
    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M2.5",
        endpoint: str = "https://api.minimax.io/v1/text/chatcompletion_v2",
        timeout: int = 90,
    ):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        res = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
        )
        res.raise_for_status()
        data = res.json()

        # Some MiniMax responses wrap in base_resp even on 200.
        base_resp = data.get("base_resp") or {}
        if isinstance(base_resp, dict) and base_resp.get("status_code", 0) not in (0, None):
            raise RuntimeError(base_resp.get("status_msg") or str(base_resp))

        return data

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(dotenv_path=Path("/Users/pavanpurohit/Documents/Adobe/leadership_insight_agent/.env"), override=True)



@dataclass
class LLMClient:
    model: str
    client: OpenAI

    @classmethod
    def from_env(cls, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> "LLMClient":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        return cls(model=model, client=client)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

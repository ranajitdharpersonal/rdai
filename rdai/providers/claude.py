import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class ClaudeProvider(BaseProvider):
    traits = ["creative", "analytical", "reasoning"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "claude-3-haiku-20240307"):
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("Claude API key is missing.")
            
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
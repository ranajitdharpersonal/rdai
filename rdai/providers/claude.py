import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class ClaudeProvider(BaseProvider):
    traits = ["creative", "analytical", "reasoning"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)

    def fallback_models(self):
        return ("claude-3-haiku-20240307",)

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
        
        # 🎯 FIX: Added explicit timeout parameter to prevent hanging
        timeout_val = kwargs.get("timeout", 15.0)
        response = requests.post(
            "https://api.anthropic.com/v1/messages", 
            headers=headers, 
            json=data, 
            timeout=timeout_val
        )
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
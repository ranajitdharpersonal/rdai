import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class MistralProvider(BaseProvider):
    traits = ["european", "efficient", "open-weights"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "mistral-large-latest"):
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("Mistral API key is missing.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
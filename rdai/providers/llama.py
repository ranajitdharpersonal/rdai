import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class LlamaProvider(BaseProvider):
    traits = ["open-source", "meta", "fast"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
        # Defaulting to Together AI endpoint for Llama models
        self.endpoint = "https://api.together.xyz/v1/chat/completions"

    def fallback_models(self):
        return ("meta-llama/Llama-3-70b-chat-hf",) 

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("Llama API key is missing.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # 🎯 FIX: Added explicit timeout parameter
        timeout_val = kwargs.get("timeout", 15.0)
        response = requests.post(self.endpoint, headers=headers, json=data, timeout=timeout_val)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
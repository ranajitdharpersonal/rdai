import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class DeepseekProvider(BaseProvider):
    traits = ["coding", "logic", "fast"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "deepseek-chat"):
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("DeepSeek API key is missing.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}", 
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model, 
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
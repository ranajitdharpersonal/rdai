import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class QwenProvider(BaseProvider):
    traits = ["multilingual", "efficient"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "qwen-turbo"):
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("Qwen API key is missing.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}", 
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": prompt}]}
        }
        
        # 🎯 FIX: Added explicit timeout parameter
        timeout_val = kwargs.get("timeout", 15.0)
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation", 
            headers=headers, 
            json=data,
            timeout=timeout_val
        )
        response.raise_for_status()
        
        return response.json()["output"]["text"]
import requests
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class HuggingfaceProvider(BaseProvider):
    traits = ["open-source", "flexible", "community"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.2"):
        # HF e model nam mane repository nam
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("HuggingFace API key is missing.")
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Instruction format for models
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # 🎯 FIX: Added explicit timeout parameter to prevent hanging
        timeout_val = kwargs.get("timeout", 15.0)
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model}",
            headers=headers,
            json={"inputs": formatted_prompt, "parameters": {"max_new_tokens": 512}},
            timeout=timeout_val
        )
        response.raise_for_status()
        
        # HF returns a list of dictionaries
        result_text = response.json()[0].get("generated_text", "")
        
        # Clean the prompt from the response
        return result_text.replace(formatted_prompt, "").strip()
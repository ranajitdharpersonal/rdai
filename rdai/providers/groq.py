from typing import Any, Optional
from rdai.providers.base import BaseProvider

class GroqProvider(BaseProvider):
    traits = ["ultra-fast", "groq-lpu", "low-latency"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)

    def fallback_models(self):
        return ("llama-3.1-8b-instant",)
    
    def available_models(self):
        if not self.is_available:
            return ()

        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            response = client.models.list()

            return tuple(
                model.id
                for model in response.data
                if getattr(model, "active", True)
                )
        except Exception:
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("Groq API key is missing.")

        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Please install the Groq SDK using: pip install groq")

        client = Groq(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return response.choices[0].message.content
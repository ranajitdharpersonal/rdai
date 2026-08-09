from typing import Any, Optional
from rdai.providers.base import BaseProvider

class GroqProvider(BaseProvider):
    traits = ["ultra-fast", "groq-lpu", "low-latency"]

    # 🎯 FIX: Updated default model before the Aug 16 shutdown
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "llama3-70b-8192"):
        super().__init__(api_key, model)

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
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class OpenAIProvider(BaseProvider):
    traits = ["coding", "reasoning", "industry-standard"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "gpt-4o-mini"):
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("OpenAI API key is missing.")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the OpenAI SDK using: pip install openai")

        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return response.choices[0].message.content
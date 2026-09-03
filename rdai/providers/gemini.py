from typing import Any, Optional
from google import genai
from rdai.providers.base import BaseProvider

class GeminiProvider(BaseProvider):
    """Gemini Provider using the latest google.genai SDK."""
    
    traits = ["coding", "reasoning", "multimodal", "fast"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__(api_key, model)
    # Client is initialized only if the API key is present
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def fallback_models(self):
        return ("gemini-2.5-pro",)

    @property
    def is_available(self) -> bool:
        """Check if Gemini is fully configured and ready."""
        return bool(self.api_key and self.client)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise RuntimeError("🚨 Gemini API Key is missing! Cannot generate content.")
            
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            **kwargs
        )
        return response.text
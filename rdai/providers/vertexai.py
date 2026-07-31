from typing import Any, Optional
from rdai.providers.base import BaseProvider
from google import genai

class VertexaiProvider(BaseProvider):
    traits = ["enterprise", "secure", "multimodal"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "gemini-2.5-flash"):
        # api_key is treated as GCP Project ID here
        super().__init__(api_key, model)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ValueError("GCP Project ID is missing for VertexAI.")
            
        # Standard Vertex AI client via modern google.genai
        client = genai.Client(vertexai=True, project=self.api_key, location="us-central1")
        
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text
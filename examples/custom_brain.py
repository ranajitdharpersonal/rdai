from rdai.providers.base import BaseProvider
from rdai import AI
import requests

# User tar nijer private ba notun custom provider banacche
class NexusCustomProvider(BaseProvider):
    traits = ["reasoning", "proprietary"]
    
    def __init__(self, api_key=None, model="nexus-v1-pro"):
        super().__init__(api_key, model)

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        # Custom API logic here...
        # Example: res = requests.post("https://api.nexus.com/v1", ...)
        return f"[NexusAI] Simulated response for: {prompt}"

# Nijer AI instance e inject kore dilo!
ai = AI(strategy="manual", providers=[NexusCustomProvider(api_key="secret_key")])
print(ai.generate("How to structure a Next.js app?"))
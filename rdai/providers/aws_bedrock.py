import json
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class AwsBedrockProvider(BaseProvider):
    traits = ["aws", "enterprise", "stable"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
    # AWS uses AWS_ACCESS_KEY_ID from env.
    # api_key here acts as region_name if provided via .env
        super().__init__(api_key, model)

    def fallback_models(self):
        return ("anthropic.claude-3-haiku-20240307-v1:0",)

    @property
    def is_available(self) -> bool:
        # 🎯 FIX: Actually check for AWS credentials, not just library presence
        try:
            import boto3
            session = boto3.Session()
            return session.get_credentials() is not None
        except ImportError:
            return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ImportError("AWS Bedrock credentials missing or boto3 not installed.")
            
        import boto3
        # Use api_key as region_name, fallback to us-east-1
        client = boto3.client("bedrock-runtime", region_name=self.api_key or "us-east-1")
        
        response = client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}]
        )
        
        return response["output"]["message"]["content"][0]["text"]
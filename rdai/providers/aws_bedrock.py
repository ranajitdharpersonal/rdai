import json
from typing import Any, Optional
from rdai.providers.base import BaseProvider

class AwsBedrockProvider(BaseProvider):
    traits = ["aws", "enterprise", "stable"]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "anthropic.claude-3-haiku-20240307-v1:0"):
        # AWS typically uses credentials from env (AWS_ACCESS_KEY_ID), 
        # so api_key here can act as region_name or profile name if needed.
        super().__init__(api_key, model)

    @property
    def is_available(self) -> bool:
        # Override to just check if boto3 can be imported
        try:
            import boto3
            return True
        except ImportError:
            return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.is_available:
            raise ImportError("Please install boto3 (pip install boto3) to use AWS Bedrock.")
            
        import boto3
        client = boto3.client("bedrock-runtime", region_name=self.api_key or "us-east-1")
        
        # Structure varies slightly depending on which model family is used inside Bedrock.
        # This example uses the Converse API which standardizes it.
        response = client.converse(
            modelId=self.model,
            messages=[{"role": "user", "content": [{"text": prompt}]}]
        )
        
        return response["output"]["message"]["content"][0]["text"]
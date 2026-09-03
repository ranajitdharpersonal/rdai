from __future__ import annotations

from typing import Any, Optional

from rdai.providers.base import BaseProvider


class AwsBedrockProvider(BaseProvider):
    """AWS Bedrock provider using the Converse API."""

    traits = (
        "aws",
        "enterprise",
        "stable",
    )

    default_region = "us-east-1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # ``api_key`` is retained for backward compatibility with rdai's
        # provider interface. For Bedrock it acts as the configured AWS
        # region, while credentials are resolved by boto3's normal chain.
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def fallback_models(self) -> tuple[str, ...]:
        """Return the default Bedrock model identifier."""
        return ("anthropic.claude-3-haiku-20240307-v1:0",)

    @property
    def region(self) -> str:
        """Return the configured AWS region."""
        return self.api_key or self.default_region

    @property
    def is_available(self) -> bool:
        """Return True when boto3 can resolve AWS credentials."""

        try:
            import boto3
        except ImportError:
            return False

        try:
            session = boto3.Session()
            return session.get_credentials() is not None
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through AWS Bedrock Converse."""

        if not self.is_available:
            raise ValueError(
                "AWS Bedrock credentials are missing or boto3 is not installed."
            )

        if not self.model:
            raise RuntimeError(
                "No AWS Bedrock model is configured or available."
            )

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Please install boto3 using: pip install boto3"
            ) from exc

        client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
        )

        converse_kwargs: dict[str, Any] = {
            "modelId": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt,
                        }
                    ],
                }
            ],
        }

        # Forward supported Converse options without forcing callers
        # to use a provider-specific client directly.
        for option in (
            "system",
            "inferenceConfig",
            "toolConfig",
            "guardrailConfig",
            "performanceConfig",
        ):
            value = kwargs.pop(option, None)

            if value is not None:
                converse_kwargs[option] = value

        response = client.converse(**converse_kwargs)

        try:
            content_blocks = response["output"]["message"]["content"]
            content = content_blocks[0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "AWS Bedrock returned an unexpected response format."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "AWS Bedrock returned an empty response."
            )

        return content
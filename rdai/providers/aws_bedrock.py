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
        # ``api_key`` is retained for compatibility with the provider
        # interface. For Bedrock it represents the configured AWS region.
        super().__init__(
            api_key=api_key,
            model=model,
        )

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
            session = boto3.Session(
                region_name=self.region,
            )
            return session.get_credentials() is not None
        except Exception:
            return False

    def available_models(self) -> tuple[str, ...]:
        """Discover active text-output foundation models."""

        if not self.is_available:
            return ()

        try:
            import boto3
        except ImportError:
            return ()

        try:
            client = boto3.client(
                "bedrock",
                region_name=self.region,
            )

            response = client.list_foundation_models(
                by_output_modality="TEXT",
                by_inference_type="ON_DEMAND",
            )

            summaries = response.get("modelSummaries", [])

            if not isinstance(summaries, list):
                return ()

            models: list[str] = []

            for item in summaries:
                if not isinstance(item, dict):
                    continue

                model_id = item.get("modelId")

                if not isinstance(model_id, str):
                    continue

                model_id = model_id.strip()

                if not model_id:
                    continue

                lifecycle = item.get("modelLifecycle", {})

                if isinstance(lifecycle, dict):
                    status = lifecycle.get("status")

                    if status and status != "ACTIVE":
                        continue

                input_modalities = item.get(
                    "inputModalities",
                    [],
                )

                if isinstance(input_modalities, list):
                    normalized_inputs = {
                        str(value).upper()
                        for value in input_modalities
                    }

                    if "TEXT" not in normalized_inputs:
                        continue

                models.append(model_id)

            return tuple(models)

        except Exception:
            return ()

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through AWS Bedrock Converse."""

        if not self.is_available:
            raise ValueError(
                "AWS Bedrock credentials are missing or boto3 "
                "is not installed."
            )

        model = self.ensure_model()

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
            "modelId": model,
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

        response = client.converse(
            **converse_kwargs,
        )

        try:
            content_blocks = response["output"]["message"]["content"]
            content = content_blocks[0]["text"]
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "AWS Bedrock returned an unexpected response format."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "AWS Bedrock returned an empty response."
            )

        return content
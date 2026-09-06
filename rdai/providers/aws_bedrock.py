from __future__ import annotations

from collections.abc import Iterator
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

    def _client(self) -> Any:
        """Create an authenticated Bedrock runtime client."""

        if not self.is_available:
            raise ValueError(
                "AWS Bedrock credentials are missing or boto3 "
                "is not installed."
            )

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "Please install boto3 using: pip install boto3"
            ) from exc

        return boto3.client(
            "bedrock-runtime",
            region_name=self.region,
        )

    def _build_request(
        self,
        model: str,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build shared Converse request parameters."""

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
            value = kwargs.pop(
                option,
                None,
            )

            if value is not None:
                converse_kwargs[option] = value

        if kwargs:
            converse_kwargs.update(
                kwargs
            )

        return converse_kwargs

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

            summaries = response.get(
                "modelSummaries",
                [],
            )

            if not isinstance(
                summaries,
                list,
            ):
                return ()

            models: list[str] = []

            for item in summaries:
                if not isinstance(item, dict):
                    continue

                model_id = item.get(
                    "modelId"
                )

                if not (
                    isinstance(model_id, str)
                    and model_id.strip()
                ):
                    continue

                lifecycle = item.get(
                    "modelLifecycle",
                    {},
                )

                if isinstance(
                    lifecycle,
                    dict,
                ):
                    status = lifecycle.get(
                        "status"
                    )

                    if status and status != "ACTIVE":
                        continue

                input_modalities = item.get(
                    "inputModalities",
                    [],
                )

                if isinstance(
                    input_modalities,
                    list,
                ):
                    normalized_inputs = {
                        str(value).upper()
                        for value in input_modalities
                    }

                    if (
                        normalized_inputs
                        and "TEXT" not in normalized_inputs
                    ):
                        continue

                models.append(
                    model_id.strip()
                )

            return tuple(models)

        except Exception:
            return ()

    def filter_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Keep text-capable Bedrock model identifiers."""

        filtered: list[str] = []

        excluded_markers = (
            "embed",
            "rerank",
            "moderation",
            "speech",
            "audio",
            "image",
        )

        for model in models:
            normalized = model.strip()

            if not normalized:
                continue

            lowered = normalized.lower()

            if any(
                marker in lowered
                for marker in excluded_markers
            ):
                continue

            filtered.append(normalized)

        return tuple(filtered)

    def rank_models(
        self,
        models: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Preserve Bedrock catalog order."""

        return tuple(models)

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a complete response through Bedrock Converse."""

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        converse_kwargs = self._build_request(
            model,
            prompt,
            request_kwargs,
        )

        response = self._client().converse(
            **converse_kwargs
        )

        try:
            content_blocks = response["output"]["message"]["content"]

            text_parts = [
                block.get("text", "")
                for block in content_blocks
                if isinstance(block, dict)
                and block.get("text")
            ]

            content = "".join(text_parts)

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "AWS Bedrock returned an unexpected response format."
            ) from exc

        if not content:
            raise RuntimeError(
                "AWS Bedrock returned an empty response."
            )

        return content

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text deltas through Bedrock ConverseStream."""

        model = self.ensure_model()

        request_kwargs = dict(kwargs)

        converse_kwargs = self._build_request(
            model,
            prompt,
            request_kwargs,
        )

        response = self._client().converse_stream(
            **converse_kwargs
        )

        stream = response.get(
            "stream",
            (),
        )

        for event in stream:
            if not isinstance(event, dict):
                continue

            delta_event = event.get(
                "contentBlockDelta"
            )

            if not isinstance(
                delta_event,
                dict,
            ):
                continue

            delta = delta_event.get(
                "delta",
                {},
            )

            if not isinstance(
                delta,
                dict,
            ):
                continue

            text = delta.get(
                "text"
            )

            if text:
                yield str(text)
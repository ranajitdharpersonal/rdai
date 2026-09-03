from __future__ import annotations

from typing import Any, Optional

import requests

from rdai.providers.base import BaseProvider


class HuggingfaceProvider(BaseProvider):
    """Hugging Face provider adapter."""

    traits = (
        "open-source",
        "flexible",
        "community",
    )

    base_endpoint = "https://api-inference.huggingface.co/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
        )

    def fallback_models(self) -> tuple[str, ...]:
        """Return the default Hugging Face model."""
        return ("mistralai/Mistral-7B-Instruct-v0.2",)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response through Hugging Face inference."""

        if not self.is_available:
            raise ValueError("HuggingFace API key is missing.")

        if not self.model:
            raise RuntimeError(
                "No HuggingFace model is configured or available."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        parameters = {
            "max_new_tokens": 512,
        }

        # Allow callers to override inference parameters without replacing
        # the complete request structure.
        supplied_parameters = kwargs.pop("parameters", None)

        if isinstance(supplied_parameters, dict):
            parameters.update(supplied_parameters)

        timeout = kwargs.pop("timeout", 15.0)

        payload = {
            "inputs": formatted_prompt,
            "parameters": parameters,
        }

        # Preserve any additional top-level inference options.
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_endpoint}/{self.model}",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "HuggingFace returned an invalid JSON response."
            ) from exc

        if not isinstance(result, list) or not result:
            raise RuntimeError(
                "HuggingFace returned an unexpected response format."
            )

        first_result = result[0]

        if not isinstance(first_result, dict):
            raise RuntimeError(
                "HuggingFace returned an unexpected response format."
            )

        generated_text = first_result.get("generated_text")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise RuntimeError(
                "HuggingFace returned an empty response."
            )

        # Some text-generation models include the original prompt in the
        # generated text. Remove it only when it appears as a prefix.
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):]

        return generated_text.strip()
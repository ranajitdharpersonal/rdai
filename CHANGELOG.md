# Changelog

All notable changes to the `rdai` project will be documented in this file.

## [1.1.0] - 2026-09-06

### Added
- Added live provider model discovery with lazy resolution and caching.
- Added provider-specific model filtering for generation-capable models.
- Added model refresh and failed-model exclusion after discovered-model failures.
- Added explicit model authority so user-selected models are never silently replaced.
- Added resilient model-error recovery before provider failover.
- Added native streaming support across built-in providers where supported.
- Added provider-agnostic streaming through `AI.stream()`.
- Added streaming-aware failover with safe handling of failures before partial output.
- Added CLI streaming support with `rdai generate "<prompt>" --stream`.
- Added backward-compatible default streaming behavior for custom providers.
- Added broader model-access and terms-acceptance error handling.
- Added expanded failover and streaming test coverage.

### Changed
- Model selection no longer relies on hardcoded provider model fallbacks.
- Provider discovery is now the source of truth when no explicit model is supplied.
- Improved routing and failover behavior for unavailable or inaccessible models.
- Updated package metadata and runtime version to `1.1.0`.
- Improved provider adapters to use live provider catalogs where available.
- Preserved provider discovery order instead of promoting hardcoded model preferences.

### Fixed
- Fixed stale model selection causing generation failures when providers retire or restrict models.
- Fixed repeated retries against a failed discovered model.
- Fixed model refresh incorrectly replacing explicitly requested models.
- Fixed provider failover for model availability and access-related errors.
- Fixed missing streaming support through the public SDK and CLI.
- Fixed packaging metadata inconsistencies between runtime and distribution configuration.

## [1.0.2] - 2026-08-09

### Fixed
- Merged duplicate Router and Failover architectures into a single unified `engine.py`.
- Added missing `requests` dependency to resolve installation crashes.
- Improved `rdai doctor` to catch and classify real HTTP network errors instead of generic auth errors.
- Added explicit 15-second timeouts to all direct REST API providers (Claude, DeepSeek, Qwen, Llama, Mistral, HuggingFace) to prevent system hangs.

## [1.0.1] - 2026-08-01

### Changed
- Removed CLI loading animation from `main.py` for lightning-fast dashboard rendering.
- Expanded highly optimized SEO keywords in `pyproject.toml`.
- Updated PyPI project URLs to point to the correct GitHub repository (`ranajitdharpersonal`).

## [1.0.0] - 2026-08-01

### Added
- Initial release of the Multi-Brain AI Orchestrator.
- Unbreakable Auto-Failover logic implementation.
- Support for Gemini, OpenAI, Claude, Groq, and custom models.
- Interactive setup wizard using `rdai init`.
- Live API network diagnostic tool via `rdai doctor`.
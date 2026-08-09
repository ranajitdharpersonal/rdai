# Changelog

All notable changes to the `rdai` project will be documented in this file.


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

## [1.0.2] - 2026-08-09
### Fixed
- Merged duplicate Router and Failover architectures into a single unified `engine.py`.
- Added missing `requests` dependency to resolve installation crashes.
- Improved `rdai doctor` to catch and classify real HTTP network errors instead of generic auth errors.
- Added explicit 15-second timeouts to all direct REST API providers (Claude, DeepSeek, Qwen, Llama, Mistral, HuggingFace) to prevent system hangs.
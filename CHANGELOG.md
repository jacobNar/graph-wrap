# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5]
### Added
- Implemented `GuardrailsMiddleware` subclass of LangChain's `AgentMiddleware` to run safety, prompt injection, and hallucination checks within the standard middleware hook system (`before_model`/`after_model` and their async variants).

### Changed
- Refactored `wrap_node_with_guardrails` to execute guardrails through a sequential middleware chain (utilizing standard LangChain `PIIMiddleware` and `GuardrailsMiddleware`).

### Removed
- Removed manual `redact_text` and `redact_state` helper methods from `GuardrailValidator`.

## [0.1.4]

### Added
- Grouped trace logs by graph invocation.
- Implemented custom HTML/CSS timeline visualization.
- Implemented interactive span inspector selector.

### Fixed
- Improved hallucination validator prompts to prevent LLMs from copying the structure of JSON agent responses which caused parsing errors.

## [0.1.3]
### Added
- Created complete pytest test suite with database and LLM mocks.
- Added OpenAI and Ollama safety/hallucination verification tests.
- Added selective node-level guardrail tests.

## [0.1.2]
### Changed
- More renaming updates.

## [0.1.1]
### Changed
- Naming updates.

## [0.1.0]
### Added
- Initial project version.

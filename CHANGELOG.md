# Changelog

All notable changes to this project will be documented in this file.

## [0.1.6]
### Added
- Added baseline retry logic (3 retries by default) for all graph nodes.
- Added default node-level error handling to update state with a fallback message and route execution to the END of the graph.
- Added configurable `fallback_message`, `default_timeout`, `hitl`, and `interrupt_on` parameters to `StateGraph`.
- Added `system_context` to `GuardrailConfig` to provide agent system prompts, capabilities, and tool lists to the outbound hallucination checks, preventing false positive violations.
- Parallelized asynchronous inbound safety and prompt injection checks using `asyncio.gather`.
- Parallelized asynchronous outbound safety and hallucination checks using `asyncio.gather`.

### Changed
- Refactored `current_config_var` context variable and `HITLMiddleware` to be imported at the top level in `guardrails.py` to eliminate inline imports.

### Fixed
- Targeted guardrails validation to newly added messages only, rather than evaluating the entire state history recursively, fixing validation loops.
- Fixed control flow redirection on guardrail safety violations by returning Command(goto=END) instead of a raw state dict.
- Updated test conftest LLM registry mocks to support top-level imported ChatOllama and ChatOpenAI symbols.
- Fixed guardrail middleware fallback state generator to return minimal updates instead of copying and returning the entire state dict.
- Refactored message history comparison in outbound validation to prevent downstream nodes from evaluating prior node outputs.
- Optimized inbound safety checks to run exactly once per graph run on the initial human message.
- Updated outbound hallucination prompts to avoid false positive classifications on conversational filler, greetings, helper phrases, and code.
- Updated `test_error_handling.py` test cases to assert `("tools",)` as the next node upon tool interrupts to match the `before_model` hook execution path.

### Removed
- Removed file logging of telemetry errors to `graph_abstract_errors.log` (now handled via PostgreSQL / standard logging exceptions).


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

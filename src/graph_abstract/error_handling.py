from typing import Any, Optional
from langgraph.errors import NodeError
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.graph import END
from graph_abstract.guardrails import GuardrailValidationError, make_fallback_response

def make_default_error_handler(fallback_message: str) -> Any:
    def default_error_handler(state: Any, error: NodeError, config: Optional[RunnableConfig] = None) -> Command:
        if isinstance(error.error, GuardrailValidationError):
            raise error.error

        updated_state = make_fallback_response(state, fallback_message)
        if isinstance(updated_state, dict):
            updated_state.pop("jump_to", None)
        return Command(update=updated_state, goto=END)

    return default_error_handler

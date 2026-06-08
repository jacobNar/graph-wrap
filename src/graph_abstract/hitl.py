from typing import Any, Optional
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware, hook_config

class HITLMiddleware(AgentMiddleware):
    def __init__(self, interrupt_on: Any) -> None:
        super().__init__()
        self.interrupt_on = interrupt_on

    def _should_interrupt(self, tool_name: str) -> bool:
        if self.interrupt_on is True or self.interrupt_on is None:
            return True
        if isinstance(self.interrupt_on, list):
            return tool_name in self.interrupt_on
        if isinstance(self.interrupt_on, dict):
            return bool(self.interrupt_on.get(tool_name))
        return False

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not isinstance(state, dict) or "messages" not in state:
            return None
            
        messages = state["messages"]
        if not messages:
            return None
            
        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return None

        pending_approvals = [
            tc for tc in last_msg.tool_calls
            if self._should_interrupt(tc["name"])
        ]
        
        if not pending_approvals:
            return None

        decision = interrupt({
            "type": "tool_approval",
            "tools": [tc["name"] for tc in pending_approvals]
        })

        is_approved = False
        if isinstance(decision, str) and decision.lower().strip() == "approve":
            is_approved = True
        elif isinstance(decision, dict) and decision.get("decision", "").lower().strip() == "approve":
            is_approved = True

        if not is_approved:
            rejection_messages = []
            for tc in pending_approvals:
                rejection_messages.append(ToolMessage(
                    content="Tool execution was rejected by the user.",
                    tool_call_id=tc["id"]
                ))
            return {
                "messages": rejection_messages,
                "jump_to": "supervisor"
            }

        return None

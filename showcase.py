import asyncio
from typing_extensions import TypedDict
from graph_wrap import StateGraph, GuardrailConfig, GuardrailProvider

class AgentState(TypedDict):
    messages: list[str]

def simple_node(state: AgentState):
    return {"messages": ["Hello, World!"]}

async def main():
    db = "postgresql://postgres:postgres@localhost:5432/my_database"
    
    guardrails = GuardrailConfig(
        inbound_provider=GuardrailProvider.OLLAMA,
        safety_model="llama3.1:8b",
        check_prompt_injection=True,
        outbound_provider=GuardrailProvider.OLLAMA,
        eval_model="llama3.1:8b",
        check_hallucination=True
    )
    
    graph = StateGraph(AgentState, db_uri=db, guardrails=guardrails)
    graph.add_node("agent", simple_node)
    graph.set_entry_point("agent")
    graph.set_finish_point("agent")
    
    compiled = graph.compile()
    result = await compiled.ainvoke(
        {"messages": []},
        config={"configurable": {"thread_id": "demo_thread"}}
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

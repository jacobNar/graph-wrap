import pytest
import asyncio
from typing import Dict, Any
from typing_extensions import TypedDict
from langchain_core.tools import tool
from graph_abstract import StateGraph

class AgentState(TypedDict):
    messages: list[str]

@tool(description="Calculates the square of a number")
def calculate_square(x: int) -> int:
    return x * x

def dummy_node(state: AgentState) -> Dict[str, Any]:
    calculate_square.invoke({"x": 5})
    return {"messages": ["node_visited"]}

@pytest.mark.asyncio
async def test_async_graph_execution():
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    graph = StateGraph(AgentState, db_uri=db_uri)
    graph.add_node("dummy", dummy_node)
    graph.set_entry_point("dummy")
    graph.set_finish_point("dummy")
    compiled = graph.compile()
    config = {"configurable": {"thread_id": "test_thread_123"}}
    state = {"messages": ["hello"]}
    result = await compiled.ainvoke(state, config=config)
    assert "node_visited" in result["messages"]

def test_sync_graph_execution():
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    graph = StateGraph(AgentState, db_uri=db_uri)
    graph.add_node("dummy", dummy_node)
    graph.set_entry_point("dummy")
    graph.set_finish_point("dummy")
    compiled = graph.compile()
    config = {"configurable": {"thread_id": "test_thread_123"}}
    state = {"messages": ["hello"]}
    result = compiled.invoke(state, config=config)
    assert "node_visited" in result["messages"]

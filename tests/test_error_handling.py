import pytest
from unittest.mock import patch, MagicMock
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from graph_abstract import StateGraph
from langgraph.graph import START, END
from langgraph.errors import NodeError
from langchain_core.messages import AIMessage

class RetriableException(Exception):
    pass

class DummyState(TypedDict):
    messages: Annotated[list, add_messages]

attempts = 0

def failing_node(state: DummyState) -> dict:
    global attempts
    attempts += 1
    raise RetriableException("forced error")

def test_retries_and_error_handling():
    global attempts
    attempts = 0
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    graph = StateGraph(
        DummyState,
        db_uri=db_uri,
        fallback_message="something went wrong"
    )
    graph.add_node("fail", failing_node)
    graph.add_edge(START, "fail")
    compiled = graph.compile()

    config = {"configurable": {"thread_id": "test_thread"}}
    state = {"messages": []}

    result = compiled.invoke(state, config=config)

    assert attempts == 3
    assert len(result["messages"]) == 1
    assert "something went wrong" in str(result["messages"][0])

def test_hitl_approval_flow_sync():
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.types import Command
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    graph = StateGraph(
        DummyState,
        db_uri=db_uri,
        hitl=True,
        interrupt_on=["test_tool"]
    )
    def model_node(state: DummyState) -> dict:
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}])]}
    def tool_node(state: DummyState) -> dict:
        return {"messages": [ToolMessage(content="tool output", tool_call_id="call_1")]}
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("model")
    graph.add_edge("model", "tools")
    graph.set_finish_point("tools")
    compiled = graph.compile()
    config = {"configurable": {"thread_id": "sync_hitl_thread"}}
    state = {"messages": []}
    
    compiled.invoke(state, config=config)
    assert compiled.get_state(config).next == ("tools",)
    
    res = compiled.invoke(Command(resume="approve"), config=config)
    assert not compiled.get_state(config).next
    assert len(res["messages"]) == 2
    assert isinstance(res["messages"][-1], ToolMessage)
    assert res["messages"][-1].content == "tool output"

@pytest.mark.asyncio
async def test_hitl_approval_flow_async():
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.types import Command
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    graph = StateGraph(
        DummyState,
        db_uri=db_uri,
        hitl=True,
        interrupt_on=["test_tool"]
    )
    async def model_node(state: DummyState) -> dict:
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}])]}
    async def tool_node(state: DummyState) -> dict:
        return {"messages": [ToolMessage(content="tool output", tool_call_id="call_1")]}
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("model")
    graph.add_edge("model", "tools")
    graph.set_finish_point("tools")
    compiled = graph.compile()
    config = {"configurable": {"thread_id": "async_hitl_thread"}}
    state = {"messages": []}
    
    await compiled.ainvoke(state, config=config)
    state_after = await compiled.aget_state(config)
    assert state_after.next == ("tools",)
    
    res = await compiled.ainvoke(Command(resume="approve"), config=config)
    state_final = await compiled.aget_state(config)
    assert not state_final.next
    assert len(res["messages"]) == 2
    assert isinstance(res["messages"][-1], ToolMessage)
    assert res["messages"][-1].content == "tool output"

def test_hitl_rejection_flow_sync():
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.types import Command
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    graph = StateGraph(
        DummyState,
        db_uri=db_uri,
        hitl=True,
        interrupt_on=["test_tool"]
    )
    def model_node(state: DummyState) -> dict:
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "call_1"}])]}
    def tool_node(state: DummyState) -> dict:
        return {"messages": [ToolMessage(content="tool output", tool_call_id="call_1")]}
    def supervisor(state: DummyState) -> dict:
        return {"messages": ["supervisor response"]}
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)
    graph.add_node("supervisor", supervisor)
    graph.set_entry_point("model")
    graph.add_edge("model", "tools")
    graph.set_finish_point("tools")
    compiled = graph.compile()
    config = {"configurable": {"thread_id": "sync_reject_thread"}}
    state = {"messages": []}
    
    compiled.invoke(state, config=config)
    assert compiled.get_state(config).next == ("tools",)
    
    res = compiled.invoke(Command(resume="reject"), config=config)
    assert not compiled.get_state(config).next
    assert len(res["messages"]) == 3
    assert any(getattr(msg, "content", "") == "Tool execution was rejected by the user." for msg in res["messages"])
    assert any(getattr(msg, "content", "") == "supervisor response" for msg in res["messages"])

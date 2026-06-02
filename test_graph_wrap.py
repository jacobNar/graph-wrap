import asyncio
import sys
from typing import Dict, Any
from typing_extensions import TypedDict
from langchain_core.tools import tool
from graph_wrap import StateGraph

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AgentState(TypedDict):
    messages: list[str]

@tool(description="Calculates the square of a number")
def calculate_square(x: int) -> int:
    return x * x

def dummy_node(state: AgentState) -> Dict[str, Any]:
    calculate_square.invoke({"x": 5})
    return {"messages": ["node_visited"]}

async def main() -> None:
    db_uri = "postgresql://postgres:postgres@127.0.0.1:5433/testdb"
    
    graph = StateGraph(AgentState, db_uri=db_uri)
    graph.add_node("dummy", dummy_node)
    graph.set_entry_point("dummy")
    graph.set_finish_point("dummy")
    
    compiled = graph.compile()
    
    config = {"configurable": {"thread_id": "test_thread_123"}}
    state = {"messages": ["hello"]}
    
    result = await compiled.ainvoke(state, config=config)
    print("Execution Result:", result)
    
    import psycopg
    with psycopg.connect(db_uri) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT thread_id, event_name, payload FROM agent_logs;")
            rows = cur.fetchall()
            print("Logged Events:")
            for row in rows:
                print(f"Thread: {row[0]}, Event: {row[1]}, Payload Keys: {list(row[2].keys()) if row[2] else None}")

if __name__ == "__main__":
    asyncio.run(main())

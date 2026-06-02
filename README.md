# graph-wrap

A local-first Python library that provides zero-config PostgreSQL checkpointing and telemetry logging for LangGraph agents simply by replacing your standard `StateGraph` import.

---

## Installation

### Core Observability and Checkpointing
To install the core package with database checkpointing and telemetry logging:
```bash
pip install graph-wrap
```

### With Optional UI Console
To install the package along with the Streamlit-based visualization console:
```bash
pip install "graph-wrap[ui]"
```

---

## Quick Start

Replace your standard LangGraph `StateGraph` import with `graph_wrap`:

```python
from typing import Dict, Any
from typing_extensions import TypedDict
from graph_wrap import StateGraph

class AgentState(TypedDict):
    messages: list[str]

def my_node(state: AgentState) -> Dict[str, Any]:
    return {"messages": ["hello"]}

db_uri = "postgresql://postgres:postgres@localhost:5432/my_database"
graph = StateGraph(AgentState, db_uri=db_uri)
graph.add_node("my_node", my_node)
graph.set_entry_point("my_node")
graph.set_finish_point("my_node")

compiled = graph.compile()
```

When you call `compiled.ainvoke`, the library dynamically setups Postgres checkpoint tables and logs execution traces (chains, tools, and LLMs) into the database under the configured `thread_id`.

---

## Observability UI Console

If you installed `graph-wrap` with the `ui` extra, you can launch the console to inspect threads, visualize trace events, and view state checkpoints.

Run the console via python:
```bash
python -m graph_wrap --db-uri postgresql://postgres:postgres@localhost:5432/my_database
```

Or run the command line executable:
```bash
graph-wrap-ui --db-uri postgresql://postgres:postgres@localhost:5432/my_database
```

# graph-abstract

A local-first Python library that provides zero-config PostgreSQL checkpointing and telemetry logging for LangGraph agents simply by replacing your standard `StateGraph` import.

---

## Installation

### Core Observability and Checkpointing
To install the core package with database checkpointing and telemetry logging:
```bash
pip install graph-abstract
```

### With Optional UI Console
To install the package along with the Streamlit-based visualization console:
```bash
pip install "graph-abstract[ui]"
```

---

## Quick Start

Replace your standard LangGraph `StateGraph` import with `graph_abstract`:

```python
from typing import Dict, Any
from typing_extensions import TypedDict
from graph_abstract import StateGraph

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

If you installed `graph-abstract` with the `ui` extra, you can launch the console to inspect threads, visualize trace events, and view state checkpoints.

Run the console via python:
```bash
python -m graph_abstract --db-uri postgresql://postgres:postgres@localhost:5432/my_database
```

Or run the command line executable:
```bash
graph-abstract-ui --db-uri postgresql://postgres:postgres@localhost:5432/my_database
```

---

## Custom Guardrails

`graph-abstract` provides custom inbound and outbound safety guardrails. Evaluations for prompt injection, content safety, and hallucination are performed using LangChain's native structured output bindings for Ollama and OpenAI. Local regex-based PII redaction runs locally with zero LLM costs.

### Quick Example

```python
from typing_extensions import TypedDict
from graph_abstract import StateGraph, GuardrailConfig, GuardrailProvider

class AgentState(TypedDict):
    messages: list[str]
    context: str

safety_config = GuardrailConfig(
    inbound_provider=GuardrailProvider.OLLAMA,
    safety_model="llama3.1:8b",
    check_prompt_injection=True,
    
    outbound_provider=GuardrailProvider.OLLAMA,
    eval_model="llama3.1:8b",
    check_hallucination=True,
    
    redact_pii=True,
    fallback_message="I cannot answer that, please try to rephrase your question."
)

workflow = StateGraph(
    AgentState, 
    db_uri="postgresql://postgres:postgres@localhost:5432/my_database",
    guardrails=safety_config
)
```

### Node-Specific Guardrails

To apply guardrails to specific nodes only, pass a dictionary mapping node names to `GuardrailConfig` objects:

```python
workflow = StateGraph(
    AgentState,
    db_uri="postgresql://postgres:postgres@localhost:5432/my_database",
    guardrails={
        "Trader": GuardrailConfig(
            outbound_provider=GuardrailProvider.OLLAMA,
            eval_model="llama3.1:8b",
            check_hallucination=True
        )
    }
)
```

### Guardrail Configuration Options

When configuring `GuardrailConfig`, the following options are available:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inbound_provider` | `GuardrailProvider` | `None` | Provider for inbound validation (`OLLAMA` or `OPENAI`). |
| `safety_model` | `str` | `None` | Model name to use for inbound safety/prompt injection checks. |
| `check_prompt_injection` | `bool` | `False` | Enable inbound prompt injection and jailbreak detection. |
| `outbound_provider` | `GuardrailProvider` | `None` | Provider for outbound validation (`OLLAMA` or `OPENAI`). |
| `eval_model` | `str` | `None` | Model name to use for outbound hallucination or safety checks. |
| `check_hallucination` | `bool` | `False` | Enable outbound hallucination check comparing node outputs to graph context. |
| `redact_pii` | `bool` | `False` | Enable local regex-based redaction of emails, credit cards, SSNs, and phone numbers. |
| `fallback_message` | `Optional[str]` | `"I cannot answer..."` | The message returned when a guardrail is tripped. Set to `None` to raise a `GuardrailValidationError` instead. |

### StateGraph Configuration Options

When instantiating `StateGraph`, the following parameters are available:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fallback_message` | `str` | `"An unexpected error occurred..."` | Fallback message returned to state and routed to `END` on roadblock errors. |
| `default_timeout` | `Optional[Any]` | `None` | Default execution timeout for async nodes. |
| `hitl` | `bool` | `False` | Enable human-in-the-loop tool execution approvals. |
| `interrupt_on` | `Any` | `None` | Specific tools that require approval (list, dictionary, or `True`/`None` for all). |

---

## Fault Tolerance & Node Controls

By default, every node registered in `StateGraph` is wrapped with a retry policy of `max_attempts=3`. 
If all retries are exhausted (hitting a roadblock error), the state is updated with the `fallback_message`, and execution gracefully transitions to `END`.

---

## Human in the Loop (HITL)

When `hitl=True` is enabled, tool execution will be automatically paused using LangGraph's native `interrupt()` primitive.

To resume execution with user approval, call the graph's invoke or ainvoke with `Command(resume="approve")`. To reject tool execution, resume with `Command(resume="reject")` which will cancel the tool calls and route execution back to the supervisor node.


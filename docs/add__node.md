# add_node

> **Method** in `langgraph`

📖 [View in docs](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_node)

Add a new node to the `StateGraph`.

## Signature

```python
add_node(
    self,
    node: str | StateNode[NodeInputT, ContextT],
    action: StateNode[NodeInputT, ContextT] | None = None,
    *,
    defer: bool = False,
    metadata: dict[str, Any] | None = None,
    input_schema: type[NodeInputT] | None = None,
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    error_handler: StateNode[Any, ContextT] | None = None,
    destinations: dict[str, str] | tuple[str, ...] | None = None,
    timeout: float | timedelta | TimeoutPolicy | None = None,
    **kwargs: Unpack[DeprecatedKwargs] = {},
) -> Self
```

## Description

**Example:**

```python
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph

class State(TypedDict):
    x: int

def my_node(state: State, config: RunnableConfig) -> State:
    return {"x": state["x"] + 1}

builder = StateGraph(State)
builder.add_node(my_node)  # node name will be 'my_node'
builder.add_edge(START, "my_node")
graph = builder.compile()
graph.invoke({"x": 1})
# {'x': 2}
```

**Customize the name::**

```python
builder = StateGraph(State)
builder.add_node("my_fair_node", my_node)
builder.add_edge(START, "my_fair_node")
graph = builder.compile()
graph.invoke({"x": 1})
# {'x': 2}
```

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `node` | `str \| StateNode[NodeInputT, ContextT]` | Yes | The function or runnable this node will run.  If a string is provided, it will be used as the node name, and action will be used as the function or runnable. |
| `action` | `StateNode[NodeInputT, ContextT] \| None` | No | The action associated with the node.  Will be used as the node function or runnable if `node` is a string (node name). (default: `None`) |
| `defer` | `bool` | No | Whether to defer the execution of the node until the run is about to end. (default: `False`) |
| `metadata` | `dict[str, Any] \| None` | No | The metadata associated with the node. (default: `None`) |
| `input_schema` | `type[NodeInputT] \| None` | No | The input schema for the node. (Default: the graph's state schema) (default: `None`) |
| `retry_policy` | `RetryPolicy \| Sequence[RetryPolicy] \| None` | No | The retry policy for the node.  If a sequence is provided, the first matching policy will be applied. (default: `None`) |
| `cache_policy` | `CachePolicy \| None` | No | The cache policy for the node. (default: `None`) |
| `error_handler` | `StateNode[Any, ContextT] \| None` | No | Optional node-level error handler callable for this node. (default: `None`) |
| `destinations` | `dict[str, str] \| tuple[str, ...] \| None` | No | Destinations that indicate where a node can route to.  Useful for edgeless graphs with nodes that return `Command` objects.  If a `dict` is provided, the keys will be used as the target node names and the values will be used as the labels for the edges.  If a `tuple` is provided, the values will be used as the target node names.  !!! warning      This is only used for graph rendering and doesn't have any effect on the graph execution. (default: `None`) |
| `timeout` | `float \| timedelta \| TimeoutPolicy \| None` | No | Timeout for each node attempt. A number or `timedelta` is a hard wall-clock cap and is not refreshed. Use `TimeoutPolicy` to configure both a wall-clock `run_timeout` and an `idle_timeout` refreshed by progress signals. When exceeded, a [`NodeTimeoutError`][langgraph.errors.NodeTimeoutError] is raised and the retry policy (if any) decides whether to retry. Timeouts are supported only for async nodes; sync nodes cannot be safely cancelled in-process. (default: `None`) |

## Returns

`Self`

The instance of the `StateGraph`, allowing for method chaining.

---

[View source on GitHub](https://github.com/langchain-ai/langgraph/blob/83dd61feaca993d2ee428706ad04c869895ce400/libs/langgraph/langgraph/graph/state.py#L662)
# astream_log

> **Method** in `langchain_core`

📖 [View in docs](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/astream_log)

Stream all output from a `Runnable`, as reported to the callback system.

This includes all inner runs of LLMs, Retrievers, Tools, etc.

Output is streamed as Log objects, which include a list of
Jsonpatch ops that describe how the state of the run has changed in each
step, and the final state of the run.

The Jsonpatch ops can be applied in order to construct state.

## Signature

```python
astream_log(
    self,
    input: Any,
    config: RunnableConfig | None = None,
    *,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any = {},
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]
```

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `input` | `Any` | Yes | The input to the `Runnable`. |
| `config` | `RunnableConfig \| None` | No | The config to use for the `Runnable`. (default: `None`) |
| `diff` | `bool` | No | Whether to yield diffs between each step or the current state. (default: `True`) |
| `with_streamed_output_list` | `bool` | No | Whether to yield the `streamed_output` list. (default: `True`) |
| `include_names` | `Sequence[str] \| None` | No | Only include logs with these names. (default: `None`) |
| `include_types` | `Sequence[str] \| None` | No | Only include logs with these types. (default: `None`) |
| `include_tags` | `Sequence[str] \| None` | No | Only include logs with these tags. (default: `None`) |
| `exclude_names` | `Sequence[str] \| None` | No | Exclude logs with these names. (default: `None`) |
| `exclude_types` | `Sequence[str] \| None` | No | Exclude logs with these types. (default: `None`) |
| `exclude_tags` | `Sequence[str] \| None` | No | Exclude logs with these tags. (default: `None`) |
| `**kwargs` | `Any` | No | Additional keyword arguments to pass to the `Runnable`. (default: `{}`) |

---

[View source on GitHub](https://github.com/langchain-ai/langchain/blob/dfca7f44246f50208fcfab914ca265a277cdc0ae/libs/core/langchain_core/runnables/base.py#L1207)
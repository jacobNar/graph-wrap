# stream_events

> **Method** in `langchain_core`

📖 [View in docs](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/stream_events)

Generate a stream of events synchronously.

Synchronous counterpart to `astream_events`. For `version='v3'`, subclasses
that implement the v3 streaming protocol (`BaseChatModel`, `CompiledGraph`)
override this method. All other versions and base-class calls raise
`NotImplementedError`.

## Signature

```python
stream_events(
    self,
    input: Any,
    config: RunnableConfig | None = None,
    *,
    version: Literal['v1', 'v2', 'v3'] = 'v2',
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any = {},
) -> Iterator[StreamEvent] | Iterator[Any]
```

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `input` | `Any` | Yes | The input to the `Runnable`. |
| `config` | `RunnableConfig \| None` | No | The config to use for the `Runnable`. (default: `None`) |
| `version` | `Literal['v1', 'v2', 'v3']` | No | The version of the schema to use. `'v3'` requires a subclass that implements the v3 streaming protocol. `'v1'` and `'v2'` are not supported on the sync path. (default: `'v2'`) |
| `include_names` | `Sequence[str] \| None` | No | Only include events from `Runnable` objects with matching names. (default: `None`) |
| `include_types` | `Sequence[str] \| None` | No | Only include events from `Runnable` objects with matching types. (default: `None`) |
| `include_tags` | `Sequence[str] \| None` | No | Only include events from `Runnable` objects with matching tags. (default: `None`) |
| `exclude_names` | `Sequence[str] \| None` | No | Exclude events from `Runnable` objects with matching names. (default: `None`) |
| `exclude_types` | `Sequence[str] \| None` | No | Exclude events from `Runnable` objects with matching types. (default: `None`) |
| `exclude_tags` | `Sequence[str] \| None` | No | Exclude events from `Runnable` objects with matching tags. (default: `None`) |
| `**kwargs` | `Any` | No | Additional keyword arguments to pass to the `Runnable`. (default: `{}`) |

---

[View source on GitHub](https://github.com/langchain-ai/langchain/blob/dfca7f44246f50208fcfab914ca265a277cdc0ae/libs/core/langchain_core/runnables/base.py#L1636)
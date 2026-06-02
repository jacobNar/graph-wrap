# AsyncPostgresSaver

> **Class** in `langgraph.checkpoint.postgres`

📖 [View in docs](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver)

Asynchronous checkpointer that stores checkpoints in a Postgres database.

## Signature

```python
AsyncPostgresSaver(
    self,
    conn: _ainternal.Conn,
    pipe: AsyncPipeline | None = None,
    serde: SerializerProtocol | None = None,
)
```

## Extends

- `BasePostgresSaver`

## Constructors

```python
__init__(
    self,
    conn: _ainternal.Conn,
    pipe: AsyncPipeline | None = None,
    serde: SerializerProtocol | None = None,
) -> None
```

| Name | Type |
|------|------|
| `conn` | `_ainternal.Conn` |
| `pipe` | `AsyncPipeline \| None` |
| `serde` | `SerializerProtocol \| None` |


## Properties

- `lock`
- `conn`
- `pipe`
- `loop`
- `supports_pipeline`

## Methods

- [`from_conn_string()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/from_conn_string)
- [`setup()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/setup)
- [`alist()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/alist)
- [`aget_tuple()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/aget_tuple)
- [`aput()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/aput)
- [`aput_writes()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/aput_writes)
- [`adelete_thread()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/adelete_thread)
- [`aget_delta_channel_history()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/aget_delta_channel_history)
- [`list()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/list)
- [`get_tuple()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/get_tuple)
- [`put()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/put)
- [`put_writes()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/put_writes)
- [`delete_thread()`](https://reference.langchain.com/python/langgraph.checkpoint.postgres/aio/AsyncPostgresSaver/delete_thread)

---

[View source on GitHub](https://github.com/langchain-ai/langgraph/blob/83dd61feaca993d2ee428706ad04c869895ce400/libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py#L40)
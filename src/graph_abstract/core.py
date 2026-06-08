import sys
import asyncio
import inspect
from typing import Any, AsyncIterator, Iterator, Optional
import psycopg
from langgraph.graph import StateGraph as BaseStateGraph

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import RetryPolicy
from graph_abstract.telemetry import PostgresTelemetryHandler, SyncPostgresTelemetryHandler
from graph_abstract.guardrails import wrap_node_with_guardrails, current_config_var
from graph_abstract.error_handling import make_default_error_handler

if sys.platform == "win32":
    try:
        if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

class WrappedCompiledGraph:
    def __init__(self, compiled_graph: Any, db_uri: str) -> None:
        self._compiled_graph = compiled_graph
        self.db_uri = db_uri

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiled_graph, name)

    async def ainvoke(
        self,
        input: Any,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        config = config or {}
        new_config = dict(config)
        configurable = dict(new_config.get("configurable", {}))
        thread_id = configurable.get("thread_id") or "default_thread"
        configurable["thread_id"] = thread_id
        new_config["configurable"] = configurable
        
        async with AsyncPostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            await checkpointer.setup()
            self._compiled_graph.checkpointer = checkpointer
            
            callbacks = list(new_config.get("callbacks", []))
            callbacks.append(PostgresTelemetryHandler(self.db_uri, thread_id))
            new_config["callbacks"] = callbacks
            
            token = current_config_var.set(new_config)
            try:
                return await self._compiled_graph.ainvoke(input, config=new_config, **kwargs)
            finally:
                current_config_var.reset(token)

    async def astream(
        self,
        input: Any,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        config = config or {}
        new_config = dict(config)
        configurable = dict(new_config.get("configurable", {}))
        thread_id = configurable.get("thread_id") or "default_thread"
        configurable["thread_id"] = thread_id
        new_config["configurable"] = configurable
        
        async with AsyncPostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            await checkpointer.setup()
            self._compiled_graph.checkpointer = checkpointer
            
            callbacks = list(new_config.get("callbacks", []))
            callbacks.append(PostgresTelemetryHandler(self.db_uri, thread_id))
            new_config["callbacks"] = callbacks
            
            token = current_config_var.set(new_config)
            try:
                async for chunk in self._compiled_graph.astream(input, config=new_config, **kwargs):
                    yield chunk
            finally:
                current_config_var.reset(token)

    def invoke(
        self,
        input: Any,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        config = config or {}
        new_config = dict(config)
        configurable = dict(new_config.get("configurable", {}))
        thread_id = configurable.get("thread_id") or "default_thread"
        configurable["thread_id"] = thread_id
        new_config["configurable"] = configurable
        
        with PostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            checkpointer.setup()
            self._compiled_graph.checkpointer = checkpointer
            
            callbacks = list(new_config.get("callbacks", []))
            callbacks.append(SyncPostgresTelemetryHandler(self.db_uri, thread_id))
            new_config["callbacks"] = callbacks
            
            token = current_config_var.set(new_config)
            try:
                return self._compiled_graph.invoke(input, config=new_config, **kwargs)
            finally:
                current_config_var.reset(token)

    def stream(
        self,
        input: Any,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        config = config or {}
        new_config = dict(config)
        configurable = dict(new_config.get("configurable", {}))
        thread_id = configurable.get("thread_id") or "default_thread"
        configurable["thread_id"] = thread_id
        new_config["configurable"] = configurable
        
        with PostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            checkpointer.setup()
            self._compiled_graph.checkpointer = checkpointer
            
            callbacks = list(new_config.get("callbacks", []))
            callbacks.append(SyncPostgresTelemetryHandler(self.db_uri, thread_id))
            new_config["callbacks"] = callbacks
            
            token = current_config_var.set(new_config)
            try:
                for chunk in self._compiled_graph.stream(input, config=new_config, **kwargs):
                    yield chunk
            finally:
                current_config_var.reset(token)

class StateGraph(BaseStateGraph):
    def __init__(
        self,
        state_schema: type,
        db_uri: str,
        guardrails: Any = None,
        context_schema: Optional[type] = None,
        *,
        input_schema: Optional[type] = None,
        output_schema: Optional[type] = None,
        fallback_message: str = "An unexpected error occurred. Please try again later.",
        default_timeout: Optional[Any] = None,
        hitl: bool = False,
        interrupt_on: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            state_schema,
            context_schema=context_schema,
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs,
        )
        self.db_uri = db_uri
        self.guardrails = guardrails
        self.fallback_message = fallback_message
        self.default_timeout = default_timeout
        self.hitl = hitl
        self.interrupt_on = interrupt_on
        self.default_error_handler = make_default_error_handler(fallback_message)

    def add_node(
        self,
        node: Any,
        action: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        orig_callable = action if isinstance(node, str) else node
        is_async = False
        if orig_callable is not None:
            if inspect.iscoroutinefunction(orig_callable) or (hasattr(orig_callable, "ainvoke") and not inspect.iscoroutinefunction(orig_callable)):
                is_async = True

        if "retry_policy" not in kwargs:
            kwargs["retry_policy"] = RetryPolicy(max_attempts=3)

        if "error_handler" not in kwargs:
            kwargs["error_handler"] = self.default_error_handler

        if is_async and "timeout" not in kwargs and self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout

        node_guardrails = self.guardrails
        if isinstance(self.guardrails, dict):
            node_name = node if isinstance(node, str) else getattr(node, "__name__", None)
            node_guardrails = self.guardrails.get(node_name)
        if isinstance(node, str):
            if action is not None:
                action = wrap_node_with_guardrails(action, node_guardrails, self.hitl, self.interrupt_on)
        else:
            node = wrap_node_with_guardrails(node, node_guardrails, self.hitl, self.interrupt_on)
        return super().add_node(node, action, **kwargs)

    def compile(self, *args: Any, **kwargs: Any) -> WrappedCompiledGraph:
        kwargs.pop("checkpointer", None)
        with psycopg.connect(self.db_uri, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_logs (
                        id SERIAL PRIMARY KEY,
                        thread_id VARCHAR(255) NOT NULL,
                        event_name VARCHAR(255) NOT NULL,
                        payload JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_logs_thread ON agent_logs(thread_id);
                    """
                )
        compiled = super().compile(*args, **kwargs)
        return WrappedCompiledGraph(compiled, self.db_uri)

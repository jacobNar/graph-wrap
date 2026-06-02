import sys
import asyncio
from typing import Any, Optional
import psycopg
from langgraph.graph import StateGraph as BaseStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from graph_wrap.telemetry import PostgresTelemetryHandler

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
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default_thread")
        
        async with AsyncPostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            await checkpointer.setup()
            self._compiled_graph.checkpointer = checkpointer
            
            new_config = dict(config)
            callbacks = list(new_config.get("callbacks", []))
            callbacks.append(PostgresTelemetryHandler(self.db_uri, thread_id))
            new_config["callbacks"] = callbacks
            
            return await self._compiled_graph.ainvoke(input, config=new_config, **kwargs)

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

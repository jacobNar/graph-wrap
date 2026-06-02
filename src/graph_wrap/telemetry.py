import datetime
import json
from typing import Any, Dict, List, Optional
from uuid import UUID
import psycopg
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler

class TelemetryJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, UUID):
            return str(o)
        try:
            return super().default(o)
        except TypeError:
            return str(o)

class PostgresTelemetryHandler(AsyncCallbackHandler):
    def __init__(self, db_uri: str, thread_id: str) -> None:
        self.db_uri = db_uri
        self.thread_id = thread_id

    async def _insert_log(self, event_name: str, payload: Dict[str, Any]) -> None:
        try:
            async with await psycopg.AsyncConnection.connect(self.db_uri, autocommit=True) as conn:
                async with conn.cursor() as cur:
                    serialized_payload = json.dumps(payload, cls=TelemetryJSONEncoder)
                    await cur.execute(
                        "INSERT INTO agent_logs (thread_id, event_name, payload) VALUES (%s, %s, %s::jsonb)",
                        (self.thread_id, event_name, serialized_payload)
                    )
        except Exception as e:
            with open("graph_wrap_errors.log", "a") as f:
                f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} - Telemetry Error: {str(e)}\n")

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "serialized": serialized,
            "inputs": inputs,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata
        }
        await self._insert_log("chain_start", payload)

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "serialized": serialized,
            "prompts": prompts,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata
        }
        await self._insert_log("llm_start", payload)

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "serialized": serialized,
            "input_str": input_str,
            "inputs": inputs,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata
        }
        await self._insert_log("tool_start", payload)

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "output": str(output),
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None
        }
        await self._insert_log("tool_end", payload)

class SyncPostgresTelemetryHandler(BaseCallbackHandler):
    def __init__(self, db_uri: str, thread_id: str) -> None:
        self.db_uri = db_uri
        self.thread_id = thread_id

    def _insert_log(self, event_name: str, payload: Dict[str, Any]) -> None:
        try:
            with psycopg.connect(self.db_uri, autocommit=True) as conn:
                with conn.cursor() as cur:
                    serialized_payload = json.dumps(payload, cls=TelemetryJSONEncoder)
                    cur.execute(
                        "INSERT INTO agent_logs (thread_id, event_name, payload) VALUES (%s, %s, %s::jsonb)",
                        (self.thread_id, event_name, serialized_payload)
                    )
        except Exception as e:
            with open("graph_wrap_errors.log", "a") as f:
                f.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} - Telemetry Error (Sync): {str(e)}\n")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "serialized": serialized,
            "inputs": inputs,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata
        }
        self._insert_log("chain_start", payload)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "serialized": serialized,
            "prompts": prompts,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata
        }
        self._insert_log("llm_start", payload)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "serialized": serialized,
            "input_str": input_str,
            "inputs": inputs,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": tags,
            "metadata": metadata
        }
        self._insert_log("tool_start", payload)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        payload = {
            "output": str(output),
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None
        }
        self._insert_log("tool_end", payload)

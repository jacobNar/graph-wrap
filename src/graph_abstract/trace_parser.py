import datetime
from typing import List, Dict, Any, Optional

class Span:
    def __init__(self, run_id: str, parent_run_id: Optional[str] = None) -> None:
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self.name = "Unknown"
        self.type = "unknown"
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None
        self.status = "success"
        self.error: Optional[str] = None
        self.payloads: Dict[str, Any] = {}
        self.children: List[Span] = []
        self.depth = 0

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000.0
        return 0.0

    @property
    def inputs(self) -> Any:
        for event in self.payloads:
            if event.endswith("_start"):
                p = self.payloads[event]
                if "inputs" in p:
                    return p["inputs"]
                if "messages" in p:
                    return p["messages"]
                if "prompts" in p:
                    return p["prompts"]
                if "input_str" in p:
                    return p["input_str"]
        return None

    @property
    def outputs(self) -> Any:
        for event in self.payloads:
            if event.endswith("_end"):
                p = self.payloads[event]
                if "outputs" in p:
                    return p["outputs"]
                if "generations" in p:
                    return p["generations"]
                if "output" in p:
                    return p["output"]
            elif event.endswith("_error"):
                p = self.payloads[event]
                if "error" in p:
                    return p["error"]
        return None

class Invocation:
    def __init__(self, root_span: Span) -> None:
        self.root_span = root_span
        self.spans: List[Span] = []
        self.start_time = root_span.start_time
        self.end_time = root_span.end_time
        self.status = root_span.status

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000.0
        return 0.0

class TraceParser:
    def parse_rows(self, rows: List[tuple]) -> List[Invocation]:
        spans_by_id: Dict[str, Span] = {}
        
        for event_name, payload, created_at in rows:
            run_id = payload.get("run_id")
            if not run_id:
                continue
                
            parent_run_id = payload.get("parent_run_id")
            
            if run_id not in spans_by_id:
                spans_by_id[run_id] = Span(run_id, parent_run_id)
                
            span = spans_by_id[run_id]
            span.payloads[event_name] = payload
            
            if event_name.endswith("_start"):
                span.start_time = created_at
                
                if event_name == "chain_start":
                    span.type = "chain"
                    metadata = payload.get("metadata") or {}
                    if "langgraph_node" in metadata:
                        span.name = f"Node: {metadata['langgraph_node']}"
                    else:
                        span.name = payload.get("serialized", {}).get("name") if payload.get("serialized") else "Chain"
                elif event_name in ("chat_model_start", "llm_start"):
                    span.type = "llm"
                    metadata = payload.get("metadata") or {}
                    span.name = f"LLM: {metadata.get('ls_model_name', 'LLM')}"
                elif event_name == "tool_start":
                    span.type = "tool"
                    serialized = payload.get("serialized") or {}
                    span.name = f"Tool: {serialized.get('name', 'Tool')}"
            
            elif event_name.endswith("_end"):
                span.end_time = created_at
                
            elif event_name.endswith("_error"):
                span.end_time = created_at
                span.status = "error"
                span.error = payload.get("error", "Unknown error")
        
        root_spans: List[Span] = []
        for span in spans_by_id.values():
            parent_id = span.parent_run_id
            if parent_id and parent_id in spans_by_id:
                spans_by_id[parent_id].children.append(span)
            else:
                root_spans.append(span)
                
        def process_span_tree(span: Span, depth: int) -> List[Span]:
            span.depth = depth
            span.children.sort(key=lambda s: s.start_time or datetime.datetime.min)
            
            flat_list = [span]
            for child in span.children:
                flat_list.extend(process_span_tree(child, depth + 1))
            return flat_list
            
        invocations: List[Invocation] = []
        for root in root_spans:
            all_spans_in_tree = process_span_tree(root, 0)
            
            if not root.start_time and all_spans_in_tree:
                root.start_time = min((s.start_time for s in all_spans_in_tree if s.start_time), default=None)
            if not root.end_time and all_spans_in_tree:
                root.end_time = max((s.end_time for s in all_spans_in_tree if s.end_time), default=None)
                
            invocation = Invocation(root)
            invocation.spans = all_spans_in_tree
            
            if any(s.status == "error" for s in all_spans_in_tree):
                invocation.status = "error"
                
            invocations.append(invocation)
            
        invocations.sort(key=lambda i: i.start_time or datetime.datetime.min, reverse=True)
        return invocations

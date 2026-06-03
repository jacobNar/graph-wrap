import pytest
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver
from graph_abstract.guardrails import SafetyCheckResult, HallucinationCheckResult

class MockCursor:
    def execute(self, *args, **kwargs):
        return self
    def fetchall(self):
        return []
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockConnection:
    def cursor(self):
        return MockCursor()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class MockPostgresSaver(MemorySaver):
    @classmethod
    def from_conn_string(cls, conn_string, **kwargs):
        return cls()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def setup(self):
        pass

class MockAsyncPostgresSaver(MemorySaver):
    @classmethod
    def from_conn_string(cls, conn_string, **kwargs):
        return cls()
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    async def setup(self):
        pass

class MockStructuredLLM:
    def __init__(self, result):
        self.result = result
    def invoke(self, *args, **kwargs):
        return self.result
    async def ainvoke(self, *args, **kwargs):
        return self.result

class MockLLMRegistry:
    def __init__(self):
        self.safety_result = SafetyCheckResult(unsafe=False, reason="safe")
        self.hallucination_result = HallucinationCheckResult(hallucination=False, reason="grounded")
        self.calls = []

class MockLLM:
    def __init__(self, registry):
        self.registry = registry
    def with_structured_output(self, schema, method=None):
        self.registry.calls.append(("with_structured_output", schema, method))
        if schema == SafetyCheckResult:
            return MockStructuredLLM(self.registry.safety_result)
        elif schema == HallucinationCheckResult:
            return MockStructuredLLM(self.registry.hallucination_result)
        return MockStructuredLLM(None)

@pytest.fixture(autouse=True)
def mock_db():
    async def mock_async_connect(*args, **kwargs):
        return MockConnection()
    with patch("psycopg.connect", return_value=MockConnection()), \
         patch("psycopg.AsyncConnection.connect", side_effect=mock_async_connect), \
         patch("graph_abstract.core.PostgresSaver", MockPostgresSaver), \
         patch("graph_abstract.core.AsyncPostgresSaver", MockAsyncPostgresSaver):
        yield

@pytest.fixture
def mock_llms():
    registry = MockLLMRegistry()
    def create_mock_llm(*args, **kwargs):
        return MockLLM(registry)
    with patch("langchain_ollama.ChatOllama", side_effect=create_mock_llm), \
         patch("langchain_openai.ChatOpenAI", side_effect=create_mock_llm):
        yield registry

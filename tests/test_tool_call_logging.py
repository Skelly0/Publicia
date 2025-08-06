import json
import importlib.util
from pathlib import Path


def load_logging_module():
    """Load the actual utils.logging module regardless of sys.modules hacks."""
    module_path = Path(__file__).resolve().parents[1] / "utils" / "logging.py"
    spec = importlib.util.spec_from_file_location("_real_logging", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_log_tool_call_trace(tmp_path, monkeypatch):
    logging_pkg = load_logging_module()
    log_file = tmp_path / "tool_calls.jsonl"
    monkeypatch.setattr(logging_pkg, "TOOL_CALL_LOG_FILE", str(log_file))
    logging_pkg.log_tool_call_trace(
        "What is Foo?",
        [{"name": "search_documents", "args": {"query": "Foo"}, "result": {"items": [1, 2]}}],
    )
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        entry = json.loads(f.readline())
    assert entry["question"] == "What is Foo?"
    assert entry["tool_calls"][0]["name"] == "search_documents"

from trajectory_eval import ToolCall, evaluate_tool_calls


def test_package_smoke() -> None:
    call = ToolCall("lookup", {"id": 1})
    assert evaluate_tool_calls((call,), (call,)).trajectory_success

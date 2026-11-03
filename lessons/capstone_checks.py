"""Progressive application assessment; not a sandbox for learner-written code."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import Agent, ToolPolicy, ToolRegistry

from .lab04_capstone import PRICES, STOCK, run


def _rejects(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


def _schemas(app):
    tools = app.create_tools(dict(STOCK), dict(PRICES))
    expected = {"lookup_stock": {"sku": {"type": "string"}},
                "quote_total": {"sku": {"type": "string"}, "quantity": {"type": "integer"}}}
    if set(tools) != set(expected):
        return False
    for name, properties in expected.items():
        entry = tools[name]
        schema = entry["schema"]
        if (entry.get("read_only") is not True or not callable(entry["fn"])
                or schema["name"] != name or not schema["description"]
                or schema["input_schema"] != {"type": "object", "properties": properties,
                    "required": list(properties), "additionalProperties": False}):
            return False
    return True


def _values(app):
    tools = app.create_tools({"CUSTOM": 11}, {"CUSTOM": 37})
    lookup, quote = tools["lookup_stock"]["fn"], tools["quote_total"]["fn"]
    return (json.loads(lookup("CUSTOM")) == {"sku": "CUSTOM", "quantity": 11}
            and json.loads(quote("CUSTOM", 3)) == {"sku": "CUSTOM", "quantity": 3, "total_cents": 111}
            and _rejects(lookup, "ABSENT") and _rejects(quote, "ABSENT", 2)
            and all(_rejects(quote, "CUSTOM", value) for value in (0, -1, True, "2", 2.5)))


def _snapshots(app):
    stock, prices = {"CUSTOM": 3}, {"CUSTOM": 19}
    first = app.create_tools(stock, prices)
    stock["CUSTOM"], prices["CUSTOM"] = 50, 900
    second = app.create_tools(stock, prices)
    return (first is not second
            and json.loads(first["lookup_stock"]["fn"]("CUSTOM"))["quantity"] == 3
            and json.loads(first["quote_total"]["fn"]("CUSTOM", 2))["total_cents"] == 38
            and json.loads(second["lookup_stock"]["fn"]("CUSTOM"))["quantity"] == 50
            and json.loads(second["quote_total"]["fn"]("CUSTOM", 2))["total_cents"] == 1800
            and all(_rejects(app.create_tools, counts, costs) for counts, costs in (
                ({"A": 1}, {"B": 1}), ({"A": -1}, {"A": 1}), ({"A": 1}, {"A": True}),
                ({"": 1}, {"": 1}), ({"A": 1.5}, {"A": 2}), ({"A": 1}, {"A": -2}),
                ({"X" * 101: 1}, {"X" * 101: 1}), ({1: 1}, {1: 1}),
                ({str(i): 1 for i in range(1001)}, {str(i): 1 for i in range(1001)}))))


def _controls(app):
    with TemporaryDirectory(prefix="little-harness-capstone-check-") as directory:
        agent = app.create_agent(lambda *a, **k: None, dict(STOCK), dict(PRICES),
                                 trace_path=Path(directory) / "trace.jsonl")
        return (isinstance(agent, Agent) and isinstance(agent.tool_policy, ToolPolicy)
                and agent.tool_policy.read_only is True
                and (agent.limits.max_model_calls, agent.limits.max_tool_calls) == (4, 3))


def _happy(app):
    result, messages, events = run(app.create_agent, sku="CUSTOM", quantity=3,
                                 stock={"CUSTOM": 11}, prices={"CUSTOM": 37})
    return ((result.status, result.answer) == ("completed", "CUSTOM: 3 units cost 111 cents; stock 11.")
            and (result.usage.model_calls, result.usage.tool_calls_attempted, result.usage.tool_errors) == (3, 2, 0)
            and [e.data["name"] for e in events if e.kind == "tool_started"] == ["lookup_stock", "quote_total"]
            and messages[-1]["content"][0]["text"] == result.answer)


def _failures(app):
    absent = run(app.create_agent, sku="ABSENT")[0]
    empty = run(app.create_agent, sku="GADGET")[0]
    invalid = run(app.create_agent, quantity=0)[0]
    return ((absent.status, absent.answer, absent.usage.tool_errors) == (
                "completed", "Unable to quote that order.", 1)
            and (empty.status, empty.answer, empty.usage.tool_errors) == (
                "completed", "GADGET: insufficient stock (0 available).", 0)
            and (invalid.status, invalid.answer, invalid.usage.tool_errors) == (
                "completed", "Unable to quote that order.", 1))


def _recovery(app):
    result, messages, events = run(app.create_agent, scenario="repair")
    results = [m["content"][0] for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    return ((result.status, result.answer) == ("completed", "WIDGET: 2 units cost 500 cents; stock 7.")
            and (result.usage.model_calls, result.usage.tool_calls_attempted, result.usage.tool_errors) == (4, 3, 1)
            and [r["is_error"] for r in results] == [True, False, False]
            and [e.data["status"] for e in events if e.kind == "tool_finished"] == ["not_run", "completed", "completed"])


def _budget(app):
    result, _, events = run(app.create_agent, scenario="repeat")
    return ((result.status, result.stop_reason, result.answer) == ("stopped", "max_tool_calls", "")
            and (result.usage.model_calls, result.usage.tool_calls_attempted) == (4, 3)
            and not any(e.kind == "tool_started" for e in events))


def _policy(app):
    executed, advertised = [], []
    extra = ToolRegistry()

    @extra.tool()
    def reserve_stock() -> str:
        """Forbidden test fixture: not declared read-only."""
        executed.append(True)
        return "reserved"

    replies = iter([
        {"content": [{"type": "tool_use", "id": "denied", "name": "reserve_stock", "input": {}}]},
        {"content": [{"type": "text", "text": "No order placed."}]}])

    def model(system, messages, tools, on_text=None):
        advertised.extend(schema["name"] for schema in tools)
        return next(replies)

    with TemporaryDirectory(prefix="little-harness-capstone-policy-") as directory:
        agent = app.create_agent(model, dict(STOCK), dict(PRICES), trace_path=Path(directory) / "trace.jsonl")
        agent.tools.update(extra)
        result = agent.run_result("Try an undeclared action.")
    return (not executed and "reserve_stock" not in advertised
            and result.status == "completed" and result.usage.tool_errors == 1)


def check_capstone(app):
    """Stop at the first unfinished contract; never print learner exception contents."""
    cases = [
        ("two read-only tools expose only sku and quantity", _schemas,
         "Use ToolRegistry decorators, primitive annotations and tool docstrings."),
        ("tools calculate from supplied data and reject invalid orders", _values,
         "Use JSON, integer cents and ValueError; reject zero, negatives and non-integer quantities."),
        ("registries snapshot and validate application data independently", _snapshots,
         "Copy both mappings; check matching SKUs, bounded names/count and nonnegative integer values."),
        ("agent wiring enforces read-only and finite call budgets", _controls,
         "Set ToolPolicy(read_only=True) and RunLimits(max_model_calls=4, max_tool_calls=3)."),
        ("a new SKU flows through both tools to the correct answer", _happy,
         "Use your registry in Agent and preserve the injected model and event observer."),
        ("unknown SKUs, empty stock and invalid quantities fail honestly", _failures,
         "Do not invent data or treat a quote as a reservation; let tool errors reach the model."),
        ("a rejected call is repaired without erasing the error", _recovery,
         "Recovery needs four model calls, including the final answer; invalid arguments must not execute."),
        ("repeated invalid calls stop without executing a tool", _budget,
         "Enforce three tool attempts; rejected attempts still consume the budget."),
        ("read-only policy blocks an undeclared action even if requested", _policy,
         "Enforce policy in Agent, not just in the prompt or advertised schemas."),
    ]
    for index, (description, observe, hint) in enumerate(cases, 1):
        try:
            passed = observe(app) is True
        except Exception:
            passed = False
        print(f"{'PASS' if passed else 'FAIL'}: {description}")
        if not passed:
            print(f"Focus on check {index}/{len(cases)}; later checks have not been assessed.")
            print(f"Hint: {hint}")
            print("Edit lessons/capstone_exercises.py; read lessons/04-build-an-application.md.")
            print("Try again: python3 -m lessons.lab04_capstone --check")
            return 1
    print("Capstone contract passed. Now add your own test and explain the observed controls.")
    return 0
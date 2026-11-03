"""Read-only inventory application using only stdlib and the public harness API.

Run: python -m examples.inventory_agent --sku WIDGET
This deterministic offline demonstration makes no model/network calls and never
loads .env. Replace OfflineModel with a model callable for another application.
"""

import argparse
import json
from functools import partial
from pathlib import Path

from harness import Agent, RunLimits, ToolPolicy, ToolRegistry


SYSTEM_PROMPT = "Check the supplied inventory using lookup_stock. Never invent stock quantities."


def lookup_stock(inventory, sku: str) -> str:
    """Look up a product's available stock using its exact case-sensitive SKU."""
    if sku not in inventory:
        raise ValueError("unknown SKU")
    return json.dumps({"sku": sku, "quantity": inventory[sku], "available": inventory[sku] > 0})


def create_tools(inventory):
    """Bind a validated snapshot; model arguments cannot replace the inventory."""
    snapshot = dict(inventory)
    if len(snapshot) > 1000 or any(type(sku) is not str or not 1 <= len(sku) <= 100
            or type(count) is not int or count < 0 for sku, count in snapshot.items()):
        raise ValueError("inventory needs at most 1000 SKU/quantity pairs with nonnegative integer counts")
    lookup = partial(lookup_stock, snapshot)
    lookup.__name__, lookup.__doc__ = lookup_stock.__name__, lookup_stock.__doc__
    tools = ToolRegistry()
    tools.tool(read_only=True)(lookup)
    return tools


def create_agent(model, inventory, **options):
    """Inject your model and runtime options; no globals, file setup or private overrides."""
    return Agent(model, SYSTEM_PROMPT, tools=create_tools(inventory), **options)


class OfflineModel:
    """Two-turn protocol demonstration, not an LLM or an accuracy evaluation."""

    def __call__(self, system, messages, tools, on_text=None):
        content = messages[-1]["content"]
        if isinstance(content, str):
            return {"stop_reason": "tool_use", "content": [{
                "type": "tool_use", "id": "stock", "name": "lookup_stock", "input": {"sku": content}}]}
        result = content[0]
        if result.get("is_error"):
            text = "Unable to check that SKU."
        else:
            stock = json.loads(result["content"])
            text = f"{stock['sku']}: {stock['quantity']} in stock."
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sku", default="WIDGET")
    parser.add_argument("--trace", type=Path, default=Path("inventory-trace.jsonl"))
    parser.add_argument("--tool-timeout", type=float, help="optional isolated-tool timeout in seconds")
    args = parser.parse_args()
    agent = create_agent(OfflineModel(), {"WIDGET": 7, "GADGET": 0}, quiet=True,
                         trace_path=args.trace, tool_policy=ToolPolicy(read_only=True),
                         limits=RunLimits(max_model_calls=2, max_tool_calls=1),
                         tool_timeout_seconds=args.tool_timeout)
    result = agent.run_result(args.sku)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" and result.usage.tool_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
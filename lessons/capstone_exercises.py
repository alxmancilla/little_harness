"""Your application. Start at lessons/04-build-an-application.md.

Edit these two factories, not the runtime or earlier exercises. This module must
depend only on the standard library and harness, not on lab helpers/solutions.
"""

from harness import Agent, RunLimits, ToolRegistry


SYSTEM_PROMPT = "Check stock and quote totals with the supplied tools. Never invent quantities or prices."


def create_tools(stock, prices):
    """TODO: snapshot the data and register lookup_stock and quote_total as read-only.

    lookup_stock(sku: str) returns JSON with sku and quantity.
    quote_total(sku: str, quantity: int) returns JSON with sku, quantity and
    total_cents. Reject unknown SKUs and nonpositive quantities with ValueError.
    Only sku/quantity, never stock/prices, belong in the model-facing schemas.
    See the walkthrough for validation rules and sample data.
    """
    return ToolRegistry()


def create_agent(model, stock, prices, *, trace_path, on_event=None):
    """TODO: enforce read-only execution and allow the bounded recovery scenario.

    Keep the injected model, trace path and observer. Do not load configuration,
    create a model client or set global state. The deliberately small starter
    limits prevent an endless script; finish with four model/three tool calls.
    """
    return Agent(model, SYSTEM_PROMPT, tools=create_tools(stock, prices), quiet=True,
                 trace_path=trace_path, on_event=on_event,
                 limits=RunLimits(max_model_calls=2, max_tool_calls=2))
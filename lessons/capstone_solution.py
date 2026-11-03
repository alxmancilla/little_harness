"""Separate reference application; imports only the standard library and harness."""

import json

from harness import Agent, RunLimits, ToolPolicy, ToolRegistry


SYSTEM_PROMPT = "Check stock and quote totals with the supplied tools. Never invent quantities or prices."


def create_tools(stock, prices):
    """Bind independent snapshots; expose only SKU and requested quantity."""
    stock, prices = dict(stock), dict(prices)
    if stock.keys() != prices.keys() or len(stock) > 1000:
        raise ValueError("stock and prices need matching SKUs, at most 1000")
    if any(type(sku) is not str or not 1 <= len(sku) <= 100 for sku in stock):
        raise ValueError("SKUs must be nonempty strings of at most 100 characters")
    if any(type(value) is not int or value < 0 for data in (stock, prices) for value in data.values()):
        raise ValueError("stock and prices must contain nonnegative integers")
    tools = ToolRegistry()

    @tools.tool(read_only=True)
    def lookup_stock(sku: str) -> str:
        """Read available units for an exact, case-sensitive SKU; do not reserve stock."""
        if sku not in stock:
            raise ValueError("unknown SKU")
        return json.dumps({"sku": sku, "quantity": stock[sku]})

    @tools.tool(read_only=True)
    def quote_total(sku: str, quantity: int) -> str:
        """Calculate an integer-cent quote; do not place an order or charge money."""
        if sku not in prices:
            raise ValueError("unknown SKU")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        return json.dumps({"sku": sku, "quantity": quantity,
                           "total_cents": prices[sku] * quantity})

    return tools


def create_agent(model, stock, prices, *, trace_path, on_event=None):
    """Application wiring and controls; no configuration loading or provider calls."""
    return Agent(model, SYSTEM_PROMPT, tools=create_tools(stock, prices), quiet=True,
                 trace_path=trace_path, on_event=on_event, tool_policy=ToolPolicy(read_only=True),
                 limits=RunLimits(max_model_calls=4, max_tool_calls=3))
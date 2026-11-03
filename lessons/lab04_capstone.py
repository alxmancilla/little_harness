"""Lab 4: build an application. Read lessons/04-build-an-application.md first."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .support import show_mode, show_run


STOCK = {"WIDGET": 7, "GADGET": 0}
PRICES = {"WIDGET": 250, "GADGET": 100}


class OrderModel:
    """Scripted protocol demonstration, not live-model reasoning or evaluation.

    happy: stock -> quote -> answer. repair: bad stock -> stock -> quote -> answer.
    repeat: keep sending bad stock arguments until the application's budget stops it.
    """

    def __init__(self, sku="WIDGET", quantity=2, scenario="happy"):
        if scenario not in ("happy", "repair", "repeat"):
            raise ValueError("unknown scenario")
        self.sku, self.quantity, self.scenario = sku, quantity, scenario
        self.calls = 0
        self.stock = None

    def request(self, name, **arguments):
        return {"stop_reason": "tool_use", "content": [{"type": "tool_use",
                "id": f"{name}-{self.calls}", "name": name, "input": arguments}]}

    def __call__(self, system, messages, tools, on_text=None):
        self.calls += 1
        content = messages[-1]["content"]
        if self.scenario == "repeat":
            return self.request("lookup_stock", sku=42)
        if isinstance(content, str):
            return self.request("lookup_stock", sku=42 if self.scenario == "repair" else self.sku)
        result = content[0]
        if result.get("is_error"):
            if self.scenario == "repair" and result["tool_use_id"] == "lookup_stock-1":
                return self.request("lookup_stock", sku=self.sku)
            text = "Unable to quote that order."
        elif result["tool_use_id"].startswith("lookup_stock-"):
            self.stock = json.loads(result["content"])["quantity"]
            return self.request("quote_total", sku=self.sku, quantity=self.quantity)
        else:
            quote = json.loads(result["content"])
            if self.stock < self.quantity:
                text = f"{self.sku}: insufficient stock ({self.stock} available)."
            else:
                text = (f"{self.sku}: {quote['quantity']} units cost {quote['total_cents']} cents; "
                        f"stock {self.stock}.")
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def implementation(use_solution):
    if use_solution:
        from . import capstone_solution
        return capstone_solution
    from . import capstone_exercises
    return capstone_exercises


def run(create_agent, *, scenario="happy", sku="WIDGET", quantity=2, stock=None, prices=None):
    """Exercise the learner's factory, not a preconfigured agent or global registry."""
    events = []
    with TemporaryDirectory(prefix="little-harness-capstone-") as directory:
        agent = create_agent(OrderModel(sku, quantity, scenario),
                             dict(STOCK if stock is None else stock),
                             dict(PRICES if prices is None else prices),
                             trace_path=Path(directory) / "trace.jsonl", on_event=events.append)
        result = agent.run_result("Quote the requested order using the supplied tools.")
    return result, agent.messages, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="assess application factories")
    parser.add_argument("--solution", action="store_true", help="use the separate reference application")
    parser.add_argument("--messages", action="store_true", help="print synthetic conversation messages")
    parser.add_argument("--events", action="store_true", help="print ordered metadata event names")
    parser.add_argument("--scenario", choices=("happy", "repair", "repeat"), default=None)
    args = parser.parse_args()
    if args.check and args.scenario is not None:
        parser.error("--check uses fixed cases; run --scenario separately")
    if args.check and (args.messages or args.events):
        parser.error("--messages and --events inspect demos; run separately from --check")
    show_mode(args.solution)
    app = implementation(args.solution)
    if args.check:
        from .capstone_checks import check_capstone
        return check_capstone(app)
    try:
        observation = run(app.create_agent, scenario=args.scenario or "happy")
    except Exception:
        print("Application could not run. Check your factory signatures and JSON tool results.")
        print("Read lessons/04-build-an-application.md; run --check for focused feedback.")
        return 1
    show_run(observation, args)
    print("Scripted demonstration only; --check assesses the application contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
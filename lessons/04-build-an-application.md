# Lab 4 — build an application

[Learning path](../LEARN.md) · Previous: [diagnose a failure](03-diagnose-a-failure.md)

**Outcome:** own a small application's tool registry, data and execution controls,
then demonstrate correct output, failure recovery and a bounded stop.
**Prerequisite:** Labs 0–3 and basic Python dictionaries/JSON. About 45–60 minutes.

## Your application: a read-only order quote

An order quote needs two facts: available stock and a calculated price. Build two
tools, then wire them into `Agent` yourself. A quote **does not** reserve stock,
place an order or charge money. All data is synthetic and stays offline.

The scripted model chooses stock → quote → answer. This is an inspectable protocol
exercise, not evidence that an LLM can plan or that this fixed workflow needs one.
In a real product, start with a direct function/workflow when that is sufficient.

Edit only [capstone_exercises.py](capstone_exercises.py). It is a separate
application module; your earlier exercises and solutions are not loaded or changed.
Do not import lab helpers or solutions into your application. The harness owns the
loop, not your product data, calculations or factory functions.

Before running, predict what happens when the starter advertises **no tools**:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab04_capstone --messages --events
python3 -m lessons.lab04_capstone --check
````
</augment_code_snippet>

Expect the starter demo to complete with `Unable to quote that order.`, one
unknown-tool error and zero executions. Completing the loop is not the same as
meeting the application requirement. The checker stops at the first unfinished
contract and prints a hint; an initial failure is expected, not a setup problem.
The checker is not a sandbox: an infinite loop in your own code can still hang.

## 1. Build your registry

Implement `create_tools(stock, prices)`. Inputs are two mappings from SKU (a
case-sensitive product identifier) to integer stock counts and integer-cent prices.
For example, `{"WIDGET": 7}` and `{"WIDGET": 250}` mean seven units at 250 cents each.

- Copy both mappings so later caller changes cannot alter this application's data.
- Require matching SKU sets and at most 1,000 entries. Each SKU must be a string
  with 1–100 characters. Values must be nonnegative integers; reject booleans.
  Empty mappings and zero stock/prices are valid.
- Raise `ValueError` for invalid application data.
- Create a fresh `ToolRegistry`; register the two functions with `read_only=True`.
- Give both functions useful docstrings, primitive argument annotations and string
  returns containing JSON. Bind the data in closures, not model-facing arguments.

| Tool signature | Successful JSON object | Failure |
| --- | --- | --- |
| `lookup_stock(sku: str) -> str` | `{"sku": "WIDGET", "quantity": 7}` | Unknown SKU raises `ValueError` |
| `quote_total(sku: str, quantity: int) -> str` | `{"sku": "WIDGET", "quantity": 2, "total_cents": 500}` | Unknown SKU or nonpositive/non-integer quantity raises `ValueError` |

Multiply price by requested quantity; use integer cents rather than floating-point
money. A quote can be calculated even when stock is insufficient; the model must
report that separately. Neither tool changes the inventory.

The model-facing schemas must expose **only** `sku`, or `sku` and `quantity`.
Keep `stock` and `prices` out of them. Follow the registry pattern from Lab 2 or the
[inventory example](../examples/inventory_agent.py); do not modify a global registry.

**Predict → run → change → explain:** why is a negative integer schema-valid but
not a valid order? Run `--check` after each change; it rechecks earlier contracts.

## 2. Wire your agent and controls

Finish `create_agent(model, stock, prices, *, trace_path, on_event=None)`:

1. Return an `Agent` using your registry and application prompt.
2. Preserve the supplied model, trace path and observer. Use `quiet=True`.
3. Enforce `ToolPolicy(read_only=True)` in the runtime. A prompt is not permission.
4. Set `RunLimits(max_model_calls=4, max_tool_calls=3)`.

These are maximums, not a requirement to spend all calls. The successful path uses
three model calls and two tool attempts. Recovery needs one extra model call and
one extra attempt, including the final-answer call.

Do not load `.env`, construct a provider client or import teaching/evaluation apps.
The runner supplies synthetic data, a scripted model and a temporary trace path.
It removes that trace afterward and never writes a session or workspace.

`read_only=True` is a **trusted declaration**, not code analysis: declaring a
function safe does not make it safe. Review the tool bodies. The policy prevents
dispatch of tools without that declaration; it is not an operating-system sandbox.

## 3. Observe success, repair and stopping

Once your factories work, run all three scenarios:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab04_capstone --scenario happy --messages
python3 -m lessons.lab04_capstone --scenario repair --events
python3 -m lessons.lab04_capstone --scenario repeat --events
````
</augment_code_snippet>

| Scenario | Result | Model calls | Tool attempts | Actual executions |
| --- | --- | --- | --- | --- |
| `happy` | `completed`; `WIDGET: 2 units cost 500 cents; stock 7.` | 3 | 2 | 2 |
| `repair` | Same answer; one earlier error remains recorded | 4 | 3 | 2 |
| `repeat` | `stopped`, `max_tool_calls`; no final answer | 4 | 3 | 0 |

In `repair`, the first stock request uses an integer SKU. Schema validation rejects
it **before execution**. The script proposes a corrected request with a new ID,
then requests the price. This is safe correction of a read-only, unexecuted call,
not a general retry rule for payments or other side effects.

In `repeat`, a fourth request is observed, but only three attempts are admitted.
The fourth request gets a paired not-run result. Check `tool_started` events to
count real executions; not every `tool_finished` event means a function ran.

Ordinary demos exit 0 when they run, even for expected stops. `--check` exits 1 on
an unfinished contract and 0 on success. Run `--messages`/`--events`/`--scenario`
separately from assessment; assessment uses fixed inputs and fresh applications.

## 4. Assess the application, not just its happy path

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab04_capstone --check
````
</augment_code_snippet>

Nine checks cover schemas, calculations, validated independent snapshots, agent
controls, a new SKU, honest failures, recovery, bounded repetition and policy
denial. The last check adds a synthetic undeclared action and requests it directly:
it must stay hidden from advertised schemas **and** never execute.

Use hints first. If stuck, compare [capstone_solution.py](capstone_solution.py),
which implements the same application contract without importing earlier solutions.
Reference mode never overwrites your code:

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m lessons.lab04_capstone --solution --check
python3 -m lessons.lab04_capstone --solution --scenario repair --messages --events
````
</augment_code_snippet>

## 5. Add a test you own

Choose a case beyond the supplied examples: free products, an empty catalogue,
case-sensitive SKUs or one unit of stock. Write the expected result **before**
running the tool. A meaningful test should fail if you break the calculation.

For example, create a new `tests/test_my_order_app.py` (do not overwrite existing work)
and adapt this test to your chosen case:

<augment_code_snippet path="tests/test_my_order_app.py" mode="EXCERPT">
````python
import json
import unittest
from lessons.capstone_exercises import create_tools
class OrderTests(unittest.TestCase):
    def test_free_product(self):
        tools = create_tools({"FREE": 1}, {"FREE": 0})
        quote = json.loads(tools["quote_total"]["fn"]("FREE", 1))
        self.assertEqual(quote["total_cents"], 0)
````
</augment_code_snippet>

<augment_code_snippet mode="EXCERPT">
````sh
python3 -m unittest -v tests.test_my_order_app
````
</augment_code_snippet>

## Completion rubric and independent handoff

- All nine application checks pass and your additional test detects a real mistake.
- Explain why the repaired run has three attempts but only two executions.
- Demonstrate an unknown SKU without inventing stock/prices; show insufficient
  stock without claiming an order was placed. The checker includes both cases.
- Explain why the read-only declaration must match the code and why hiding a
  tool schema alone does not authorize or prevent dispatch.
- Name a case where a fixed workflow is simpler than an agent.

To start your own application, copy your completed module to a **new** file such as
`order_app.py`, outside `lessons/`, and update your test import to `order_app`.
It needs only the standard library and the independently copyable `harness/`
package. Supply your own model callable and fresh trace path from a caller; neither
the lesson runner nor its scripted model is part of your application contract.
Never reuse a sensitive trace or overwrite another application's data.

Then change the domain or add one tool and its tests. Keep the scope read-only
until you can explain authorization, idempotency and uncertain effects.
For optional next topics, return to [after this milestone](../LEARN.md#after-this-milestone).
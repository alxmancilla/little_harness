# From the loop to a useful application

Little Harness is a small runtime, not an agent platform. A useful application
supplies the model, instructions, approved data, tool registry and output contract.
The runtime executes that contract's tool loop with explicit limits and controls.
Start with [the offline labs](../LEARN.md) if these mechanics are new.

## Follow the core loop

Read [Agent._run](../harness/agent.py), starting at `def _run(self, task)`:

1. Append the user task and advertise only the application's allowed schemas.
2. Compact older context if necessary, counting that model call against the budget.
3. Ask the model for a normalized response; append its text/tool requests to history.
4. With no tool requests, return an execution result. This is not proof of correctness.
5. Otherwise, check each attempt's budget and dispatch through `_run_tool`.
6. `_run_tool` calls its `_gate_*` methods in order (admission, known tool, inputs,
   policy, timeout setting, approval) before `_dispatch` runs any action.
7. Append paired tool results together, then ask the model again. Stop on a limit
   or uncertain effect; do not blindly retry an action that might already have happened.

The optional events, worker processes and journal surround this same loop; they
do not introduce another planner or execution engine. For the full model callable
and message contract, see [the specification](HARNESS_SPEC.md#32-model-and-conversation-representation).

## Run a useful narrow example

[The support assistant](../examples/support_agent.py) searches two approved local
notes and looks up a product. Its data and policies are fictional, not real
hardware instructions or warranties. It uses the same factory offline and live.

````sh
python3 -m examples.support_agent
python3 -m examples.support_agent "warranty review" --sku GADGET
python3 -m examples.support_agent "penguin migration"
````

The default run makes two scripted model calls and two read-only tool calls.
The no-evidence request escalates with `needs_human=true`, not an invented answer.
Offline success demonstrates the protocol, not real-model accuracy.

The CLI writes metadata-only `support-trace.jsonl`; it never loads `.env`.
For paid calls, configure provider settings in your shell and explicitly run
`python3 -m examples.support_agent --live`. Call/time limits are cooperative;
they cannot guarantee a billing ceiling. Do not submit confidential requests
without checking your provider and data-access policies.

## Own the application factory

Inject a model instead of selecting one inside tools or runtime imports:

````python
from examples.support_agent import OfflineModel, PRODUCTS, create_agent, load_documents
agent = create_agent(OfflineModel(), load_documents(), PRODUCTS)
````

The factory takes fresh snapshots and defaults to six model calls, six tool
attempts, 30 seconds, quiet output and a read-only policy. Its two module-level
tool functions are bound with `functools.partial`; only `query` or `sku` appears
in model-visible schemas. Replace the approved documents/catalog for your domain.
Do not expose credentials, database connections or entire filesystems as arguments.

Tools accept explicitly annotated `str`, `int`, `float` or `bool` parameters,
including keyword-only parameters. `Literal[...]` limits a parameter to fixed
string or integer choices that the runtime enforces; `Annotated[type, "text"]`
gives the model a per-parameter description. Unsupported/missing annotations, incompatible
defaults, duplicate names, positional-only/variadic parameters, coroutines and
generators fail at registration. Builtin postponed annotations work without
evaluating annotation expressions. Named synchronous callable objects and partials
also work; their underlying implementation must be synchronous, not a coroutine
or generator. Return annotations do not define an output schema.

## Validate answers separately from execution

The support application's `parse_recommendation` checks:

- Execution completed, rather than stopping on a budget, refusal or truncation.
- One JSON object, no duplicate fields or heuristic extraction from prose.
- Exact keys and bounded, correctly typed values for the requested SKU.
- Unique citations to successful search results from the current task only.
- A successful lookup of that product before accepting a non-escalated answer.
- Missing evidence/product data requires escalation, not a claimed supported answer.

The caller uses the returned validated value, never unchecked `result.answer`:

````python
task = json.dumps({"sku": "WIDGET", "question": "How do I reconnect?"})
result = agent.run_result(task)
recommendation = parse_recommendation(result, agent.messages, "WIDGET")
````

Here `json` and `parse_recommendation` are imports in the application module.
Invalid output raises an application error: the runtime makes no hidden repair call
or automatic retry.

The application may choose to give the model one visible chance to fix its answer.
`run_validated` sends the validation error back as a new user turn and validates
the reply; evidence retrieved earlier in the same task still counts:

````python
result, recommendation = run_validated(agent, task, "WIDGET", repairs=1)
````

The error text is written by the application, never copied from model output. Only a
completed run with an invalid answer is repaired; budget stops, refusals and model
errors raise immediately. Each repair turn has its own budget, and `result` describes
only the last turn. Provenance checks do **not** verify that prose correctly interprets
evidence. Review citations and domain correctness, especially with real models.
Use a fresh agent per independent request; keep histories and traces separate.

## Approve a draft, never send automatically

````sh
python3 -m examples.support_agent --save-draft support-draft.json
````

Only after output validation does host code show the complete draft and ask for
approval. The draft approval callback must return literal `True`; strings, numbers
and other truthy values do not authorize saving. EOF/default refusal creates no
draft. Approved saving exclusively creates
the human-selected path, never overwrites an existing file and never sends a message.
This host operation is outside the model loop and its tool budget/journal; the caller
owns file permissions, errors and retention. Keep it a draft-only workflow.

## Keep observation content explicit

Traces and events independently default to metadata only:

- `trace_include_content=True`: include tasks, normalized model content, tool
  arguments/results, summaries and error details in JSONL traces.
- `event_include_content=True`: include content in callback events.
- Teaching CLI: `HARNESS_TRACE_CONTENT=1` explicitly enables content traces.
- Support CLI: `--trace-content` explicitly enables content traces.

**Migration:** trace consumers that read `text`, `content`, `input`, `output`,
`summary` or `error` must explicitly opt in. HotpotQA evaluation already opts in
for its controlled benchmark artifacts. Sessions and returned results still contain
conversation content. Tool names, IDs, stop reasons and caller `trace_context`
can also disclose information; none of these flags is general secret redaction.

## Reuse without a framework

Copy the complete `harness/` directory into your project; no installation is required.
For optional installation, build a pure-Python wheel with the standard library:

````sh
package_dir=$(mktemp -d)
python3 -m tools.package_runtime --output "$package_dir"
python3 -m pip install --no-index --no-deps "$package_dir/little_harness-0.1.0-py3-none-any.whl"
````

Use your own virtual environment for installation. The builder packages only runtime
Python source and wheel metadata, records file hashes and refuses overwrites. It
does not require a build backend or third-party build packages. Labs, application
examples, datasets and secrets are excluded. This is a local distribution utility,
not a release/publishing system; choose a license before public redistribution and
update its version before publishing a changed release. The import remains `harness`.

## Evaluate the application, not only the loop

Run `python3 -m unittest -v tests.test_support_agent`. Cases cover useful answers,
unknown products, absent evidence, invalid output/citations, recovery from bad
arguments, budgets, privacy, approved/denied saving and standalone execution.
These deterministic tests establish contracts, not general answer quality. For live
evaluation, review a small domain-approved set for factual correctness, escalation,
citation support and action safety; do not treat scripted answers as ground truth.

The [support evaluation workflow](SUPPORT_EVALUATION.md) provides seventeen explicitly
unreviewed synthetic draft cases, an offline-first runner, bounded live opt-in,
mechanical checks, resource reporting and an unscored human-review worksheet.
Run `python3 -m evaluation.support.evaluate --output runs/support-smoke-01`.
Review the draft expectations before treating them as a benchmark; structural
success does not verify that prose is correct or safe.
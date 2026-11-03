# Little Harness — tool dispatch policies

`Agent(..., tool_policy=ToolPolicy(read_only=False, argument_checks={}))` adds
application-owned checks before tool dispatch. `ToolPolicy` is importable from
`harness`. Omitting it preserves default behavior, including
schemas, approval arguments and trace/event kinds. No dependencies are required.

## Explicit read-only mode

- `ToolPolicy(read_only=True)` allows only registry entries whose `read_only` is
  literal `True`. Missing values, `False`, `1` and strings do not qualify.
- Built-in declarations cover `calculator`, `current_time`, `list_files`,
  `read_file` and `search_documents`. `write_file` and `http_get` are not read-only;
  even an HTTP GET can cause remote effects or disclose data.
- Use `@tools.tool(read_only=True)` on a `ToolRegistry` for trusted custom tools with no externally visible
  effects. Manually assembled registry entries can set `"read_only": True` too.
  This declaration is trusted application metadata, not a property the harness
  verifies. The default decorator leaves tools unclassified.
- `risky` controls approval independently. A risky declared reader still needs
  approval; `risky=False` does not imply read-only. No policy bypasses approval.
- Blocked tool schemas are hidden from the model, but dispatch checks still apply
  to fabricated requests or calls present in conversation history. Argument
  restrictions do not rewrite model schemas; they are enforced on every dispatch.

The CLI accepts `HARNESS_READ_ONLY=1`, `0` or unset/empty. Other values fail startup
without echoing the value. This setting applies to every teaching CLI mode,
including auto-approved mock writes. Library and HotpotQA agents do not read it;
library callers must pass a policy explicitly.

## Argument checks

`argument_checks` maps allowlisted tool names to trusted callables `(args) -> bool`.
Unknown names fail agent construction to catch configuration mistakes. The mapping
is copied and frozen; its callables can still close over mutable application state.
Only tools named in this mapping receive argument restrictions. Read-only mode and
an argument check both have to allow the call when both are configured.

<augment_code_snippet mode="EXCERPT">
````python
from harness import Agent, ToolPolicy
policy = ToolPolicy(argument_checks={
    "write_file": lambda args: args["path"] == "report.txt" and len(args["content"]) <= 2000,
})
agent = Agent(model, "Draft a short report.", tools=tools, tool_policy=policy, approve=application_approval)
````
</augment_code_snippet>

Here `tools` is the application's explicit registry containing `write_file`.

- Return literal `True` to allow; all other return values deny. This is validation,
  not argument rewriting. A denial returns a not-run tool error to the model so it
  may choose a different permitted request within its remaining budgets.
- Ordinary predicate exceptions fail closed and stop the task with
  `RunResult.status="stopped"`, `stop_reason="tool_policy_error"`. Exception text
  is omitted. Later calls in the same batch get explicit not-run results; completed
  results are retained. Fix the policy before explicitly continuing.
- With any policy configured, executed arguments are a detached JSON snapshot.
  Predicates receive another deep copy, and approval receives its own copy. Their
  mutations cannot rewrite executed arguments or the model's conversation. JSON
  snapshot failures also stop as `tool_policy_error`; NaN/infinity are rejected.
- Optional arguments remain omitted; function defaults are not materialized. A
  predicate must account for the tool's actual defaults, e.g.
  `args.get("subdir", ".")`. It must not assume every optional key is present.
- Exceptions derived directly from `BaseException` propagate rather than becoming
  policy errors. Ctrl+C uses existing cancellation cleanup: the active request is
  conservatively reported uncertain even if the interruption occurred in a check
  before dispatch. No approval, journal action or tool invocation has yet occurred.

## Ordering, accounting and observations

Dispatch follows this order:

1. Existing run recovery/budget gates, allowlist lookup and input validation.
2. Detached argument snapshot; explicit read-only gate; optional argument predicate.
3. Metadata-only `tool_policy` trace/event and cooperative budget recheck.
4. Risky-action approval and another budget recheck.
5. Optional durable journal start, tool start notification and inline/worker execution.
6. Existing outcome commits, result preservation and finish notifications.

Invalid/unknown/truncated requests do not invoke predicates. Policy denials/errors
create no action record, worker, approval request or tool start. They still count
as tool attempts and their results count as errors. Predicates and observers consume
elapsed time; successful checks recheck the deadline before approval/dispatch.
Policy checks run in the parent even when a tool uses an isolated worker.

`tool_policy` observations contain tool name/call ID, `allowed` and `reason`:
`allowed`, `read_only_required`, `argument_rejected` or `policy_error`. They contain
no arguments, outputs or exception details, even with event content enabled.
The corresponding `tool_finished` is `not_run` on denial/error. Ordinary observer
failures cannot grant permission or cause an automatic tool retry. Normal traces,
sessions and content-enabled finish events still contain sensitive request data.

## Trust boundaries

These checks are **not an OS sandbox, authorization service or secret detector**.
Application code, tool declarations, predicates, approval callbacks and observers
must be trusted. Do not let model-generated content modify policies or the registry.
Do not mutate the agent, registry or execution configuration inside a callback.

Predicates run synchronously and cannot be forcibly timed out. Keep them fast,
side-effect-free and deterministic; do not perform network requests or writes in
a check. A timed tool worker does not isolate its predicate. Policy errors stop
execution but cannot undo side effects performed by buggy application code.

Read-only does not mean confidential: permitted reads can disclose data to the
model provider, traces and saved sessions. Restrict `tools=`, use OS permissions,
curate readable data, and validate paths/URLs inside tools. Path/string predicates
are not race-free filesystem confinement and cannot control redirects or remote
effects. Policies add no process/network isolation, exactly-once guarantee, retry
or output replay. Existing approval, timeout and journal limitations still apply.
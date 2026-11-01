# mini_harness

A minimal agent harness in a single file, [mini_harness.py](mini_harness.py), built only on the Python 3 standard library. No dependencies to install.

It shows the five pieces every agent harness has:

1. **Tool registry**: a `@tool` decorator turns a plain Python function into a tool. The JSON schema comes from the signature and the description from the docstring.
2. **Model client**: `AnthropicModel` streams the Messages API (`"stream": true`) with `urllib`, parsing server-sent events line by line and printing text as it arrives. `ScriptedModel` replays canned responses for offline runs.
3. **Agent loop**: model, tool calls, tool results, model, until the model stops calling tools.
4. **Guardrails**: a step budget, a sandboxed workspace, errors returned to the model as data, truncated tool output, and human approval for risky tools.
5. **Tracing**: every event is appended to `trace.jsonl`.

On top of these, **context management** keeps the history from growing forever: when it passes a size threshold, old turns are replaced by a model-written summary (see [Context compaction](#context-compaction)).

## Requirements

- Python 3.9 or later (the code uses `Path.is_relative_to`)
- An Anthropic API key, for real runs only

## Setup with a virtual environment

The harness has no dependencies, so the venv only isolates the Python version and keeps your shell tidy.

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python mini_harness.py mock        # confirm it runs
```

Run `deactivate` to leave the environment. Delete `.venv/` to remove it. `.venv/` is already in `.gitignore`.

## Usage

```sh
python3 mini_harness.py mock               # offline demo, no API key needed
python3 mini_harness.py "your task here"   # one task against the real API
python3 mini_harness.py                    # interactive chat; saved to session.json and resumed on restart
```

Type `exit` or press Ctrl+C to leave interactive mode.

## Configuration

Copy [.env.example](.env.example) to `.env` and fill it in. The harness reads environment variables only and does not load `.env` itself, so export the file before running:

```sh
cp .env.example .env
set -a; source .env; set +a
python3 mini_harness.py "What is 17 * 23 + 4?"
```

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | For real runs | none | API credential |
| `HARNESS_MODEL` | No | `claude-sonnet-5-5` | Model ID |
| `ANTHROPIC_WORKSPACE_ID` | No | none | Only needed if the key is not scoped to one workspace |
| `HARNESS_COMPACT_CHARS` | No | `40000` | History size (JSON characters) that triggers compaction |
| `HARNESS_SESSION` | No | `session.json` | Interactive mode: file the conversation is saved to and resumed from |
| `HARNESS_WORKSPACE` | No | `./workspace` | Sandbox directory for file tools |

## Built-in tools

| Tool | Risky | Description |
| --- | --- | --- |
| `calculator` | no | Safe arithmetic via an `ast` interpreter, never `eval()` |
| `current_time` | no | Current UTC time, ISO 8601 |
| `list_files` | no | List files in the workspace |
| `read_file` | no | Read a UTF-8 file from the workspace |
| `write_file` | yes | Create or overwrite a file; asks for `y/N` approval first |
| `http_get` | yes | GET an http(s) URL and return the body as text; asks for `y/N` approval first |

File paths are resolved against the workspace and rejected if they escape it.

## Adding a tool

```python
@tool()                      # use @tool(risky=True) to require approval
def word_count(text: str) -> str:
    """Count the words in a piece of text."""
    return str(len(text.split()))
```

Supported parameter types are `str`, `int`, `float` and `bool`. Parameters without a default are marked required.

## Streaming

Every request sets `"stream": true`. `sse_events` splits the response into events (a blank line ends one), and `AnthropicModel._read_stream` folds them back into the same response dict a non-streaming call returns: text deltas are concatenated and passed to the `on_text` callback, tool arguments arrive as JSON fragments and are parsed at `content_block_stop`, and `message_delta` supplies `stop_reason` and `usage`.

A model is any callable `(system, messages, tools, on_text=None) -> response`. Models that cannot stream, like `ScriptedModel`, ignore `on_text`, and the agent prints their text after the call instead. The summarizer call for compaction streams but is not printed.

A stream that ends before `message_stop`, is cut off by a dropped connection or read timeout, or carries an `error` event such as `overloaded_error`, raises `RuntimeError`. Retries with backoff cover only failures before the stream starts, because a retry after text was printed would repeat it.

## Context compaction

Before each model call, the agent measures the JSON size of `messages`. Above `HARNESS_COMPACT_CHARS` it:

1. keeps the newest turns, up to a third of the threshold (or just the newest assistant turn if that alone is larger), starting at an assistant message so no `tool_use` is separated from its `tool_result`;
2. asks the model, with no tools, to summarize everything older, working from a plain-text transcript with long tool inputs and results clipped;
3. replaces the old messages with one user message holding the summary. If the model returns no summary text, history is left untouched.

A later compaction folds the earlier summary into the new one. It runs again only after the history has grown by at least a third of the threshold, so a large recent tail does not trigger a summary on every step. Size is counted in characters, not tokens, to stay dependency-free.

## Sessions

Interactive mode writes `messages` to `session.json` (or `HARNESS_SESSION`) and loads it on startup, so a restart picks up the conversation where it left off. `mock` and one-shot runs do not read or write a session.

- The file is written atomically (temp file, then rename) after the task is added, after each batch of tool results, and after the final reply. It is never written between a `tool_use` and its results.
- If the process died mid-run, the file ends on a user message. Your next input is folded into that message, so the model sees the interrupted work and the new request together.
- A file that is not valid JSON stops the program without being overwritten.
- To start fresh, delete the file. `trace.jsonl` is a separate log and is not used for resuming.

## Tracing

Each run appends JSON lines to `trace.jsonl` with these `kind` values: `task`, `model`, `tool`, `compact` and `stopped`. Inspect a run with:

```sh
jq -c '{kind, step, name}' trace.jsonl
```

## Files

- `mini_harness.py`: the harness
- `workspace/`: sandbox the agent reads and writes (created on first run)
- `trace.jsonl`: event log (created on first run)
- `session.json`: saved interactive conversation (created on first interactive turn)

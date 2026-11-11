# Documentation

New here? Start with the [README](../README.md) or the [learning path](../LEARN.md).

## Build an agent

- [Build a useful application](BUILD_AN_AGENT.md): a step-by-step recipe, reading the loop, and reusing the runtime in your project
- [Reference](REFERENCE.md): providers, configuration, controls, retrieval, storage and tests
- [Tool policies and approval](TOOL_POLICIES.md): read-only mode, argument checks and their order
- [Event callbacks](HARNESS_EVENTS.md): watching a run from your own code or UI

## Understand the runtime

- [Architecture](ARCHITECTURE.md): components, process boundaries and diagrams
- [Runtime specification](HARNESS_SPEC.md): the supported API and its guarantees
- [Storage specification](STORAGE_SPEC.md): PostgreSQL and MongoDB sessions, journals and memory

## Evaluate

- [Support assistant evaluation](SUPPORT_EVALUATION.md): offline cases and a human review worksheet
- [Live baselines](BASELINES.md): first recorded results with real models
- [HotpotQA comparison](../datasets/HOTPOTQA_BASELINE.md): single agent vs. researcher/reviewer

## Design history

Plans and checkpoints from building the project. Useful for contributors; not needed
to use the harness.

- [Harness separation contract](design/HARNESS_SEPARATION_SPEC.md)
- [Harness migration plan](design/HARNESS_MIGRATION_PLAN.md)
- [Storage plan](design/STORAGE_PLAN.md)

# Agent guidelines

## Scope and style

This is a small project, not a large application. Prefer **good Python practice** without over-engineering: clear, readable code; sensible structure; avoid unnecessary abstractions, layers, or ceremony.

Take a **data science** mindset: experiments, metrics, reproducibility, and straightforward scripts/notebooks over heavy frameworks.

## Tooling

Use **[uv](https://docs.astral.sh/uv/)** for the environment and dependencies:

- `uv add <package>` to add dependencies
- Run commands with `uv run` as appropriate for this repo

## Documentation layout

- **`EXPERIMENT_PLAN.md`** and **`TEST_TASK.md`** at the repo root are **authoritative**. Treat them as the source of truth for what to build and how to validate it. Generally follow what they say.
- If implementation reveals something that should change in those documents (scope, metrics, assumptions, or acceptance criteria), **stop and notify the user** before diverging—do not silently reinterpret them.
- Less important or auxiliary notes for agents: put under **`.llm_context/`** (not the repo root).

# Project rules

- Keep changes small and reviewable.
- Do not add features that were not requested.
- Do not refactor unrelated code.
- Behavior changes require tests.
- Bug fixes require regression tests.
- Never remove or weaken tests merely to make CI pass.
- Never claim a test passed unless it was actually executed.
- Keep runtime and development dependencies declared in pyproject.toml.
- Do not hard-code secrets.
- Prefer simple implementations over unnecessary abstractions.
- Before considering work complete, run lint, format check, type check, and tests:

  ```sh
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src
  uv run pytest
  ```

- Report any failures honestly.

- Code comments must be in English.
- Annotate key function parameters with inline comments.
- Do not use emoji anywhere in code, comments, commit messages, or docs.
- Use `uv add` / `uv remove` to change dependencies; never edit uv.lock by hand.
- Pin third-party GitHub Actions to a full commit SHA with the version in a trailing comment. Never guess action versions; verify the tag exists.
- Never hold a database session or connection across an external call (LLM provider, HTTP). Scope sessions to the database work only.

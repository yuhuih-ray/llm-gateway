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
  ruff check .
  ruff format --check .
  mypy src
  python -m pytest
  ```

- Report any failures honestly.

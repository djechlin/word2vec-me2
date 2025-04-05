All Claude modes:

- Trim trailing whitespace.
- All functions, including private functions, should have language-appropriate docstrings. A good docstring begins with a concise fragment writen from the function's point of view (e.g. "Converts..." not "Convert..."). It is important the documentation be local. For instance "outputs data for use in processing by the xyz function" is bad documentation as it forces the reader to understand both functions. "outputs data in abc structure for ease of parsing by the xyz function" achieves both high context and locality of documentation.
- Unless I instruct otherwise, one export per file.

Claude agent mode:

- After every change create a git commit whose commit message begins with "[Claude]".
- Git pre-commit hook will automatically run `ruff check` and `ruff format` to ensure code quality. The hook will block commits that don't pass linting.


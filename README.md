# sqlfluff-pyspark

Lint and optionally fix SQL embedded in `spark.sql(...)` calls using [SQLFluff](https://github.com/sqlfluff/sqlfluff).

## Installation

```bash
pip install sqlfluff-pyspark
```

## Command Line Usage

```bash
# Lint spark.sql strings
sqlfluff-pyspark path/to/file.py another_file.py

# Apply fixes (writes changes back to the files)
sqlfluff-pyspark --fix path/to/file.py
```

Exit codes:
- 0: Success / no actionable snippets / no violations
- 1: Lint violations found (printed to stderr) or fixes applied and re-lint failed
- >1: Unexpected error from sqlfluff invocation

## Pre-commit Hook Integration

This project provides pre-commit hooks so you can automatically lint (and optionally fix) `spark.sql` strings before committing.

Add the following to your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/danieltom/sqlfluff-pyspark
    rev: v0.1.1  # or the latest tag
    hooks:
      - id: sqlfluff-pyspark-lint
      # Optional fix hook (will modify files). Normally run separately or in CI.
      # - id: sqlfluff-pyspark-fix
```

Then install:

```bash
pre-commit install
```

### Choosing lint vs fix hook

Use the lint hook locally to keep commits clean. Run the fix hook manually:

```bash
pre-commit run sqlfluff-pyspark-fix --all-files
```

or directly:

```bash
sqlfluff-pyspark --fix your_script.py
```

## How It Works

1. Parses Python source with `ast` to find calls to `spark.sql(...)`.
2. Extracts string literals (skips f-strings and non-constant expressions).
3. Writes each snippet to a temp SQL file and calls `sqlfluff` on them.
4. Translates violation line numbers back to original file positions.
5. For fixes, rewrites the original string literals preserving style (indent, quoting) when possible.

Inline directive `sqlfluff:` anywhere in the literal will skip linting/fixing for that snippet.

## Limitations / Notes

- F-strings are skipped (dynamic content).
- Concatenated string literals are supported (`"SELECT" + " 1"`).
- Multi-line snippets are normalized with indentation preserved.
- Only supports `.py` files passed explicitly; it does not auto-discover.

## Development

```bash
pip install -e .[dev]
pytest -vv
```

## Versioning

The hook `rev` should match a published tag. If using main, be aware of potential breaking changes.

## License

MIT


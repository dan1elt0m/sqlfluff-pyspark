#!/usr/bin/env python3
"""Lint or fix SQL strings passed to spark.sql using sqlfluff."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

STRING_PREFIX_RE = re.compile(r'^([furbFURB]*)("""|\'\'\'|"|\')')
DEFAULT_QUOTE = '"""'


@dataclass
class ExtractedSql:
    """Container for extracted SQL snippets."""

    sql: str
    file_path: Path
    base_line: int
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None
    literal: str
    prefix: str
    quote: str
    indent: str
    continuation_indent: str
    leading_newline: bool
    trailing_newline: bool
    fixable: bool
    skip: bool = False

    def temp_content(self) -> str:
        text = self.sql
        if not text.endswith("\n"):
            text = f"{text}\n"
        return text

    def build_literal(self, new_sql: str) -> str:
        multi_line = "\n" in new_sql or "\r" in new_sql

        quote = self.quote
        indent = self.indent
        leading_newline = self.leading_newline
        trailing_newline = self.trailing_newline

        if multi_line:
            indent = indent or self.continuation_indent
            leading_newline = True
            trailing_newline = True
            if len(quote) == 1:
                quote = DEFAULT_QUOTE

        lines = new_sql.splitlines()
        if indent and lines:
            reindented: list[str] = []
            for line in lines:
                if line:
                    reindented.append(f"{indent}{line}")
                else:
                    reindented.append(indent.rstrip())
            lines = reindented

        body = "\n".join(lines)
        if leading_newline:
            body = f"\n{body}"
        if trailing_newline:
            body = f"{body}\n"

        return f"{self.prefix}{quote}{body}{quote}"


class SparkSqlExtractor(ast.NodeVisitor):
    """AST visitor that collects string literals passed into spark.sql."""

    def __init__(self, source: str, file_path: Path) -> None:
        self._source = source
        self._file_path = file_path
        self._results: list[ExtractedSql] = []

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_spark_sql_call(node):
            extracted = self._extract_sql_argument(node)
            if extracted is not None:
                self._results.append(extracted)
        self.generic_visit(node)

    def results(self) -> list[ExtractedSql]:
        return self._results

    def _is_spark_sql_call(self, node: ast.Call) -> bool:
        func = node.func
        return isinstance(func, ast.Attribute) and func.attr == "sql"

    def _extract_sql_argument(self, node: ast.Call) -> ExtractedSql | None:
        if not node.args:
            return None

        value_node = node.args[0]
        raw_sql = self._coerce_to_string(value_node)
        if raw_sql is None:
            return None

        normalized_sql = textwrap.dedent(raw_sql).strip()
        if not normalized_sql:
            return None

        literal_segment = ast.get_source_segment(self._source, value_node) or ""
        prefix, quote = _split_prefix_and_quote(literal_segment)
        lowered_sql = raw_sql.lower()
        skip = "sqlfluff:" in lowered_sql
        fixable = "f" not in prefix.lower() and not skip

        base_line = getattr(value_node, "lineno", getattr(node, "lineno", 1))
        leading_newlines = _count_leading_newlines(raw_sql)
        base_line += leading_newlines

        indent = _infer_indent(raw_sql)
        continuation_indent = _build_continuation_indent(
            getattr(value_node, "col_offset", 0)
        )
        leading_newline = raw_sql.startswith("\n")
        trailing_newline = raw_sql.endswith("\n")

        return ExtractedSql(
            sql=normalized_sql,
            file_path=self._file_path,
            base_line=base_line,
            lineno=getattr(value_node, "lineno", 1),
            col_offset=getattr(value_node, "col_offset", 0),
            end_lineno=getattr(value_node, "end_lineno", None),
            end_col_offset=getattr(value_node, "end_col_offset", None),
            literal=literal_segment,
            prefix=prefix,
            quote=quote or DEFAULT_QUOTE,
            indent=indent,
            continuation_indent=continuation_indent,
            leading_newline=leading_newline,
            trailing_newline=trailing_newline,
            fixable=fixable,
            skip=skip,
        )

    def _coerce_to_string(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

        if isinstance(node, ast.JoinedStr):
            # f-strings introduce runtime expressions; skip linting these snippets.
            return None

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._coerce_to_string(node.left)
            right = self._coerce_to_string(node.right)
            if left is not None and right is not None:
                return left + right
            return None

        return None


def _split_prefix_and_quote(literal: str) -> tuple[str, str]:
    match = STRING_PREFIX_RE.match(literal)
    if not match:
        return "", DEFAULT_QUOTE
    prefix, quote = match.groups()
    return prefix or "", quote or DEFAULT_QUOTE


def _count_leading_newlines(value: str) -> int:
    count = 0
    for char in value:
        if char == "\n":
            count += 1
        elif char.isspace():
            continue
        else:
            break
    return count


def _build_continuation_indent(col_offset: int) -> str:
    return " " * max(col_offset, 0)


def _infer_indent(raw_sql: str) -> str:
    for line in raw_sql.splitlines():
        stripped = line.lstrip()
        if stripped:
            return line[: len(line) - len(stripped)]
    return ""


def extract_sql_from_file(path: Path) -> list[ExtractedSql]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - surfaces as pre-commit failure
        sys.stderr.write(f"Failed to parse {path}: {exc}\n")
        return []

    extractor = SparkSqlExtractor(source, path)
    extractor.visit(tree)
    return extractor.results()


def lint_snippets(snippets: list[ExtractedSql]) -> int:
    actionable = [snippet for snippet in snippets if not snippet.skip]

    for skipped in (snippet for snippet in snippets if snippet.skip):
        sys.stderr.write(
            f"Skipping sqlfluff lint for {skipped.file_path}:{skipped.base_line} due to inline 'sqlfluff:' directive.\n"
        )

    if not actionable:
        return 0

    with TemporarySqlFiles(actionable) as mapping:
        file_paths = list(mapping.keys())
        cmd = [
            "sqlfluff",
            "lint",
            "--disable-progress-bar",
            "--format",
            "json",
        ]

        config = _resolve_sqlfluff_config()
        if config is not None:
            cmd.extend(["--config", str(config)])

        cmd.extend(file_paths)

        process = subprocess.run(cmd, capture_output=True, text=True)  # noqa: PLW1510

        if process.returncode == 0:
            return 0

        try:
            payload = json.loads(process.stdout or "[]")
        except json.JSONDecodeError:
            sys.stderr.write(process.stdout)
            sys.stderr.write(process.stderr)
            return process.returncode

        output_messages: list[str] = []
        for entry in payload:
            temp_path = entry.get("filepath")
            violations = entry.get("violations", [])
            snippet = mapping.get(temp_path)
            if not snippet or not violations:
                continue

            for violation in violations:
                line_no = violation.get("start_line_no") or violation.get("line_no")
                col = violation.get("start_line_pos") or violation.get("line_pos") or 1
                code = violation.get("code", "")
                description = violation.get("description", "")

                if line_no is None:
                    continue

                origin_line = snippet.base_line + (line_no - 1)
                message = f"{snippet.file_path}:{origin_line}:{col} {code} {description}".rstrip()
                output_messages.append(message)

        if not output_messages:
            sys.stderr.write(process.stdout)
            sys.stderr.write(process.stderr)
            return process.returncode

        sys.stderr.write("\n".join(sorted(set(output_messages))) + "\n")
        return 1


def fix_snippets(snippets: list[ExtractedSql]) -> int:
    actionable = [snippet for snippet in snippets if not snippet.skip]

    for skipped in (snippet for snippet in snippets if snippet.skip):
        sys.stderr.write(
            f"Skipping sqlfluff fix for {skipped.file_path}:{skipped.base_line} due to inline 'sqlfluff:' directive.\n"
        )

    if not actionable:
        return 0

    target_files = sorted({snippet.file_path for snippet in actionable})

    with TemporarySqlFiles(actionable) as mapping:
        file_paths = list(mapping.keys())
        cmd = [
            "sqlfluff",
            "fix",
            "--disable-progress-bar",
        ]

        config = _resolve_sqlfluff_config()
        if config is not None:
            cmd.extend(["--config", str(config)])

        cmd.extend(file_paths)

        process = subprocess.run(cmd, capture_output=True, text=True)  # noqa: PLW1510
        if process.returncode not in {0, 1}:
            sys.stderr.write(process.stdout)
            sys.stderr.write(process.stderr)
            return process.returncode

        if process.stdout:
            sys.stderr.write(process.stdout)
        if process.stderr:
            sys.stderr.write(process.stderr)

        replacements: dict[Path, list[tuple[ExtractedSql, str]]] = {}
        skipped: list[ExtractedSql] = []

        for temp_path, snippet in mapping.items():
            temp_file = Path(temp_path)
            fixed_sql = temp_file.read_text(encoding="utf-8").rstrip("\n")

            if fixed_sql == snippet.sql:
                continue

            if not snippet.fixable:
                skipped.append(snippet)
                continue

            new_literal = snippet.build_literal(fixed_sql)
            replacements.setdefault(snippet.file_path, []).append(
                (snippet, new_literal)
            )

        if skipped:
            for snippet in skipped:
                sys.stderr.write(
                    f"Skipping fix for {snippet.file_path}:{snippet.base_line} (unsupported literal form)\n"
                )

        _apply_replacements(replacements)

    refreshed_snippets: list[ExtractedSql] = []
    for path in target_files:
        refreshed_snippets.extend(extract_sql_from_file(path))

    return lint_snippets(refreshed_snippets)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--fix", action="store_true", help="Apply sqlfluff fixes to spark.sql strings"
    )
    args = parser.parse_args(argv)

    python_files = [Path(path) for path in args.files if path.endswith(".py")]

    snippets: list[ExtractedSql] = []
    for file_path in python_files:
        snippets.extend(extract_sql_from_file(file_path))

    if args.fix:
        return fix_snippets(snippets)

    return lint_snippets(snippets)


def _resolve_sqlfluff_config() -> Path | None:
    for candidate in (Path("pyproject.toml"), Path(".sqlfluff"), Path("setup.cfg")):
        if candidate.exists():
            return candidate
    return None


def _apply_replacements(
    replacements: dict[Path, list[tuple[ExtractedSql, str]]],
) -> None:
    for path, entries in replacements.items():
        try:
            original_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue

        line_offsets = _compute_line_offsets(original_text)
        edits: list[tuple[int, int, str]] = []

        for snippet, new_literal in entries:
            start = _offset_from_line_col(
                line_offsets, snippet.lineno, snippet.col_offset
            )
            if snippet.end_lineno is not None and snippet.end_col_offset is not None:
                end = _offset_from_line_col(
                    line_offsets, snippet.end_lineno, snippet.end_col_offset
                )
            else:
                end = start + len(snippet.literal)
            edits.append((start, end, new_literal))

        edits.sort(key=lambda item: item[0], reverse=True)

        updated_text = original_text
        for start, end, replacement in edits:
            updated_text = f"{updated_text[:start]}{replacement}{updated_text[end:]}"

        path.write_text(updated_text, encoding="utf-8")


def _compute_line_offsets(text: str) -> list[int]:
    offsets = [0]
    running_total = 0
    for line in text.splitlines(keepends=True):
        running_total += len(line)
        offsets.append(running_total)
    return offsets


def _offset_from_line_col(offsets: list[int], lineno: int, col: int) -> int:
    lineno_index = max(lineno - 1, 0)
    if lineno_index >= len(offsets):
        return (len(offsets) and offsets[-1]) or 0
    return offsets[lineno_index] + col


class TemporarySqlFiles:
    """Context manager that writes SQL snippets to temporary files."""

    def __init__(self, snippets: list[ExtractedSql]):
        self._snippets = snippets
        self._temp_dir: Path | None = None
        self._mapping: dict[str, ExtractedSql] = {}

    def __enter__(self) -> dict[str, ExtractedSql]:
        from tempfile import TemporaryDirectory

        temp_dir_obj = TemporaryDirectory(prefix="spark_sql_snippets_")
        self._temp_dir = Path(temp_dir_obj.name)
        self._temp_dir_obj = temp_dir_obj

        for index, snippet in enumerate(self._snippets, start=1):
            safe_name = "_".join(snippet.file_path.parts)
            temp_file = self._temp_dir / f"{safe_name}_{index}.sql"
            temp_file.write_text(snippet.temp_content(), encoding="utf-8")
            self._mapping[str(temp_file)] = snippet

        return self._mapping

    def __exit__(self, exc_type, exc, tb) -> None:
        if hasattr(self, "_temp_dir_obj"):
            self._temp_dir_obj.cleanup()


if __name__ == "__main__":
    sys.exit(main())

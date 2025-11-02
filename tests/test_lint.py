"""Unit tests for sqlfluff_pyspark.lint module."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import pytest

from sqlfluff_pyspark.lint import (
    SparkSqlExtractor,
    extract_sql_from_file,
    main,
)


def _make_py(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "example.py"
    path.write_text(dedent(content), encoding="utf-8")
    return path


def test_extract_simple_sql(tmp_path: Path):
    path = _make_py(
        tmp_path,
        """
        spark.sql("SELECT 1")
        """,
    )
    snippets = extract_sql_from_file(path)
    assert len(snippets) == 1
    snippet = snippets[0]
    assert snippet.sql == "SELECT 1"
    assert snippet.file_path == path
    assert snippet.fixable is True


def test_skip_fstring(tmp_path: Path):
    path = _make_py(
        tmp_path,
        """
        value = 2
        spark.sql(f"SELECT {value}")
        """,
    )
    snippets = extract_sql_from_file(path)
    # f-string should not be linted/extracted
    assert snippets == []


def test_concatenated_strings(tmp_path: Path):
    path = _make_py(
        tmp_path,
        """
        spark.sql("SELECT" + " 1")
        """,
    )
    snippets = extract_sql_from_file(path)
    assert len(snippets) == 1
    assert snippets[0].sql == "SELECT 1"


def test_skip_inline_directive(tmp_path: Path):
    path = _make_py(
        tmp_path,
        """
        spark.sql("SELECT 1 -- sqlfluff: disable")
        """,
    )
    snippets = extract_sql_from_file(path)
    assert len(snippets) == 1
    assert snippets[0].skip is True


@pytest.mark.parametrize(
    "raw, expected_indent",
    [
        ("SELECT 1", ""),
        ("  SELECT\n    1", "  "),
    ],
)
def test_infer_indent(raw, expected_indent):
    # Use internal function via constructing ExtractedSql through extractor
    source = f"spark.sql('''\n{raw}\n''')"
    path = Path("dummy.py")
    tree_path = path
    extractor = SparkSqlExtractor(source, tree_path)
    import ast

    tree = ast.parse(source)
    extractor.visit(tree)
    snippets = extractor.results()
    assert snippets
    assert snippets[0].indent == expected_indent


def test_cli_no_files(monkeypatch):
    # Expect 0 exit code when no python files provided
    monkeypatch.setattr(sys, "argv", ["sqlfluff-pyspark"])  # simulate invocation
    assert main([]) == 0

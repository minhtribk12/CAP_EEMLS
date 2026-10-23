from __future__ import annotations

from eemls.metrics_database.schema import COLUMN, CONSTRAINTS, TABLES

# The PostgreSQL types this schema is allowed to use.
TYPES: frozenset[str] = frozenset(
    {
        "text",
        "integer",
        "boolean",
        "jsonb",
        "timestamptz",
        "double precision",
    }
)


def declared_types() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for table, ddl in TABLES.items():
        for line in ddl.splitlines():
            match = COLUMN.match(line)
            if not match or match.group(1) in CONSTRAINTS:
                continue
            rest = line.strip()[len(match.group(1)) :].strip().rstrip(",")
            kind = rest.split(" NOT NULL")[0].strip()
            found.append((table, match.group(1), kind))
    return found


def test_every_column_declares_a_type_the_database_knows() -> None:
    wrong = [
        f"{table}.{column} is {kind!r}"
        for table, column, kind in declared_types()
        if kind not in TYPES
    ]

    assert not wrong, f"the schema would not apply: {wrong}"


def test_the_schema_declares_every_table_the_store_writes() -> None:
    assert declared_types(), "the schema declares no column at all"

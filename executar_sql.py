"""
Executa uma instrução SQL diretamente no banco SQLite do CAWM.

Uso:
    .venv/bin/python executar_sql.py "SELECT nome FROM bacias ORDER BY nome"
    .venv/bin/python executar_sql.py --sql-file consulta.sql
    .venv/bin/python executar_sql.py --db meu_banco.db "SELECT * FROM bacias"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent))

from db_init import initialize_db


def _format_value(value):
    if value is None:
        return "NULL"
    return value


def executar_sql(sql: str, db_path: str | None = None) -> int:
    engine, _session_factory = initialize_db(db_path)
    try:
        with engine.begin() as connection:
            result = connection.execute(text(sql))

            if result.returns_rows:
                rows = result.fetchall()
                headers = list(result.keys())
                print(" | ".join(headers))
                print("-" * max(1, len(" | ".join(headers))))
                for row in rows:
                    print(" | ".join(str(_format_value(row._mapping[h])) for h in headers))
                print(f"\n{len(rows)} linha(s)")
            else:
                print(f"OK: {result.rowcount if result.rowcount is not None else 0} linha(s) afetada(s)")
        return 0
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa SQL no banco SQLite do CAWM")
    parser.add_argument("sql", nargs="?", help="Instrução SQL a executar")
    parser.add_argument("--sql-file", help="Arquivo .sql com a instrução a executar")
    parser.add_argument("--db", help="Caminho do banco SQLite a usar")
    args = parser.parse_args()

    if not args.sql and not args.sql_file:
        parser.error("Informe a instrução SQL como argumento ou use --sql-file")

    if args.sql_file:
        sql_path = Path(args.sql_file)
        if not sql_path.exists():
            print(f"Erro: arquivo SQL não encontrado: {sql_path}", file=sys.stderr)
            return 1
        sql = sql_path.read_text(encoding="utf-8")
    else:
        sql = args.sql

    if not sql or not sql.strip():
        print("Erro: SQL vazio", file=sys.stderr)
        return 1

    return executar_sql(sql.strip(), args.db)


if __name__ == "__main__":
    raise SystemExit(main())

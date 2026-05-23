"""
Executa todas as seeds do projeto na ordem correta.

Ordem:
1. bacias
2. períodos de calibração/validação
3. resultados de desempenho
4. séries hidrológicas (evaporação, precipitação e vazão)

Por padrão, recria o banco do zero removendo o arquivo SQLite atual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from db_init import DEFAULT_DB

import seed_bacias
import seed_periodos
import seed_resultados
import seed_hydrologic_series
import seed_poco_do_fumo


def remove_database(db_path: str) -> None:
    path = Path(db_path)
    if path.exists():
        path.unlink()


def run_master(reset: bool = True, db_path: str | None = None) -> None:
    target_db = db_path or DEFAULT_DB

    if reset:
        remove_database(target_db)

    print("=" * 70)
    print("SEED MASTER - RECRIANDO BANCO E DADOS")
    print("=" * 70)

    print("\n1) Seed de bacias...")
    seed_bacias.popular_bacias(target_db)

    print("\n2) Seed de períodos...")
    periodos_path = Path("Tabela 2: Períodos de calibração e validação.csv")
    periodos_result = seed_periodos.import_periods(periodos_path, target_db)
    print(f"   Criados: {periodos_result['created']}")
    print(f"   Ignorados por ausência de bacia: {len(periodos_result['missing'])}")

    print("\n3) Seed de resultados...")
    resultados_path = Path("Tabela 3: Parâmetros do modelo e coeficientes de desempenho_cleaned.csv")
    resultados_result = seed_resultados.import_results(resultados_path, target_db)
    print(f"   Criados: {resultados_result['created']}")
    print(f"   Sem período correspondente: {len(resultados_result['missing'])}")

    print("\n4) Seed das séries hidrológicas...")
    series_result = seed_hydrologic_series.import_hydrologic_series("Dados", target_db)
    print(f"   Criados: {series_result['created']}")
    print(f"   Atualizados: {series_result['updated']}")
    print(f"   Bacias sem cadastro correspondente: {len(series_result['missing'])}")
    print(f"   Arquivos inválidos: {len(series_result['invalid'])}")

    print("\n5) Seed específico de Poço do Fumo...")
    poco_result = seed_poco_do_fumo.seed_poco_do_fumo(target_db)
    print(f"   Criados: {poco_result['created']}")
    print(f"   Atualizados: {poco_result['updated']}")
    if poco_result["missing_params"]:
        print(f"   Parâmetros ausentes no XLSM: {', '.join(poco_result['missing_params'])}")

    print("\n" + "=" * 70)
    print("SEED MASTER FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa todas as seeds na ordem correta")
    parser.add_argument("--no-reset", action="store_true", help="Não remove o banco SQLite antes de rodar")
    parser.add_argument("--db", default=None, help="Caminho opcional para o banco SQLite")
    args = parser.parse_args()

    run_master(reset=not args.no_reset, db_path=args.db)

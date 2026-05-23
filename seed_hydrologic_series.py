"""
Importa evaporação, precipitação e vazão para tabelas normalizadas.

Estrutura final:
- evaporation_monthly: 12 linhas por bacia, uma por mês
- precipitation_daily: 1 linha por bacia/dia
- flow_daily: 1 linha por bacia/dia
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from db_init import initialize_db
from db_models import Bacia, EvaporationMonthly, PrecipitationDaily, FlowDaily


SUFIXO_REDUZIDO = " REDUZIDO"


def normalize_name(value: str) -> str:
    if not isinstance(value, str):
        return str(value).upper().strip()
    normalized = ''.join(
        character for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )
    return normalized.upper().strip()


def is_reduced_basin(name: str) -> bool:
    return normalize_name(name).endswith(SUFIXO_REDUZIDO)


def find_value_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        lowered = str(column).lower()
        if lowered not in {"data", "date", "dia"}:
            return column
    return frame.columns[0]


def read_monthly_evaporation(csv_path: Path) -> list[dict]:
    frame = pd.read_csv(csv_path, header=None)
    if frame.shape[1] < 2:
        raise ValueError(f"Evaporação inválida em {csv_path}")

    months = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    clean = []
    for month, value in zip(months, values):
        if pd.isna(month) or pd.isna(value):
            continue
        clean.append({"mes": int(month), "valor": round(float(value), 3)})
    return clean


def read_daily_series(csv_path: Path) -> list[dict]:
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 2:
        raise ValueError(f"Série diária inválida em {csv_path}")

    date_column = frame.columns[0]
    value_column = find_value_column(frame)

    dates = pd.to_datetime(frame[date_column], dayfirst=False, errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")

    series = []
    for date_value, numeric_value in zip(dates, values):
        if pd.isna(date_value) or pd.isna(numeric_value):
            continue
        series.append(
            {
                "data": date_value.date(),
                "valor": round(float(numeric_value), 3),
            }
        )

    return series


def build_bacia_directory_map(root: Path) -> dict[str, dict[str, Path]]:
    directory_map: dict[str, dict[str, Path]] = {}
    for region_dir in root.iterdir():
        if not region_dir.is_dir():
            continue
        for basin_dir in region_dir.iterdir():
            if not basin_dir.is_dir() or is_reduced_basin(basin_dir.name):
                continue
            evap = basin_dir / "evaporacao.csv"
            precip = basin_dir / "precipitacao.csv"
            pao = basin_dir / "pao.csv"
            if evap.exists() and precip.exists() and pao.exists():
                directory_map[normalize_name(basin_dir.name)] = {
                    "path": basin_dir,
                    "evaporacao": evap,
                    "precipitacao": precip,
                    "pao": pao,
                }
    return directory_map


def upsert_monthly_evaporation(session, bacia_id: int, evaporation: list[dict]) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for item in evaporation:
        existing = session.query(EvaporationMonthly).filter_by(bacia_id=bacia_id, mes=item["mes"]).first()
        if existing:
            existing.valor = item["valor"]
            updated += 1
        else:
            session.add(EvaporationMonthly(bacia_id=bacia_id, mes=item["mes"], valor=item["valor"]))
            inserted += 1
    return inserted, updated


def upsert_daily_series(session, model, bacia_id: int, rows: list[dict]) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for item in rows:
        existing = session.query(model).filter_by(bacia_id=bacia_id, data=item["data"]).first()
        if existing:
            existing.valor = item["valor"]
            updated += 1
        else:
            session.add(model(bacia_id=bacia_id, data=item["data"], valor=item["valor"]))
            inserted += 1
    return inserted, updated


def import_hydrologic_series(pasta_raiz: str = "Dados", db_path: str | None = None):
    engine, Session = initialize_db(db_path)
    session = Session()

    created = 0
    updated = 0
    missing = []
    invalid = []

    try:
        bacias = session.query(Bacia).all()
        bacias_por_nome = {normalize_name(bacia.nome): bacia for bacia in bacias}

        directory_map = build_bacia_directory_map(Path(pasta_raiz))

        for normalized_name, files in sorted(directory_map.items()):
            bacia = bacias_por_nome.get(normalized_name)
            if not bacia:
                missing.append(files["path"].name)
                continue

            try:
                evaporacao = read_monthly_evaporation(files["evaporacao"])
                precipitacao = read_daily_series(files["precipitacao"])
                vazao = read_daily_series(files["pao"])
            except Exception as exc:
                invalid.append((files["path"].name, str(exc)))
                continue

            if len(evaporacao) != 12:
                invalid.append((files["path"].name, f"evaporação com {len(evaporacao)} valores (esperado 12)"))
                continue

            inserted, refreshed = upsert_monthly_evaporation(session, bacia.id, evaporacao)
            created += inserted
            updated += refreshed

            inserted, refreshed = upsert_daily_series(session, PrecipitationDaily, bacia.id, precipitacao)
            created += inserted
            updated += refreshed

            inserted, refreshed = upsert_daily_series(session, FlowDaily, bacia.id, vazao)
            created += inserted
            updated += refreshed

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()

    return {
        "created": created,
        "updated": updated,
        "missing": missing,
        "invalid": invalid,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Importa séries hidrológicas únicas por bacia")
    parser.add_argument("--root", default="Dados", help="Diretório raiz com as bacias")
    parser.add_argument("--db", default=None, help="Caminho opcional para o banco SQLite")
    args = parser.parse_args()

    result = import_hydrologic_series(args.root, args.db)
    print(f"Criados: {result['created']}")
    print(f"Atualizados: {result['updated']}")
    if result['missing']:
        print(f"\nBacias sem cadastro correspondente: {len(result['missing'])}")
        for name in result['missing']:
            print(f" - {name}")
    if result['invalid']:
        print(f"\nArquivos inválidos: {len(result['invalid'])}")
        for name, error in result['invalid']:
            print(f" - {name}: {error}")

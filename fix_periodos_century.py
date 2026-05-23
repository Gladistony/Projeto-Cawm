"""
Corrige anos de século incorreto na tabela calibration_periods.

Regra:
- para campos de data (calib_start, calib_end, val_start, val_end)
- se ano > (ano_atual + 1), subtrai 100 anos

Exemplo: 2067-04-01 -> 1967-04-01
"""
from __future__ import annotations

from datetime import date
from typing import Tuple

from db_init import initialize_db
from db_models import CalibrationPeriod


def _corrigir_data(d: date | None, ano_limite: int) -> tuple[date | None, bool]:
    if d is None:
        return None, False
    if d.year <= ano_limite:
        return d, False

    novo_ano = d.year - 100
    dia = d.day
    while dia >= 1:
        try:
            return date(novo_ano, d.month, dia), True
        except ValueError:
            dia -= 1

    # fallback extremo (não esperado)
    return date(novo_ano, d.month, 1), True


def corrigir_periodos(db_path: str | None = None, dry_run: bool = False) -> Tuple[int, int]:
    engine, Session = initialize_db(db_path)
    session = Session()

    ano_limite = date.today().year + 1
    total_registros = 0
    total_alteracoes = 0

    try:
        periodos = session.query(CalibrationPeriod).all()
        total_registros = len(periodos)

        for p in periodos:
            alterou = False

            novo_calib_start, ch = _corrigir_data(p.calib_start, ano_limite)
            alterou = alterou or ch
            if ch:
                p.calib_start = novo_calib_start

            novo_calib_end, ch = _corrigir_data(p.calib_end, ano_limite)
            alterou = alterou or ch
            if ch:
                p.calib_end = novo_calib_end

            novo_val_start, ch = _corrigir_data(p.val_start, ano_limite)
            alterou = alterou or ch
            if ch:
                p.val_start = novo_val_start

            novo_val_end, ch = _corrigir_data(p.val_end, ano_limite)
            alterou = alterou or ch
            if ch:
                p.val_end = novo_val_end

            if alterou:
                total_alteracoes += 1

        if dry_run:
            session.rollback()
        else:
            session.commit()

        return total_registros, total_alteracoes
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corrige anos de século incorreto em calibration_periods")
    parser.add_argument("--db", default=None, help="Caminho opcional do banco SQLite")
    parser.add_argument("--dry-run", action="store_true", help="Somente simula, sem salvar")
    args = parser.parse_args()

    total, alterados = corrigir_periodos(db_path=args.db, dry_run=args.dry_run)
    modo = "SIMULAÇÃO" if args.dry_run else "APLICADO"
    print(f"[{modo}] Registros analisados: {total}")
    print(f"[{modo}] Registros corrigidos: {alterados}")

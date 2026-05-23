"""
Importa os períodos de calibração/validação do CSV para o banco.

Assume formato semicolon-separated. As colunas no arquivo são:
- coluna 0: região (ex: CAPIBARIBE, PAJEÚ)
- coluna 1: nome da bacia / estação (pode conter sufixos como (c.), (r.), (c. r.))
- coluna 2: período de calibração (ex: '01/04/67 a 12/01/85')
- coluna 3: período de validação

Cria registros em `calibration_periods` vinculados por nome da bacia
(com mapeamento explícito e matching fuzzy). Informa bacias não encontradas.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches
import pandas as pd

from db_init import initialize_db
from db_models import CalibrationPeriod, Bacia

DATE_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{2,4})")


def parse_period(period_str: str):
    """Extrai até dois pares de datas (start,end) de uma string.
    Retorna (start_date, end_date) como datetime.date ou (None,None).
    """
    if not isinstance(period_str, str) or not period_str.strip():
        return None, None

    found = DATE_RE.findall(period_str)
    if len(found) < 2:
        return None, None

    # take first two
    def to_date(s):
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
        return None

    d1 = to_date(found[0])
    d2 = to_date(found[1])
    return d1, d2


def clean_name_and_method(raw_name: str):
    """Remove sufixos entre parênteses e retorna (clean_name, method).
    method: 'c', 'r', 'c.r' or 'normal'
    """
    if not isinstance(raw_name, str):
        return raw_name, 'normal'
    name = raw_name.strip()
    method = 'normal'

    # find parentheses content
    m = re.search(r"\(([^)]*)\)", name)
    if m:
        content = m.group(1).lower()
        if 'c' in content and 'r' in content:
            method = 'c.r'
        elif 'c' in content:
            method = 'c'
        elif 'r' in content:
            method = 'r'
        # remove the parentheses part
        name = re.sub(r"\s*\([^)]*\)", "", name).strip()

    return name, method


def normalize_name(s: str) -> str:
    """Remove acentos e converte para maiúscula."""
    if not isinstance(s, str):
        return str(s).upper()
    # Remove acentos
    s_norm = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    return s_norm.upper().strip()


# Mapeamento explícito: nomes normalizados do CSV → nomes exatos no banco
EXPLICIT_MAPPING = {
    'ENGENHO SITIO': 'ENGENHO SÍTIO',
    'LIMOEIRO': 'LIMOEIRO',
    'PAUDALHO': 'PAUDALHO',
    'S LOURENCO': 'S LOURENCO',
    'SALGADINHO': 'SALGADINHO',
    'SANTA CRUZ': 'SANTA CRUZ',
    'S. L. DA MATA II': 'S LOURENCO',  # Equivalente a S LOURENCO
    'TORITAMA': 'TORITAMA',
    'VITÓRIA DE S.A.': 'VITORIA',
    'AFOGADOS': 'AFOGADOS',
    'FLORES': 'FLORES',
    'FLORESTA': 'FLORESTA',
    'ILHA GRANDE': 'ILHA GRANDE',
    'SERRA TALHADA': 'SERRA TALHADA',
    'POÇO DO FUMO': None,  # Não existe no banco (ignorar)
    'ARACAIBA': 'Aracoiaba',
    'CHOROZINHO': 'Chorozinho',
    'MORADA NOVA': 'Morada Nova',
    'QUIXERAMOBIM': 'Quixeramobim',
    'SENADOR POMPEU': 'Senador Pompeu',
    'PARACURU': 'Paracuru',
}


def find_bacia_fuzzy(name: str, session, threshold: float = 0.6):
    """Tenta encontrar a bacia por:
    1. Mapeamento explícito
    2. Busca exata normalizada
    3. Fuzzy matching
    """
    name_norm = normalize_name(name)
    
    # 1. Tenta mapeamento explícito
    if name_norm in EXPLICIT_MAPPING:
        mapped_name = EXPLICIT_MAPPING[name_norm]
        if mapped_name is None:
            # Explicitamente marcado como "não existe"
            return None
        # Busca pela nome mapeado
        b = session.query(Bacia).filter_by(nome=mapped_name).first()
        if b:
            return b
    
    # 2. Depois tenta exato normalizado (case-insensitive)
    all_bacias = session.query(Bacia).all()
    for b in all_bacias:
        if normalize_name(b.nome) == name_norm:
            return b
    
    # 3. Por fim, tenta fuzzy matching
    all_names_norm = [(b.id, b.nome, normalize_name(b.nome)) for b in all_bacias]
    matches = get_close_matches(name_norm, [n[2] for n in all_names_norm], n=1, cutoff=threshold)
    if matches:
        for bid, orig_name, norm_name in all_names_norm:
            if norm_name == matches[0]:
                return session.query(Bacia).filter_by(id=bid).first()
    
    return None


def import_periods(csv_path: Path, db_path: str | None = None):
    df = pd.read_csv(csv_path, sep=';', header=None, dtype=str, keep_default_na=False)

    engine, Session = initialize_db(db_path)
    session = Session()

    created = 0
    updated = 0
    missing_bacias = []
    errors = []

    try:
        for idx, row in df.iterrows():
            region = (row[0] or '').strip()
            raw_name = (row[1] or '').strip()
            calib_raw = (row[2] or '').strip() if len(row) > 2 else ''
            val_raw = (row[3] or '').strip() if len(row) > 3 else ''

            # Skip header row
            if region.lower() == 'bacia' or raw_name.lower() == 'estação fluviométrica':
                continue

            name, method = clean_name_and_method(raw_name)

            # lookup bacia com mapeamento explícito + fuzzy matching
            b = find_bacia_fuzzy(name, session)
            if not b:
                missing_bacias.append((name, raw_name, region))
                continue

            calib_start, calib_end = parse_period(calib_raw)
            val_start, val_end = parse_period(val_raw)

            cp = CalibrationPeriod(
                bacia_id=b.id,
                station=raw_name,
                method=method,
                calib_start=calib_start,
                calib_end=calib_end,
                val_start=val_start,
                val_end=val_end,
            )
            session.add(cp)
            created += 1

        session.commit()
    except Exception as e:
        session.rollback()
        errors.append(str(e))
    finally:
        session.close()
        engine.dispose()

    return {'created': created, 'missing': missing_bacias, 'errors': errors}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Importa períodos de calibração/validação')
    parser.add_argument('csv', help='Arquivo CSV (separador ";")')
    parser.add_argument('--db', default=None, help='Caminho opcional para o banco SQLite')
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print('Arquivo não encontrado:', path)
        raise SystemExit(1)

    result = import_periods(path, args.db)
    print('Criados:', result['created'])
    if result['missing']:
        print('\nBacias não encontradas (nome limpo, raw, regiao):')
        for t in result['missing']:
            print(' -', t)
    if result['errors']:
        print('\nErros:')
        for e in result['errors']:
            print(' -', e)

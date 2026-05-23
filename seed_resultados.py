"""
Importa os resultados de desempenho do modelo (Tabela 3) para o banco de dados.

Vincula cada resultado a seu período de calibração/validação correspondente
via correspondência de estação (Flu. Station na Tabela 3 = station no CalibrationPeriod).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
import pandas as pd

from db_init import initialize_db
from db_models import ModelResult, CalibrationPeriod


def normalize_name(s: str) -> str:
    """Remove acentos e converte para maiúscula."""
    if not isinstance(s, str):
        return str(s).upper()
    s_norm = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    return s_norm.upper().strip()


def extract_method_from_station(station_name: str) -> str:
    """Extrai o método de um nome como 'Limoeiro (c.r.)'."""
    if not isinstance(station_name, str):
        return 'normal'
    
    station_name = station_name.strip()
    method = 'normal'
    
    # Procura por sufixo entre parênteses: (c.), (r.), (c.r.), (c. r.)
    m = re.search(r'\(([^)]*)\)', station_name)
    if m:
        content = m.group(1).lower().replace(' ', '')
        if 'c' in content and 'r' in content:
            method = 'c.r'
        elif 'c' in content:
            method = 'c'
        elif 'r' in content:
            method = 'r'
    
    return method


def find_period(station_name_from_csv: str, session):
    """
    Encontra um período calibração que corresponde à estação da Tabela 3.
    
    Estratégia:
    1. Extrai o método do nome da estação (ex: "Limoeiro (c.)" -> "Limoeiro", "c")
    2. Busca um CalibrationPeriod com station que corresponda (após normalização)
       e método exato.
    3. Se não encontrar, tenta sem method (busca apenas por estação).
    """
    # Remove acentos e passa para maiúsculas para comparação
    station_norm = normalize_name(station_name_from_csv)
    method_csv = extract_method_from_station(station_name_from_csv)
    
    # Limpa o nome removendo parênteses e tudo que está dentro
    clean_name = re.sub(r'\s*\([^)]*\)', '', station_name_from_csv).strip()
    clean_name_norm = normalize_name(clean_name)
    
    # Busca todos os períodos
    all_periods = session.query(CalibrationPeriod).all()
    
    # Primeira tentativa: busca por método exato + estação normalizada
    for period in all_periods:
        if period.station:
            period_station_norm = normalize_name(period.station)
            if period_station_norm == station_norm and period.method == method_csv:
                return period
    
    # Segunda tentativa: busca por estação normalizada sem verificar parênteses
    for period in all_periods:
        if period.station:
            # Remove parênteses do period.station também
            clean_period_station = re.sub(r'\s*\([^)]*\)', '', period.station).strip()
            clean_period_norm = normalize_name(clean_period_station)
            if clean_period_norm == clean_name_norm and period.method == method_csv:
                return period
    
    # Terceira tentativa: busca apenas por estação (ignora método)
    for period in all_periods:
        if period.station:
            period_station_norm = normalize_name(period.station)
            if period_station_norm == station_norm:
                return period
    
    return None


def import_results(csv_path: Path):
    """Importa resultados de desempenho do CSV para o banco."""
    df = pd.read_csv(csv_path)
    
    engine, Session = initialize_db()
    session = Session()
    
    created = 0
    missing_periods = []
    errors = []
    
    try:
        for idx, row in df.iterrows():
            bacia_name = (row['Bacia'] or '').strip()
            station_name = (row['Flu. Station'] or '').strip()
            
            # Skip if no station name
            if not station_name:
                continue
            
            # Procura o período calibração
            period = find_period(station_name, session)
            if not period:
                missing_periods.append((bacia_name, station_name))
                continue
            
            # Verifica se já há resultado para este período
            existing = session.query(ModelResult).filter_by(
                calibration_period_id=period.id
            ).first()
            if existing:
                # Já existe resultado para este período, pula
                continue
            
            # Cria o resultado
            result = ModelResult(
                calibration_period_id=period.id,
                area_km2=float(row['Area (km²)']) if pd.notna(row['Area (km²)']) else None,
                s_mm=float(row['S (mm)']) if pd.notna(row['S (mm)']) else None,
                ks=float(row['Ks']) if pd.notna(row['Ks']) else None,
                nse_calib=float(row['NSE Cal.']) if pd.notna(row['NSE Cal.']) else None,
                nse_sqrt_calib=float(row['NSEsqrt Cal.']) if pd.notna(row['NSEsqrt Cal.']) else None,
                nse_log_calib=float(row['NSElog Cal.']) if pd.notna(row['NSElog Cal.']) else None,
                pbias_calib=float(row['Pbias (%) Cal.']) if pd.notna(row['Pbias (%) Cal.']) else None,
                nse_val=float(row['NSE Val.']) if pd.notna(row['NSE Val.']) else None,
                nse_sqrt_val=float(row['NSEsqrt Val.']) if pd.notna(row['NSEsqrt Val.']) else None,
                nse_log_val=float(row['NSElog Val.']) if pd.notna(row['NSElog Val.']) else None,
                pbias_val=float(row['Pbias (%) Val.']) if pd.notna(row['Pbias (%) Val.']) else None,
            )
            session.add(result)
            created += 1
        
        session.commit()
    except Exception as e:
        session.rollback()
        errors.append(str(e))
    finally:
        session.close()
        engine.dispose()
    
    return {'created': created, 'missing': missing_periods, 'errors': errors}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Importa resultados de desempenho do modelo')
    parser.add_argument('csv', help='Arquivo CSV com resultados')
    args = parser.parse_args()
    
    path = Path(args.csv)
    if not path.exists():
        print('Arquivo não encontrado:', path)
        raise SystemExit(1)
    
    result = import_results(path)
    print('Criados:', result['created'])
    if result['missing']:
        print(f'\n{len(result["missing"])} período(s) não encontrado(s):')
        for bacia, station in result['missing'][:10]:
            print(f'  - {bacia}: {station}')
        if len(result['missing']) > 10:
            print(f'  ... e mais {len(result["missing"]) - 10}')
    if result['errors']:
        print('\nErros:')
        for e in result['errors']:
            print(' -', e)

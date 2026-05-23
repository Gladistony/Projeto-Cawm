"""
Inventário inicial do banco CAWM.

Este script conecta ao banco e verifica, por bacia:
- se existem períodos de calibração/validação
- se existem resultados finais de métricas
- se as séries de evaporação, precipitação e vazão existem
- se os parâmetros iniciais da bacia estão preenchidos

Por enquanto não executa calibração nem simulação.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func

from db_init import initialize_db
from db_models import (
    Bacia,
    CalibrationPeriod,
    ModelResult,
    EvaporationMonthly,
    PrecipitationDaily,
    FlowDaily,
)

PARAMETROS_INICIAIS = [
    "k",
    "a",
    "expo_perdas",
    "beta",
    "kg",
    "rio",
    "submax",
    "gmax",
    "reserva_solo_inicial",
    "profundo_inicial",
    "s3_inicial",
    "s1_inicial",
    "s2_inicial",
]


@dataclass
class InventarioBacia:
    nome: str
    regiao: str | None
    periodos: int
    resultados: int
    evap_meses: int
    precipitacao_dias: int
    vazao_dias: int
    parametros_faltando: list[str]

    @property
    def tem_periodos(self) -> bool:
        return self.periodos > 0

    @property
    def tem_resultados(self) -> bool:
        return self.resultados > 0

    @property
    def tem_series_completas(self) -> bool:
        return self.evap_meses == 12 and self.precipitacao_dias > 0 and self.vazao_dias > 0

    @property
    def parametros_completos(self) -> bool:
        return len(self.parametros_faltando) == 0


def normalizar_texto(valor: Any) -> str | None:
    texto = str(valor).strip() if valor is not None else ""
    return texto or None


def obter_parametros_faltando(bacia: Bacia) -> list[str]:
    faltando = []
    for campo in PARAMETROS_INICIAIS:
        if getattr(bacia, campo) is None:
            faltando.append(campo)
    return faltando


def coletar_contagens(session):
    periodos_por_bacia = dict(
        session.query(CalibrationPeriod.bacia_id, func.count(CalibrationPeriod.id))
        .group_by(CalibrationPeriod.bacia_id)
        .all()
    )

    resultados_por_bacia = dict(
        session.query(CalibrationPeriod.bacia_id, func.count(ModelResult.id))
        .outerjoin(ModelResult, CalibrationPeriod.id == ModelResult.calibration_period_id)
        .group_by(CalibrationPeriod.bacia_id)
        .all()
    )

    evap_por_bacia = dict(
        session.query(EvaporationMonthly.bacia_id, func.count(EvaporationMonthly.id))
        .group_by(EvaporationMonthly.bacia_id)
        .all()
    )

    precip_por_bacia = dict(
        session.query(PrecipitationDaily.bacia_id, func.count(PrecipitationDaily.id))
        .group_by(PrecipitationDaily.bacia_id)
        .all()
    )

    vazao_por_bacia = dict(
        session.query(FlowDaily.bacia_id, func.count(FlowDaily.id))
        .group_by(FlowDaily.bacia_id)
        .all()
    )

    return periodos_por_bacia, resultados_por_bacia, evap_por_bacia, precip_por_bacia, vazao_por_bacia


def inventariar_bacias(session) -> list[InventarioBacia]:
    periodos_por_bacia, resultados_por_bacia, evap_por_bacia, precip_por_bacia, vazao_por_bacia = coletar_contagens(session)
    bacias = session.query(Bacia).order_by(Bacia.nome).all()

    inventario: list[InventarioBacia] = []
    for bacia in bacias:
        inventario.append(
            InventarioBacia(
                nome=bacia.nome,
                regiao=normalizar_texto(bacia.regiao),
                periodos=periodos_por_bacia.get(bacia.id, 0),
                resultados=resultados_por_bacia.get(bacia.id, 0),
                evap_meses=evap_por_bacia.get(bacia.id, 0),
                precipitacao_dias=precip_por_bacia.get(bacia.id, 0),
                vazao_dias=vazao_por_bacia.get(bacia.id, 0),
                parametros_faltando=obter_parametros_faltando(bacia),
            )
        )

    return inventario


def imprimir_inventario(inventario: list[InventarioBacia]) -> None:
    total_bacias = len(inventario)
    total_periodos = sum(item.periodos for item in inventario)
    total_resultados = sum(item.resultados for item in inventario)
    bacias_com_series = sum(1 for item in inventario if item.tem_series_completas)
    bacias_com_parametros = sum(1 for item in inventario if item.parametros_completos)

    print("=" * 80)
    print("INVENTÁRIO DO BANCO CAWM")
    print("=" * 80)
    print(f"Total de bacias: {total_bacias}")
    print(f"Total de períodos: {total_periodos}")
    print(f"Total de resultados: {total_resultados}")
    print(f"Bacias com séries completas: {bacias_com_series}")
    print(f"Bacias com parâmetros completos: {bacias_com_parametros}")
    print("-" * 80)

    for item in inventario:
        print(
            f"{item.nome} | regiao={item.regiao or '-'} | "
            f"periodos={item.periodos} | resultados={item.resultados} | "
            f"evap={item.evap_meses} | chuva={item.precipitacao_dias} | vazao={item.vazao_dias} | "
            f"parametros={'OK' if item.parametros_completos else f'FALTAM {len(item.parametros_faltando)}'}"
        )
        if item.parametros_faltando:
            print(f"  - faltando: {', '.join(item.parametros_faltando)}")

    print("=" * 80)


def main() -> None:
    engine, Session = initialize_db()
    session = Session()

    try:
        inventario = inventariar_bacias(session)
        imprimir_inventario(inventario)
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    main()

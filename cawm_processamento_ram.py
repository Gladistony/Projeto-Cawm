"""
Processamento CAWM em memória RAM.

Objetivo deste script:
- carregar as bacias e seus dados diretamente do banco em memória
- respeitar os períodos de calibração/validação vindos do banco
- calibrar Ks/Kl com PSO usando 2000 partículas e 3 interações
- testar dois conjuntos de inicialização:
  1) Ks/Kl na média dos limites e A/Expo nos valores padrão da bacia
  2) Ks/Kl/A/Expo todos na média dos limites
- executar os dois métodos do modelo:
  - estado_inicial
  - spinup
- gerar 4 combinações finais por período
- calcular índices estatísticos em memória
- comparar com os resultados já salvos na tabela de métricas

Nada é salvo em disco e nada é persistido no banco. O script é apenas para teste.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from cawm_config import get_array_backend
from db_init import initialize_db
from db_models import (
    Bacia,
    CalibrationPeriod,
    EvaporationMonthly,
    FlowDaily,
    ModelResult,
    PrecipitationDaily,
)
from funcoes import (
    calcular_indices_estatisticos,
    calibrar_parametros_pso,
)

LIMITES_PADRAO = {
    "ks": (0.0, 1.0),
    "kl": (0.0, 2.0),
    "a": (0.0, 2.0),
    "expo_perdas": (0.5, 3.0),
}

N_PARTICULAS = 5000
MAX_ITER = 100
PACIENCIA = 10
METODOS = ("estado_inicial", "spinup")


@dataclass
class ResultadoMetodo:
    cenario: str
    periodo_id: int
    bacia: str
    station: str
    method: str
    area_km2_usada: float
    fonte_area: str
    fase: str
    nse_calib: float
    nse_val: float
    nse_sqrt_calib: float
    nse_sqrt_val: float
    nse_log_calib: float
    nse_log_val: float
    pbias_calib: float
    pbias_val: float
    nash_calib: float
    nash_val: float
    ks: float
    kl: float
    a: float | None
    expo_perdas: float | None
    referencia_nse_calib: float | None
    referencia_nse_val: float | None
    delta_nse_calib: float | None
    delta_nse_val: float | None


def _mask_from_dates(datas_texto, inicio, fim):
    datas = pd.to_datetime(pd.Series(datas_texto), errors="coerce")
    if datas.isna().all():
        return None
    inicio_dt = pd.Timestamp(inicio)
    fim_dt = pd.Timestamp(fim)
    return (datas >= inicio_dt) & (datas <= fim_dt)


def _corrigir_seculo(data_original: date | None, serie_min: pd.Timestamp, serie_max: pd.Timestamp) -> date | None:
    if data_original is None:
        return None

    ts = pd.Timestamp(data_original)
    if pd.isna(ts):
        return None

    # Corrige datas de dois dígitos que viraram 20xx (ex.: 67 -> 2067)
    if ts.year > (serie_max.year + 1):
        ano_corrigido = ts.year - 100
        if ano_corrigido >= (serie_min.year - 1):
            dia = int(ts.day)
            mes = int(ts.month)
            while dia > 28:
                try:
                    return date(ano_corrigido, mes, dia)
                except ValueError:
                    dia -= 1
            return date(ano_corrigido, mes, dia)

    return ts.date()


def _corrigir_periodo_para_serie(periodo, datas_serie: np.ndarray):
    serie_datas = pd.to_datetime(pd.Series(datas_serie), errors="coerce").dropna()
    if serie_datas.empty:
        return periodo.calib_start, periodo.calib_end, periodo.val_start, periodo.val_end, False

    serie_min = pd.Timestamp(serie_datas.min())
    serie_max = pd.Timestamp(serie_datas.max())

    calib_start = _corrigir_seculo(periodo.calib_start, serie_min, serie_max)
    calib_end = _corrigir_seculo(periodo.calib_end, serie_min, serie_max)
    val_start = _corrigir_seculo(periodo.val_start, serie_min, serie_max)
    val_end = _corrigir_seculo(periodo.val_end, serie_min, serie_max)

    ajustou = any(
        [
            calib_start != periodo.calib_start,
            calib_end != periodo.calib_end,
            val_start != periodo.val_start,
            val_end != periodo.val_end,
        ]
    )
    return calib_start, calib_end, val_start, val_end, ajustou


def _media_limites(nome_parametro: str) -> float:
    minimo, maximo = LIMITES_PADRAO[nome_parametro]
    return (minimo + maximo) / 2.0


def _to_numpy_float_array(valor: Any) -> np.ndarray:
    if hasattr(valor, "get"):
        valor = valor.get()
    return np.asarray(valor, dtype=float)


def _to_float_scalar(valor: Any, padrao: float = np.nan) -> float:
    if valor is None:
        return float(padrao)
    if hasattr(valor, "get"):
        valor = valor.get()
    if hasattr(valor, "item"):
        valor = valor.item()
    return float(valor)


def _carregar_base_referencia(session):
    referencia = {}
    for period in session.query(CalibrationPeriod).all():
        result = session.query(ModelResult).filter_by(calibration_period_id=period.id).first()
        referencia[period.id] = result
    return referencia


def _carregar_periodos_por_bacia(session):
    periodos_por_bacia: dict[int, list[CalibrationPeriod]] = {}
    for periodo in session.query(CalibrationPeriod).order_by(CalibrationPeriod.bacia_id, CalibrationPeriod.id).all():
        periodos_por_bacia.setdefault(periodo.bacia_id, []).append(periodo)
    return periodos_por_bacia


def _normalizar_parametros_bacia(bacia: Bacia) -> dict[str, Any]:
    parametros = bacia.to_dict()
    parametros["beta"] = 2.0 if parametros.get("beta") is None else float(parametros["beta"])
    parametros["kg"] = 1.0 if parametros.get("kg") is None else float(parametros["kg"])
    return parametros


def _carregar_series_do_banco(session, bacia_id: int):
    evap = (
        session.query(EvaporationMonthly)
        .filter_by(bacia_id=bacia_id)
        .order_by(EvaporationMonthly.mes)
        .all()
    )
    chuva = (
        session.query(PrecipitationDaily)
        .filter_by(bacia_id=bacia_id)
        .order_by(PrecipitationDaily.data)
        .all()
    )
    vazao = (
        session.query(FlowDaily)
        .filter_by(bacia_id=bacia_id)
        .order_by(FlowDaily.data)
        .all()
    )

    if not evap or not chuva or not vazao:
        return None

    df_chuva = pd.DataFrame(
        {
            "data": [item.data for item in chuva],
            "chuva": [float(item.valor) for item in chuva],
        }
    )
    df_vazao = pd.DataFrame(
        {
            "data": [item.data for item in vazao],
            "vazao": [float(item.valor) for item in vazao],
        }
    )
    df = pd.merge(df_vazao, df_chuva, on="data", how="inner").sort_values("data")
    if df.empty:
        return None

    evaporacao_por_mes = {int(item.mes): float(item.valor) for item in evap}
    df["mes"] = pd.to_datetime(df["data"], errors="coerce").dt.month
    df["evaporacao"] = df["mes"].map(evaporacao_por_mes)

    if df["evaporacao"].isna().any():
        return None

    return (
        df["vazao"].to_numpy(dtype=float),
        df["chuva"].to_numpy(dtype=float),
        df["evaporacao"].to_numpy(dtype=float),
        df["data"].to_numpy(),
    )


def _resultado_referencia(periodo_id: int, referencia: dict[int, ModelResult | None]):
    result = referencia.get(periodo_id)
    if result is None:
        return {
            "nse_calib": None,
            "nse_val": None,
            "nse_sqrt_calib": None,
            "nse_sqrt_val": None,
            "nse_log_calib": None,
            "nse_log_val": None,
            "pbias_calib": None,
            "pbias_val": None,
        }
    return {
        "nse_calib": float(getattr(result, "nse_calib")) if getattr(result, "nse_calib") is not None else None,
        "nse_val": float(getattr(result, "nse_val")) if getattr(result, "nse_val") is not None else None,
        "nse_sqrt_calib": float(getattr(result, "nse_sqrt_calib")) if getattr(result, "nse_sqrt_calib") is not None else None,
        "nse_sqrt_val": float(getattr(result, "nse_sqrt_val")) if getattr(result, "nse_sqrt_val") is not None else None,
        "nse_log_calib": float(getattr(result, "nse_log_calib")) if getattr(result, "nse_log_calib") is not None else None,
        "nse_log_val": float(getattr(result, "nse_log_val")) if getattr(result, "nse_log_val") is not None else None,
        "pbias_calib": float(getattr(result, "pbias_calib")) if getattr(result, "pbias_calib") is not None else None,
        "pbias_val": float(getattr(result, "pbias_val")) if getattr(result, "pbias_val") is not None else None,
    }


def _comparar_metrica(valor_novo: float, valor_ref: float | None) -> float | None:
    if valor_ref is None:
        return None
    return float(valor_novo - valor_ref)


def _resolver_area_parametros(
    parametros_base: dict[str, Any],
    referencia_periodo: ModelResult | None,
    usar_area_resultado: bool,
) -> tuple[dict[str, Any], float, str]:
    parametros = dict(parametros_base)

    area_bacia = parametros.get("area_km2")
    area_bacia = float(area_bacia) if area_bacia is not None else None

    area_resultado = None
    if referencia_periodo is not None and getattr(referencia_periodo, "area_km2") is not None:
        area_resultado = float(getattr(referencia_periodo, "area_km2"))

    if usar_area_resultado and area_resultado is not None:
        parametros["area_km2"] = area_resultado
        return parametros, area_resultado, "resultado"

    if area_bacia is None:
        raise ValueError("área da bacia ausente")

    parametros["area_km2"] = area_bacia
    return parametros, area_bacia, "bacia"


def _imprimir_inicio_periodo(cenario: str, bacia: str, station: str, method: str, periodo_id: int, area_usada: float, fonte_area: str):
    print("\n" + "-" * 110)
    print(
        f"▶️ [{cenario}] Iniciando período {periodo_id}: {bacia} / {station} / {method} "
        f"| área usada={area_usada:.3f} km² ({fonte_area})"
    )


def _imprimir_resultado_periodo(cenario: str, periodo_id: int, bacia: str, station: str, method: str, resultados_periodo: list[ResultadoMetodo]):
    if not resultados_periodo:
        print(f"⚠️ [{cenario}] {bacia} / {station} / {method} (período {periodo_id}): sem resultados válidos")
        return

    df = pd.DataFrame([r.__dict__ for r in resultados_periodo])
    colunas = [
        "fase",
        "nse_calib",
        "nse_val",
        "nse_sqrt_calib",
        "nse_sqrt_val",
        "nse_log_calib",
        "nse_log_val",
        "pbias_calib",
        "pbias_val",
        "ks",
        "kl",
        "a",
        "expo_perdas",
    ]
    print(df[colunas].to_string(index=False))

    melhor = df.sort_values("nse_calib", ascending=False).iloc[0]
    print(
        f"✅ [{cenario}] período {periodo_id} finalizado | melhor fase={melhor['fase']} "
        f"| NSE_cal={melhor['nse_calib']:.5f} | NSE_val={melhor['nse_val']:.5f}"
    )


def _calcular_indices_por_intervalo(simulado, observado, mascara_calibracao, mascara_validacao):
    resultados = {}
    if mascara_calibracao is not None and mascara_calibracao.any():
        resultado_cal = calcular_indices_estatisticos(simulado[mascara_calibracao], observado[mascara_calibracao])
        resultados["calib"] = resultado_cal
    else:
        resultados["calib"] = {"Nash": np.nan, "Nash_log": np.nan, "Nash_sqrt": np.nan, "RMS": np.nan, "Pbias": np.nan}

    if mascara_validacao is not None and mascara_validacao.any():
        resultado_val = calcular_indices_estatisticos(simulado[mascara_validacao], observado[mascara_validacao])
        resultados["val"] = resultado_val
    else:
        resultados["val"] = {"Nash": np.nan, "Nash_log": np.nan, "Nash_sqrt": np.nan, "RMS": np.nan, "Pbias": np.nan}

    return resultados


def calibrar_periodo(
    nome_bacia: str,
    parametros: dict[str, Any],
    dados: dict[str, Any],
    xp,
    metodo: str,
    usar_parametros_a_expo: bool,
):
    mascara_calibracao = _mask_from_dates(dados["datas"], dados["calib_start"], dados["calib_end"])
    mascara_validacao = _mask_from_dates(dados["datas"], dados["val_start"], dados["val_end"])

    if mascara_calibracao is None or not mascara_calibracao.any():
        raise ValueError(f"{nome_bacia}: não foi possível construir máscaras de calibração/validação")

    if mascara_validacao is None:
        mascara_validacao = np.zeros(len(dados["datas"]), dtype=bool)

    observado = np.asarray(dados["vazao_obs"], dtype=float)
    observado_cal = observado[np.asarray(mascara_calibracao)]
    denom_cal = float(np.sum((observado_cal - np.mean(observado_cal)) ** 2)) if observado_cal.size else 0.0
    if denom_cal == 0.0:
        raise ValueError(f"{nome_bacia}: calibração com variância zero em vazão observada")

    if usar_parametros_a_expo:
        nomes_parametros = ("ks", "kl", "a", "expo_perdas")
        limites = LIMITES_PADRAO
        valores_iniciais = {
            "ks": _media_limites("ks"),
            "kl": _media_limites("kl"),
            "a": _media_limites("a"),
            "expo_perdas": _media_limites("expo_perdas"),
        }
        resultado = calibrar_parametros_pso(
            nome_bacia=nome_bacia,
            parametros=parametros,
            chuva=dados["chuva"],
            evaporacao=dados["evaporacao"],
            vazao_obs=dados["vazao_obs"],
            xp=xp,
            metodo=metodo,
            nomes_parametros=nomes_parametros,
            limites=limites,
            n_particulas=N_PARTICULAS,
            max_iter=MAX_ITER,
            paciencia=PACIENCIA,
            coeficientes=(0.8, 1.0, 2.0),
            mascara_calibracao=mascara_calibracao,
            valores_iniciais=valores_iniciais,
        )
    else:
        valores_iniciais = {
            "ks": _media_limites("ks"),
            "kl": _media_limites("kl"),
        }
        resultado = calibrar_parametros_pso(
            nome_bacia=nome_bacia,
            parametros=parametros,
            chuva=dados["chuva"],
            evaporacao=dados["evaporacao"],
            vazao_obs=dados["vazao_obs"],
            xp=xp,
            metodo=metodo,
            nomes_parametros=("ks", "kl"),
            limites={"ks": LIMITES_PADRAO["ks"], "kl": LIMITES_PADRAO["kl"]},
            n_particulas=N_PARTICULAS,
            max_iter=MAX_ITER,
            paciencia=PACIENCIA,
            coeficientes=(0.8, 1.0, 2.0),
            mascara_calibracao=mascara_calibracao,
            valores_iniciais=valores_iniciais,
        )

    simulado = _to_numpy_float_array(resultado["simulado"])
    observado = _to_numpy_float_array(dados["vazao_obs"])
    indices = _calcular_indices_por_intervalo(simulado, observado, mascara_calibracao, mascara_validacao)

    return {
        "resultado": resultado,
        "indices": indices,
        "mascara_calibracao": mascara_calibracao,
        "mascara_validacao": mascara_validacao,
    }


def carregar_resultados_referencia(session):
    referencia = {}
    for periodo in session.query(CalibrationPeriod).all():
        result = session.query(ModelResult).filter_by(calibration_period_id=periodo.id).first()
        referencia[periodo.id] = result
    return referencia


def processar(cenario: str, usar_area_resultado: bool):
    xp, _usar_gpu = get_array_backend(prefer_gpu=True)

    engine, Session = initialize_db()
    session = Session()
    try:
        periodos_por_bacia = _carregar_periodos_por_bacia(session)
        referencia = carregar_resultados_referencia(session)
        bacias = session.query(Bacia).order_by(Bacia.nome).all()

        resultados_finais: list[ResultadoMetodo] = []

        for bacia in bacias:
            nome_bacia = bacia.nome
            parametros = _normalizar_parametros_bacia(bacia)

            if parametros.get("area_km2") is None:
                print(f"⚠️ {nome_bacia}: sem parâmetros no banco, ignorando")
                continue

            if bacia.id not in periodos_por_bacia:
                print(f"⚠️ {nome_bacia}: sem períodos cadastrados, ignorando")
                continue

            series = _carregar_series_do_banco(session, bacia.id)
            if series is None:
                print(f"⚠️ {nome_bacia}: sem séries completas no banco, ignorando")
                continue

            vazao_obs, chuva, evaporacao, datas = series

            dados_base = {
                "vazao_obs": np.asarray(vazao_obs, dtype=float),
                "chuva": np.asarray(chuva, dtype=float),
                "evaporacao": np.asarray(evaporacao, dtype=float),
                "datas": pd.to_datetime(pd.Series(datas), errors="coerce").to_numpy(),
                "calib_start": None,
                "calib_end": None,
                "val_start": None,
                "val_end": None,
            }

            for periodo in sorted(periodos_por_bacia[bacia.id], key=lambda p: (p.station or "", p.method or "")):
                periodo_id = int(getattr(periodo, "id"))
                periodo_station = str(getattr(periodo, "station") or nome_bacia)
                periodo_method = str(getattr(periodo, "method") or "normal")
                referencia_periodo = referencia.get(periodo_id)

                try:
                    parametros_periodo, area_usada, fonte_area = _resolver_area_parametros(
                        parametros_base=parametros,
                        referencia_periodo=referencia_periodo,
                        usar_area_resultado=usar_area_resultado,
                    )
                except ValueError as exc:
                    print(f"⚠️ [{cenario}] {nome_bacia} / {periodo_station} / {periodo_method}: {exc}")
                    continue

                _imprimir_inicio_periodo(
                    cenario=cenario,
                    bacia=nome_bacia,
                    station=periodo_station,
                    method=periodo_method,
                    periodo_id=periodo_id,
                    area_usada=area_usada,
                    fonte_area=fonte_area,
                )

                calib_start, calib_end, val_start, val_end, ajustou_periodo = _corrigir_periodo_para_serie(periodo, dados_base["datas"])
                if ajustou_periodo:
                    print(
                        f"⚠️ {nome_bacia} / {periodo_station}: datas do período ajustadas para século correto "
                        f"({periodo.calib_start}..{periodo.calib_end} -> {calib_start}..{calib_end})"
                    )

                dados_periodo = dict(dados_base)
                dados_periodo["calib_start"] = calib_start
                dados_periodo["calib_end"] = calib_end
                dados_periodo["val_start"] = val_start
                dados_periodo["val_end"] = val_end

                resultados_periodo: list[ResultadoMetodo] = []

                for metodo in METODOS:
                    for usar_parametros_a_expo in (False, True):
                        fase = "ks_kl" if not usar_parametros_a_expo else "ks_kl_a_expo"
                        try:
                            saida = calibrar_periodo(
                                nome_bacia=nome_bacia,
                                parametros=parametros_periodo,
                                dados=dados_periodo,
                                xp=xp,
                                metodo=metodo,
                                usar_parametros_a_expo=usar_parametros_a_expo,
                            )
                        except ValueError as exc:
                            print(f"⚠️ {nome_bacia} / {periodo_station} / {periodo_method} / {fase}__{metodo}: {exc}")
                            continue

                        resultado = saida["resultado"]
                        indices = saida["indices"]
                        ref = _resultado_referencia(periodo_id, referencia)

                        item = ResultadoMetodo(
                            cenario=cenario,
                            periodo_id=periodo_id,
                            bacia=nome_bacia,
                            station=periodo_station,
                            method=periodo_method,
                            area_km2_usada=area_usada,
                            fonte_area=fonte_area,
                            fase=fase + f"__{metodo}",
                            nse_calib=float(indices["calib"]["Nash"]),
                            nse_val=float(indices["val"]["Nash"]),
                            nse_sqrt_calib=float(indices["calib"]["Nash_sqrt"]),
                            nse_sqrt_val=float(indices["val"]["Nash_sqrt"]),
                            nse_log_calib=float(indices["calib"]["Nash_log"]),
                            nse_log_val=float(indices["val"]["Nash_log"]),
                            pbias_calib=float(indices["calib"]["Pbias"]),
                            pbias_val=float(indices["val"]["Pbias"]),
                            nash_calib=float(indices["calib"]["Nash"]),
                            nash_val=float(indices["val"]["Nash"]),
                            ks=_to_float_scalar(resultado.get("ks", np.nan)),
                            kl=_to_float_scalar(resultado.get("kl", np.nan)),
                            a=_to_float_scalar(resultado.get("a")) if resultado.get("a") is not None else None,
                            expo_perdas=_to_float_scalar(resultado.get("expo_perdas")) if resultado.get("expo_perdas") is not None else None,
                            referencia_nse_calib=ref["nse_calib"],
                            referencia_nse_val=ref["nse_val"],
                            delta_nse_calib=_comparar_metrica(float(indices["calib"]["Nash"]), ref["nse_calib"]),
                            delta_nse_val=_comparar_metrica(float(indices["val"]["Nash"]), ref["nse_val"]),
                        )
                        resultados_finais.append(item)
                        resultados_periodo.append(item)

                _imprimir_resultado_periodo(
                    cenario=cenario,
                    periodo_id=periodo_id,
                    bacia=nome_bacia,
                    station=periodo_station,
                    method=periodo_method,
                    resultados_periodo=resultados_periodo,
                )

        return resultados_finais
    finally:
        session.close()
        engine.dispose()


def imprimir_relatorio(resultados_finais: list[ResultadoMetodo], titulo: str):
    print("\n" + "=" * 90)
    print(titulo)
    print("=" * 90)

    if not resultados_finais:
        print("Nenhum resultado gerado.")
        return

    df = pd.DataFrame([r.__dict__ for r in resultados_finais])
    colunas_utiles = [
        "bacia",
        "station",
        "method",
        "area_km2_usada",
        "fonte_area",
        "fase",
        "nash_calib",
        "nash_val",
        "nse_sqrt_calib",
        "nse_sqrt_val",
        "nse_log_calib",
        "nse_log_val",
        "pbias_calib",
        "pbias_val",
        "referencia_nse_calib",
        "referencia_nse_val",
        "delta_nse_calib",
        "delta_nse_val",
        "ks",
        "kl",
        "a",
        "expo_perdas",
    ]
    print(df[colunas_utiles].to_string(index=False))

    agrupado = df.groupby(["periodo_id", "bacia", "station", "method"], dropna=False)
    print("\nMelhor NSE por período (calibração):")
    for (periodo_id, bacia, station, method), grupo in agrupado:
        melhor = grupo.sort_values("nash_calib", ascending=False).iloc[0]
        ref = melhor["referencia_nse_calib"]
        delta = melhor["delta_nse_calib"]
        status = "melhor" if delta is not None and delta > 0 else "pior ou igual"
        print(
            f"- {bacia} / {station} ({method}) -> melhor NSE={melhor['nash_calib']:.5f} | "
            f"referência={ref if pd.notna(ref) else 'n/d'} | {status}"
        )


def comparar_cenarios_individualmente(resultados_bacia: list[ResultadoMetodo], resultados_resultado: list[ResultadoMetodo]):
    print("\n" + "=" * 90)
    print("COMPARAÇÃO INDIVIDUAL: ÁREA DA BACIA x ÁREA DE MODEL_RESULT")
    print("=" * 90)

    if not resultados_bacia or not resultados_resultado:
        print("Sem dados suficientes para comparar os dois cenários.")
        return

    df_bacia = pd.DataFrame([r.__dict__ for r in resultados_bacia]).rename(
        columns={
            "nse_calib": "nse_calib_bacia",
            "nse_val": "nse_val_bacia",
            "nse_sqrt_calib": "nse_sqrt_calib_bacia",
            "nse_sqrt_val": "nse_sqrt_val_bacia",
            "nse_log_calib": "nse_log_calib_bacia",
            "nse_log_val": "nse_log_val_bacia",
            "pbias_calib": "pbias_calib_bacia",
            "pbias_val": "pbias_val_bacia",
            "area_km2_usada": "area_km2_bacia",
        }
    )
    df_res = pd.DataFrame([r.__dict__ for r in resultados_resultado]).rename(
        columns={
            "nse_calib": "nse_calib_resultado",
            "nse_val": "nse_val_resultado",
            "nse_sqrt_calib": "nse_sqrt_calib_resultado",
            "nse_sqrt_val": "nse_sqrt_val_resultado",
            "nse_log_calib": "nse_log_calib_resultado",
            "nse_log_val": "nse_log_val_resultado",
            "pbias_calib": "pbias_calib_resultado",
            "pbias_val": "pbias_val_resultado",
            "area_km2_usada": "area_km2_resultado",
        }
    )

    chaves = ["periodo_id", "bacia", "station", "method", "fase"]
    comparado = df_bacia.merge(df_res, on=chaves, how="inner")

    if comparado.empty:
        print("Não houve interseção de resultados entre os dois cenários.")
        return

    comparado["delta_nse_calib"] = comparado["nse_calib_resultado"] - comparado["nse_calib_bacia"]
    comparado["delta_nse_val"] = comparado["nse_val_resultado"] - comparado["nse_val_bacia"]
    comparado["delta_nse_sqrt_calib"] = comparado["nse_sqrt_calib_resultado"] - comparado["nse_sqrt_calib_bacia"]
    comparado["delta_nse_log_calib"] = comparado["nse_log_calib_resultado"] - comparado["nse_log_calib_bacia"]
    comparado["delta_pbias_calib"] = comparado["pbias_calib_resultado"] - comparado["pbias_calib_bacia"]

    colunas = [
        "periodo_id",
        "bacia",
        "station",
        "method",
        "fase",
        "area_km2_bacia",
        "area_km2_resultado",
        "nse_calib_bacia",
        "nse_calib_resultado",
        "delta_nse_calib",
        "nse_val_bacia",
        "nse_val_resultado",
        "delta_nse_val",
        "delta_nse_sqrt_calib",
        "delta_nse_log_calib",
        "delta_pbias_calib",
    ]
    print(comparado[colunas].sort_values(["periodo_id", "fase"]).to_string(index=False))


if __name__ == "__main__":
    resultados_bacia = processar(cenario="area_da_bacia", usar_area_resultado=False)
    imprimir_relatorio(resultados_bacia, "RELATÓRIO DE PROCESSAMENTO EM RAM - CENÁRIO: ÁREA DA BACIA")


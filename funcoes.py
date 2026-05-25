import glob
import os
import time
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


WARMUP_PADRAO_DIAS = 365


def _converter_escalar(valor):
    """Converte valores numpy/cupy para escalar Python."""
    if hasattr(valor, "item"):
        try:
            return valor.item()
        except Exception:
            pass
    if hasattr(valor, "get"):
        try:
            return valor.get().item()
        except Exception:
            pass
    return valor


def _converter_para_numpy(valor):
    if hasattr(valor, "get"):
        try:
            return valor.get()
        except Exception:
            pass
    return np.asarray(valor)


def _mascara_intervalo(n_dias, intervalo_df, usar_linha=0):
    if intervalo_df is None or intervalo_df.empty:
        return None

    indice = min(max(int(usar_linha), 0), len(intervalo_df) - 1)
    inicio = int(intervalo_df.iloc[indice]["inicio"])
    fim = int(intervalo_df.iloc[indice]["fim"])
    dias = np.arange(1, n_dias + 1)
    return (dias >= inicio) & (dias <= fim)


def _mascara_intervalo_total(n_dias, intervalo_df):
    if intervalo_df is None or intervalo_df.empty:
        return None

    inicio = int(intervalo_df["inicio"].min())
    fim = int(intervalo_df["fim"].max())
    dias = np.arange(1, n_dias + 1)
    return (dias >= inicio) & (dias <= fim)


def calcular_indices_estatisticos(simulado, observado):
    simulado = pd.Series(simulado).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    observado = pd.Series(observado).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(simulado) == 0 or len(observado) == 0:
        return {
            "Nash": np.nan,
            "Nash_log": np.nan,
            "Nash_sqrt": np.nan,
            "RMS": np.nan,
            "Pbias": np.nan,
        }

    def _nse(a, b):
        denom = np.sum((b - np.mean(b)) ** 2)
        if denom == 0:
            return np.nan
        return 1.0 - (np.sum((a - b) ** 2) / denom)

    sim = simulado.to_numpy(dtype=float)
    obs = observado.to_numpy(dtype=float)
    eps = 1e-6

    nash = _nse(sim, obs)
    nash_log = _nse(np.log1p(np.clip(sim, 0.0, None) + eps), np.log1p(np.clip(obs, 0.0, None) + eps))
    nash_sqrt = _nse(np.sqrt(np.clip(sim, 0.0, None)), np.sqrt(np.clip(obs, 0.0, None)))
    rms = float(np.sqrt(np.mean((sim - obs) ** 2)))
    denom_pbias = np.sum(obs)
    pbias = np.nan if denom_pbias == 0 else float(100.0 * np.sum(sim - obs) / denom_pbias)

    return {
        "Nash": float(nash) if pd.notna(nash) else np.nan,
        "Nash_log": float(nash_log) if pd.notna(nash_log) else np.nan,
        "Nash_sqrt": float(nash_sqrt) if pd.notna(nash_sqrt) else np.nan,
        "RMS": rms,
        "Pbias": pbias,
    }


def _montar_variaveis_otimizadas(posicao, nomes_parametros):
    indices = {nome: idx for idx, nome in enumerate(nomes_parametros)}

    def _coluna(nome):
        indice = indices.get(nome)
        return None if indice is None else posicao[:, indice]

    return {
        "ks": _coluna("ks"),
        "kl": _coluna("kl"),
        "a": _coluna("a"),
        "expo_perdas": _coluna("expo_perdas"),
    }


def _avaliar_pso_parametros(
    parametros,
    chuva,
    evaporacao,
    vazao_obs,
    xp,
    posicao,
    nomes_parametros,
    metodo,
    dias_spinup=WARMUP_PADRAO_DIAS,
    mascara_calibracao=None,
):
    variaveis = _montar_variaveis_otimizadas(posicao, nomes_parametros)
    ks = variaveis["ks"]
    kl = variaveis["kl"]

    simulado = simular_cawm_particulas(
        parametros,
        chuva,
        evaporacao,
        ks,
        kl,
        xp,
        metodo=metodo,
        a_override=variaveis["a"],
        expo_perdas_override=variaveis["expo_perdas"],
    )

    mascara = mascara_calibracao
    if mascara is not None:
        mascara = xp.asarray(mascara, dtype=bool)
    if metodo == "spinup":
        mascara_spinup = xp.arange(simulado.shape[1]) >= dias_spinup
        mascara = mascara_spinup if mascara is None else (mascara & mascara_spinup)

    nse = calcular_nse(simulado, vazao_obs, xp, mascara=mascara)
    objetivo = 1.0 - nse
    return simulado, nse, objetivo


def calcular_nse(simulado, observado, xp, mascara=None):
    """Calcula o NSE para uma série ou para várias partículas em paralelo."""
    observado = xp.asarray(observado, dtype=xp.float32)
    simulado = xp.asarray(simulado, dtype=xp.float32)

    if mascara is not None:
        mascara = xp.asarray(mascara, dtype=bool)
        observado = observado[mascara]
        simulado = simulado[:, mascara] if simulado.ndim == 2 else simulado[mascara]

    if observado.size == 0:
        if simulado.ndim == 2:
            return xp.full((simulado.shape[0],), -1e6, dtype=xp.float32)
        return xp.array(-1e6, dtype=xp.float32)

    observado = xp.nan_to_num(observado, nan=0.0, posinf=0.0, neginf=0.0)
    media_observado = xp.mean(observado)
    denominador = xp.sum((observado - media_observado) ** 2)

    if float(_converter_escalar(denominador)) == 0.0:
        if simulado.ndim == 2:
            return xp.full((simulado.shape[0],), -1e6, dtype=xp.float32)
        return xp.array(-1e6, dtype=xp.float32)

    if simulado.ndim == 1:
        simulado = xp.nan_to_num(simulado, nan=0.0, posinf=0.0, neginf=0.0)
        numerador = xp.sum((simulado - observado) ** 2)
        return 1.0 - numerador / denominador

    simulado = xp.nan_to_num(simulado, nan=0.0, posinf=0.0, neginf=0.0)
    numerador = xp.sum((simulado - observado) ** 2, axis=1)
    return 1.0 - numerador / denominador


def simular_cawm_particulas(
    parametros,
    chuva,
    evaporacao,
    ks,
    kl,
    xp,
    metodo="estado_inicial",
    a_override=None,
    expo_perdas_override=None,
):
    """Simula o CAWM para várias partículas simultâneas."""
    chuva = xp.asarray(chuva, dtype=xp.float32)
    evaporacao = xp.asarray(evaporacao, dtype=xp.float32)
    ks = xp.asarray(ks, dtype=xp.float32)
    kl = xp.asarray(kl, dtype=xp.float32)

    n_particulas = int(ks.shape[0])
    n_dias = int(chuva.shape[0])

    area_km2 = float(parametros["area_km2"])
    area_m2 = area_km2 * 1_000_000.0
    submax = max(float(parametros.get("submax", 1.0) or 1.0), 1e-6)
    gmax = max(float(parametros.get("gmax", 0.0) or 0.0), 0.0)
    a_base = float(parametros.get("a", 0.0) or 0.0)
    expo_base = float(parametros.get("expo_perdas", 1.0) or 1.0)
    if a_override is None:
        a = xp.full(n_particulas, a_base, dtype=xp.float32)
    else:
        a = xp.asarray(a_override, dtype=xp.float32)
        if a.ndim == 0:
            a = xp.full(n_particulas, float(_converter_escalar(a)), dtype=xp.float32)
    if expo_perdas_override is None:
        expo_perdas = xp.full(n_particulas, expo_base, dtype=xp.float32)
    else:
        expo_perdas = xp.asarray(expo_perdas_override, dtype=xp.float32)
        if expo_perdas.ndim == 0:
            expo_perdas = xp.full(n_particulas, float(_converter_escalar(expo_perdas)), dtype=xp.float32)
    beta = float(parametros.get("beta", 2.0) or 2.0)
    kg = float(parametros.get("kg", 1.0) or 1.0)
    rio = int(parametros.get("rio", 0) or 0)
    k = float(parametros.get("k", 0.0) or 0.0)
    b = float(parametros.get("b", 1.666666667) or 1.666666667)
    T = float(parametros.get("T", 86400.0) or 86400.0)

    if metodo == "spinup":
        reserv_solo = xp.zeros(n_particulas, dtype=xp.float32)
        profundo_corrigido = xp.zeros(n_particulas, dtype=xp.float32)
        s3 = xp.zeros(n_particulas, dtype=xp.float32)
    else:
        reserv_solo = xp.full(n_particulas, float(parametros.get("reserva_solo_inicial", 0.0) or 0.0), dtype=xp.float32)
        reserv_solo = xp.clip(reserv_solo, 0.0, submax)
        profundo_corrigido = xp.full(n_particulas, float(parametros.get("profundo_inicial", 0.0) or 0.0), dtype=xp.float32)
        if gmax > 0:
            profundo_corrigido = xp.clip(profundo_corrigido, 0.0, gmax)
        else:
            profundo_corrigido = xp.maximum(profundo_corrigido, 0.0)
        s3 = xp.full(n_particulas, float(parametros.get("s3_inicial", 0.0) or 0.0), dtype=xp.float32)
        s3 = xp.maximum(s3, 0.0)

    ret_corrig = xp.zeros(n_particulas, dtype=xp.float32)
    simulado = xp.zeros((n_particulas, n_dias), dtype=xp.float32)

    divisor_beta = beta if beta != 0 else 1.0

    for dia in range(n_dias):
        chuva_dia = chuva[dia]
        evap_dia = evaporacao[dia]

        evap_inicial = xp.minimum(ret_corrig + chuva_dia, evap_dia)
        retencao = xp.maximum(ret_corrig + chuva_dia - evap_inicial, 0.0)
        evap_n_atendida = evap_dia - evap_inicial
        ret_corrig = xp.where(retencao > 0, xp.zeros_like(retencao), retencao)

        pn = chuva_dia - evap_inicial
        if rio == 0:
            hiperb = xp.tanh(pn / submax)
            sub = reserv_solo / submax
            denominador = 1.0 + sub * hiperb
            denominador = xp.where(xp.abs(denominador) < 1e-12, 1e-12, denominador)
            ps = submax * (1.0 - sub ** 2) * hiperb / denominador
        else:
            ps = pn * (1.0 - reserv_solo / submax / divisor_beta)

        ps = xp.maximum(ps, 0.0)
        rec_solo = ps
        escoamento = chuva_dia - evap_inicial - rec_solo

        re_raw = (1.0 - xp.exp(-a * (reserv_solo / submax))) * evap_n_atendida
        re = xp.minimum(xp.minimum(evap_n_atendida, reserv_solo), re_raw)
        solo = xp.maximum(reserv_solo - re, 0.0)
        rec_rio = ks * solo

        s1 = s3 + escoamento + rec_rio
        c = xp.where(s1 <= 0, 0.0, xp.minimum(k * (s1 ** b), s1))
        perdas = xp.minimum(kl * (c ** expo_perdas), c)
        simulado[:, dia] = ((c - perdas) / 1000.0) * (area_m2 / T)

        reserv_solo_corrig = xp.maximum(solo + rec_solo - rec_rio, 0.0)

        if rio == 0:
            profundo = xp.zeros_like(s1)
            profundo_corrigido = xp.zeros_like(s1)
        else:
            precisa_recarregar = (reserv_solo + ps - rec_rio) > submax
            profundo = xp.where(
                precisa_recarregar,
                profundo_corrigido + reserv_solo + ps - rec_rio - submax,
                profundo_corrigido,
            )
            fg = profundo * kg
            limite_superior = gmax if gmax > 0 else xp.max(profundo - fg)
            profundo_corrigido = xp.clip(profundo - fg, 0.0, limite_superior)

        s3 = s1 - c + profundo - profundo_corrigido
        reserv_solo = xp.minimum(reserv_solo_corrig, submax)

    return simulado


def avaliar_metodo_calibracao(
    parametros,
    chuva,
    evaporacao,
    vazao_obs,
    xp,
    ks,
    kl,
    metodo,
    dias_spinup=WARMUP_PADRAO_DIAS,
    mascara_calibracao=None,
):
    """Simula e devolve NSE por partícula para um método de inicialização."""
    simulado = simular_cawm_particulas(parametros, chuva, evaporacao, ks, kl, xp, metodo=metodo)
    mascara = mascara_calibracao
    if mascara is not None:
        mascara = xp.asarray(mascara, dtype=bool)
    if metodo == "spinup":
        mascara_spinup = xp.arange(simulado.shape[1]) >= dias_spinup
        mascara = mascara_spinup if mascara is None else (mascara & mascara_spinup)

    nse = calcular_nse(simulado, vazao_obs, xp, mascara=mascara)
    objetivo = 1.0 - nse
    return simulado, nse, objetivo


def calibrar_parametros_pso(
    nome_bacia,
    parametros,
    chuva,
    evaporacao,
    vazao_obs,
    xp,
    metodo,
    nomes_parametros,
    limites,
    n_particulas=2000,
    max_iter=10,
    paciencia=10,
    coeficientes=(0.8, 1.0, 2.0),
    dias_spinup=WARMUP_PADRAO_DIAS,
    mascara_calibracao=None,
    valores_iniciais=None,
):
    """Calibra qualquer conjunto de parâmetros do CAWM com PSO."""
    w, c1, c2 = coeficientes
    nomes_parametros = list(nomes_parametros)

    posicoes_iniciais = []
    for nome in nomes_parametros:
        if nome not in limites:
            raise KeyError(f"Limite não informado para o parâmetro '{nome}'")
        minimo, maximo = limites[nome]
        posicoes_iniciais.append(xp.random.uniform(minimo, maximo, size=n_particulas).astype(xp.float32))

    posicao = xp.stack(posicoes_iniciais, axis=1)
    if valores_iniciais:
        for nome, valor in valores_iniciais.items():
            if nome in nomes_parametros:
                indice = nomes_parametros.index(nome)
                minimo, maximo = limites[nome]
                posicao[0, indice] = xp.clip(xp.asarray(valor, dtype=xp.float32), minimo, maximo)

    velocidade = xp.zeros_like(posicao)

    _, nse_atual, objetivo_atual = _avaliar_pso_parametros(
        parametros,
        chuva,
        evaporacao,
        vazao_obs,
        xp,
        posicao,
        nomes_parametros,
        metodo,
        dias_spinup=dias_spinup,
        mascara_calibracao=mascara_calibracao,
    )

    melhor_posicao = posicao.copy()
    melhor_objetivo = objetivo_atual.copy()
    melhor_indice = int(_converter_escalar(xp.argmin(melhor_objetivo)))
    melhor_pos_global = melhor_posicao[melhor_indice].copy()
    melhor_obj_global = melhor_objetivo[melhor_indice].copy()

    historico = [
        {
            "iteracao": 0,
            "melhor_nse": float(_converter_escalar(1.0 - melhor_obj_global)),
            "melhor_objetivo": float(_converter_escalar(melhor_obj_global)),
            **{nome: float(_converter_escalar(melhor_pos_global[idx])) for idx, nome in enumerate(nomes_parametros)},
        }
    ]

    sem_melhoria = 0

    for iteracao in range(max_iter):
        r1 = xp.random.random(size=posicao.shape).astype(xp.float32)
        r2 = xp.random.random(size=posicao.shape).astype(xp.float32)

        melhor_pos_global_expandida = xp.broadcast_to(melhor_pos_global, posicao.shape)
        velocidade = (
            w * velocidade
            + c1 * r1 * (melhor_posicao - posicao)
            + c2 * r2 * (melhor_pos_global_expandida - posicao)
        )
        posicao = posicao + velocidade

        for indice, nome in enumerate(nomes_parametros):
            minimo, maximo = limites[nome]
            posicao[:, indice] = xp.clip(posicao[:, indice], minimo, maximo)

        _, nse_atual, objetivo_atual = _avaliar_pso_parametros(
            parametros,
            chuva,
            evaporacao,
            vazao_obs,
            xp,
            posicao,
            nomes_parametros,
            metodo,
            dias_spinup=dias_spinup,
            mascara_calibracao=mascara_calibracao,
        )

        melhorou = objetivo_atual < melhor_objetivo
        melhor_objetivo = xp.where(melhorou, objetivo_atual, melhor_objetivo)
        melhor_posicao = xp.where(melhorou[:, None], posicao, melhor_posicao)

        melhor_indice = int(_converter_escalar(xp.argmin(melhor_objetivo)))
        houve_melhora_global = False
        if melhor_objetivo[melhor_indice] < melhor_obj_global:
            melhor_obj_global = melhor_objetivo[melhor_indice].copy()
            melhor_pos_global = melhor_posicao[melhor_indice].copy()
            houve_melhora_global = True

        if houve_melhora_global:
            sem_melhoria = 0
        else:
            sem_melhoria += 1

        print(
            f"    Iteração {iteracao + 1:02d}/{max_iter} - {metodo}: "
            f"melhor NSE parcial = {float(_converter_escalar(1.0 - melhor_obj_global)):.5f}"
        )

        historico.append(
            {
                "iteracao": iteracao + 1,
                "melhor_nse": float(_converter_escalar(1.0 - melhor_obj_global)),
                "melhor_objetivo": float(_converter_escalar(melhor_obj_global)),
                **{nome: float(_converter_escalar(melhor_pos_global[idx])) for idx, nome in enumerate(nomes_parametros)},
            }
        )

        if paciencia is not None and paciencia > 0 and sem_melhoria >= paciencia:
            print(
                f"    ⏹️ Parada por paciência ({paciencia}) em {metodo}: "
                f"sem melhora nas últimas {sem_melhoria} iterações"
            )
            break

    parametros_melhores = {
        nome: float(_converter_escalar(melhor_pos_global[idx])) for idx, nome in enumerate(nomes_parametros)
    }

    simulado_final, nse_final, _ = _avaliar_pso_parametros(
        parametros,
        chuva,
        evaporacao,
        vazao_obs,
        xp,
        xp.asarray([melhor_pos_global], dtype=xp.float32),
        nomes_parametros,
        metodo,
        dias_spinup=dias_spinup,
        mascara_calibracao=mascara_calibracao,
    )

    resultado = {
        "nome_bacia": nome_bacia,
        "metodo": metodo,
        "nse": float(_converter_escalar(nse_final[0])) if getattr(nse_final, "ndim", 0) else float(_converter_escalar(nse_final)),
        "simulado": simulado_final[0],
        "melhor_objetivo": float(_converter_escalar(melhor_obj_global)),
        "historico": historico,
        "n_particulas": n_particulas,
        "max_iter": max_iter,
        "coeficientes": coeficientes,
        "parametros_otimizados": parametros_melhores,
    }
    resultado.update(parametros_melhores)
    return resultado


def calibrar_ks_kl_pso(
    nome_bacia,
    parametros,
    chuva,
    evaporacao,
    vazao_obs,
    xp,
    metodo,
    n_particulas=2000,
    max_iter=10,
    paciencia=10,
    coeficientes=(0.8, 1.0, 2.0),
    limites=None,
    dias_spinup=WARMUP_PADRAO_DIAS,
    mascara_calibracao=None,
):
    """Calibra Ks e Kl com PSO e devolve o melhor NSE do método informado."""
    if limites is None:
        limites = {
            "ks": (0.00001, 1.0),
            "kl": (0.0, 2.0),
        }

    return calibrar_parametros_pso(
        nome_bacia,
        parametros,
        chuva,
        evaporacao,
        vazao_obs,
        xp,
        metodo,
        nomes_parametros=("ks", "kl"),
        limites=limites,
        n_particulas=n_particulas,
        max_iter=max_iter,
        paciencia=paciencia,
        coeficientes=coeficientes,
        dias_spinup=dias_spinup,
        mascara_calibracao=mascara_calibracao,
    )


def comparar_metodos_calibracao(
    nome_bacia,
    parametros,
    chuva,
    evaporacao,
    vazao_obs,
    xp,
    n_particulas=2000,
    max_iter=10,
    paciencia=10,
    coeficientes=(0.8, 1.0, 2.0),
    limites=None,
    dias_spinup=WARMUP_PADRAO_DIAS,
    intervalos=None,
):
    """Executa os dois métodos e devolve o melhor resultado por NSE."""
    inicio_calibracao = time.perf_counter()
    if len(chuva) <= dias_spinup:
        print(
            f"  ⚠️ {nome_bacia}: série com {len(chuva)} dias; o método spin-up ficará sem janela útil."
        )

    mascara_calibracao = _mascara_intervalo(len(chuva), intervalos, usar_linha=0)

    resultados = []
    for metodo in ("estado_inicial", "spinup"):
        print(f"\n🔧 {nome_bacia}: calibrando método '{metodo}'")
        resultado = calibrar_ks_kl_pso(
            nome_bacia,
            parametros,
            chuva,
            evaporacao,
            vazao_obs,
            xp,
            metodo,
            n_particulas=n_particulas,
            max_iter=max_iter,
            paciencia=paciencia,
            coeficientes=coeficientes,
            limites=limites,
            dias_spinup=dias_spinup,
            mascara_calibracao=mascara_calibracao,
        )
        resultados.append(resultado)
        print(
            f"  ✅ {nome_bacia} [{metodo}]: NSE = {resultado['nse']:.5f} | Ks = {resultado['ks']:.6f} | Kl = {resultado['kl']:.6f}"
        )

    melhor = max(resultados, key=lambda item: item["nse"])
    tempo_total_segundos = time.perf_counter() - inicio_calibracao
    melhor["tempo_total_segundos"] = tempo_total_segundos
    melhor["intervalos"] = intervalos
    melhor["resultados_metodos"] = resultados
    melhor["parametros"] = parametros
    print(
        f"\n🏁 {nome_bacia}: melhor método = {melhor['metodo']} com NSE = {melhor['nse']:.5f}"
    )
    return resultados, melhor, tempo_total_segundos


def simular_cawm_serie_unica(
    parametros,
    dados,
    ks,
    kl,
    metodo="estado_inicial",
    a_override=None,
    expo_perdas_override=None,
):
    datas = list(dados["datas"])
    # Converter CuPy arrays para NumPy se necessário
    vasao_raw = dados["vazao_obs"]
    chuva_raw = dados["chuva"]
    evaporacao_raw = dados["evaporacao"]
    
    vazao_obs = np.asarray(vasao_raw.get() if hasattr(vasao_raw, "get") else vasao_raw, dtype=float)
    chuva = np.asarray(chuva_raw.get() if hasattr(chuva_raw, "get") else chuva_raw, dtype=float)
    evaporacao = np.asarray(evaporacao_raw.get() if hasattr(evaporacao_raw, "get") else evaporacao_raw, dtype=float)

    area_km2 = float(parametros["area_km2"])
    area_m2 = area_km2 * 1_000_000.0
    submax = max(float(parametros.get("submax", 1.0) or 1.0), 1e-6)
    gmax = max(float(parametros.get("gmax", 0.0) or 0.0), 0.0)
    a = float(parametros.get("a", 0.0) or 0.0) if a_override is None else float(a_override)
    expo_perdas = float(parametros.get("expo_perdas", 1.0) or 1.0) if expo_perdas_override is None else float(expo_perdas_override)
    beta = float(parametros.get("beta", 2.0) or 2.0)
    kg = float(parametros.get("kg", 1.0) or 1.0)
    rio = int(parametros.get("rio", 0) or 0)
    k = float(parametros.get("k", 0.0) or 0.0)
    b = float(parametros.get("b", 1.666666667) or 1.666666667)
    T = float(parametros.get("T", 86400.0) or 86400.0)
    reserva_solo_inicial = float(parametros.get("reserva_solo_inicial", 0.0) or 0.0)
    profundo_inicial = float(parametros.get("profundo_inicial", 0.0) or 0.0)
    s3_inicial = float(parametros.get("s3_inicial", 0.0) or 0.0)

    if metodo == "spinup":
        reserv_solo_estado = 0.0
        profundo_corrigido_estado = 0.0
        s3_estado = 0.0
    else:
        reserv_solo_estado = min(max(reserva_solo_inicial, 0.0), submax)
        profundo_corrigido_estado = max(profundo_inicial, 0.0)
        if gmax > 0:
            profundo_corrigido_estado = min(profundo_corrigido_estado, gmax)
        s3_estado = max(s3_inicial, 0.0)

    ret_corrig = 0.0
    n = len(chuva)
    coluna_mes = []
    evap_inicial = np.zeros(n, dtype=float)
    retencao = np.zeros(n, dtype=float)
    evap_n_atendida = np.zeros(n, dtype=float)
    ret_corrig_hist = np.zeros(n, dtype=float)
    escoamento = np.zeros(n, dtype=float)
    reserv_solo_hist = np.zeros(n, dtype=float)
    s1 = np.zeros(n, dtype=float)
    re = np.zeros(n, dtype=float)
    solo = np.zeros(n, dtype=float)
    c = np.zeros(n, dtype=float)
    s2 = np.zeros(n, dtype=float)
    vazao_calc = np.zeros(n, dtype=float)
    ps = np.zeros(n, dtype=float)
    rec_solo = np.zeros(n, dtype=float)
    rec_rio = np.zeros(n, dtype=float)
    perdas = np.zeros(n, dtype=float)
    reserv_solo_corrig = np.zeros(n, dtype=float)
    profundo = np.zeros(n, dtype=float)
    fg = np.zeros(n, dtype=float)
    profundo_corrigido_hist = np.zeros(n, dtype=float)
    s3_hist = np.zeros(n, dtype=float)

    meses = pd.to_datetime(pd.Series(datas), errors="coerce").dt.month.fillna(1).astype(int).to_numpy()
    divisor_beta = beta if beta != 0 else 1.0
    area_fator = (area_m2 / T) / 1000.0

    for i in range(n):
        coluna_mes.append(int(meses[i]))
        if ret_corrig + chuva[i] >= evaporacao[i]:
            evap_inicial[i] = evaporacao[i]
        else:
            evap_inicial[i] = ret_corrig + chuva[i]

        retencao[i] = max(ret_corrig + chuva[i] - evap_inicial[i], 0.0)
        evap_n_atendida[i] = evaporacao[i] - evap_inicial[i]
        ret_corrig = 0.0 if retencao[i] > 0 else retencao[i]
        ret_corrig_hist[i] = ret_corrig

        reserv_solo_hist[i] = reserv_solo_estado
        reserv_solo = min(max(reserv_solo_estado, 0.0), submax)

        pn = chuva[i] - evap_inicial[i]
        if rio == 0:
            hiperb = np.tanh(pn / submax)
            sub = reserv_solo / submax
            termo1 = submax * (1.0 - sub ** 2) * hiperb
            termo2 = 1.0 + sub * hiperb
            ps[i] = max(termo1 / termo2 if abs(termo2) > 1e-12 else 0.0, 0.0)
        else:
            ps[i] = max(pn * (1.0 - reserv_solo / submax / divisor_beta), 0.0)

        rec_solo[i] = ps[i]
        escoamento[i] = chuva[i] - evap_inicial[i] - rec_solo[i]
        re_termo = (1.0 - np.exp(-a * (reserv_solo / submax))) * evap_n_atendida[i]
        re[i] = min(evap_n_atendida[i], reserv_solo, re_termo)
        solo[i] = max(reserv_solo - re[i], 0.0)
        rec_rio[i] = ks * solo[i]

        s1[i] = s3_estado + escoamento[i] + rec_rio[i]
        if s1[i] <= 0:
            c[i] = 0.0
        else:
            c[i] = min(k * (s1[i] ** b), s1[i])

        s2[i] = s1[i] - c[i]
        perdas[i] = min(kl * (c[i] ** expo_perdas), c[i])
        vazao_calc[i] = ((c[i] - perdas[i]) / 1000.0) * area_fator
        reserv_solo_corrig[i] = max(solo[i] + rec_solo[i] - rec_rio[i], 0.0)

        if rio == 0:
            profundo[i] = 0.0
            profundo_corrigido_hist[i] = 0.0
        else:
            if reserv_solo + ps[i] - rec_rio[i] > submax:
                profundo[i] = profundo_corrigido_estado + reserv_solo + ps[i] - rec_rio[i] - submax
            else:
                profundo[i] = profundo_corrigido_estado
            fg[i] = profundo[i] * kg
            if gmax > 0:
                profundo_corrigido_hist[i] = min(max(profundo[i] - fg[i], 0.0), gmax)
            else:
                profundo_corrigido_hist[i] = max(profundo[i] - fg[i], 0.0)

        s3_estado = s2[i] + profundo[i] - profundo_corrigido_hist[i]
        s3_hist[i] = s3_estado
        profundo_corrigido_estado = profundo_corrigido_hist[i]
        reserv_solo_estado = min(max(reserv_solo_corrig[i], 0.0), submax)

    sumario = {
        "volume_precipitado": float(np.sum(chuva)),
        "lamina_Q_observada": float(1000.0 * (np.sum(vazao_obs) * 86400.0) / area_m2),
        "evaporacao_potencial": float(np.sum(evaporacao)),
        "evaporacao_real_maxima": float(np.sum(chuva) - float(1000.0 * (np.sum(vazao_obs) * 86400.0) / area_m2)),
        "evaporacao_real": float(np.sum(evap_inicial) + np.sum(re)),
        "solo_inicio": float(reserva_solo_inicial + profundo_inicial),
        "solo_fim": float(reserv_solo_corrig[-1] + profundo_corrigido_hist[-1]),
        "armazenamento_sup_inicio": float(s1[0]),
        "armazenamento_sup_fim": float(s1[-1]),
        "escoado": float(np.sum(c) - np.sum(perdas)),
        "balanco": float(np.sum(chuva) + (reserva_solo_inicial + profundo_inicial) + s1[0] - np.sum(evap_inicial + re) - (np.sum(c) - np.sum(perdas)) - (reserv_solo_corrig[-1] + profundo_corrigido_hist[-1]) - s1[-1] - np.sum(perdas)),
    }

    df = pd.DataFrame(
        {
            "data": datas,
            "vazao": vazao_obs,
            "chuva_media": chuva,
            "mes": coluna_mes,
            "evaporacao": evaporacao,
            "evap_inicial": evap_inicial,
            "retencao": retencao,
            "evap_n_atendida": evap_n_atendida,
            "ret_corrig": ret_corrig_hist,
            "escoamento": escoamento,
            "reserv_solo": reserv_solo_hist if np.any(reserv_solo_hist) else np.full(n, reserva_solo_inicial if metodo != "spinup" else 0.0),
            "S1": s1,
            "RE": re,
            "Solo": solo,
            "C": c,
            "S2": s2,
            "vazao_calc": vazao_calc,
            "Ps": ps,
            "rec_solo": rec_solo,
            "rec_rio": rec_rio,
            "perdas": perdas,
            "reserv_solo_corrig": reserv_solo_corrig,
            "profundo": profundo,
            "Fg": fg,
            "profundo_corrigido": profundo_corrigido_hist,
            "S3": s3_hist,
        }
    )

    for chave, valor in sumario.items():
        df[chave] = ""
    if len(df) > 0:
        for chave, valor in sumario.items():
            df.loc[0, chave] = f"{valor:.6f}"

    return df


def salvar_resultados_calibracao(
    nome_bacia,
    dados,
    melhor,
    n_particulas,
    simulacao_inicial=None,
    rotulo_inicial="baseline",
    rotulo_saida=None,
):
    pasta_resultados = os.path.join(dados["pasta_bacia"], f"resultados_{n_particulas}", melhor["metodo"])
    os.makedirs(pasta_resultados, exist_ok=True)

    sufixo_saida = "" if not rotulo_saida else f"_{rotulo_saida}"

    df_cawm = simular_cawm_serie_unica(
        melhor["parametros"],
        dados,
        melhor["ks"],
        melhor["kl"],
        metodo=melhor["metodo"],
        a_override=melhor.get("a"),
        expo_perdas_override=melhor.get("expo_perdas"),
    )

    caminho_cawm = os.path.join(pasta_resultados, f"CAWM{sufixo_saida}.csv")
    df_cawm.to_csv(caminho_cawm, index=True)

    if simulacao_inicial is not None:
        caminho_cawm_inicial = os.path.join(pasta_resultados, f"CAWM_{rotulo_inicial}{sufixo_saida}.csv")
        simulacao_inicial.to_csv(caminho_cawm_inicial, index=True)
        caminho_metricas_inicial = os.path.join(pasta_resultados, f"Resultados_estatisticos_{rotulo_inicial}{sufixo_saida}.csv")
        caminho_pdf_inicial = os.path.join(pasta_resultados, f"Graficos_calibracao_{rotulo_inicial}{sufixo_saida}.pdf")
        calcular_e_salvar_estatisticas(
            caminho_cawm_inicial,
            dados.get("intervalos"),
            caminho_metricas_inicial,
            caminho_pdf_inicial,
        )
        melhor["caminho_cawm_inicial"] = caminho_cawm_inicial
        melhor["caminho_metricas_inicial"] = caminho_metricas_inicial
        melhor["caminho_pdf_inicial"] = caminho_pdf_inicial

    caminho_convergencia = os.path.join(pasta_resultados, f"Grafico_convergencia{sufixo_saida}.png")
    plt.figure(figsize=(9, 5))
    resultados_convergencia = melhor.get("resultados_metodos") or [melhor]
    for resultado in resultados_convergencia:
        historico = resultado.get("historico", [])
        if not historico:
            continue
        iteracoes = [item["iteracao"] for item in historico]
        objetivos = [item["melhor_objetivo"] for item in historico]
        plt.plot(iteracoes, objetivos, marker="o", linewidth=2, label=f"{resultado.get('metodo', melhor['metodo'])}")
    plt.xlabel("Iteração")
    plt.ylabel("Função objetivo (1 - NSE)")
    plt.title(f"Convergência da calibração - {nome_bacia}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_convergencia, dpi=150)
    plt.close()

    caminho_metricas = os.path.join(pasta_resultados, f"Resultados_estatisticos{sufixo_saida}.csv")
    caminho_pdf = os.path.join(pasta_resultados, f"Graficos_calibracao{sufixo_saida}.pdf")
    calcular_e_salvar_estatisticas(caminho_cawm, dados.get("intervalos"), caminho_metricas, caminho_pdf)

    melhor["pasta_resultados"] = pasta_resultados
    melhor["caminho_cawm"] = caminho_cawm
    melhor["caminho_convergencia"] = caminho_convergencia
    melhor["caminho_metricas"] = caminho_metricas
    melhor["caminho_pdf"] = caminho_pdf
    return melhor



def calcular_e_salvar_estatisticas(caminho_cawm, intervalo_df, caminho_csv_saida, caminho_pdf_saida):
    df = pd.read_csv(caminho_cawm)
    if df.columns[0] == "Unnamed: 0":
        df = df.drop(columns=[df.columns[0]])

    df["vazao"] = pd.to_numeric(df["vazao"], errors="coerce").fillna(0.0)
    df["vazao_calc"] = pd.to_numeric(df["vazao_calc"], errors="coerce").fillna(0.0)
    df["chuva_media"] = pd.to_numeric(df["chuva_media"], errors="coerce").fillna(0.0)

    if intervalo_df is None or intervalo_df.empty:
        intervalos = pd.DataFrame([{"intervalo": "total", "inicio": 1, "fim": len(df)}])
    else:
        intervalos = intervalo_df.copy().reset_index(drop=True)
        if len(intervalos) == 1:
            intervalos.loc[0, "intervalo"] = "calibracao"
        else:
            rotulos = ["calibracao", "validacao"]
            intervalos["intervalo"] = [rotulos[i] if i < len(rotulos) else f"intervalo_{i + 1}" for i in range(len(intervalos))]

    linhas = []
    with PdfPages(caminho_pdf_saida) as pp:
        for _, linha in intervalos.iterrows():
            inicio = int(linha["inicio"])
            fim = int(linha["fim"])
            mascara = (df.index + 1 >= inicio) & (df.index + 1 <= fim)
            sub = df.loc[mascara, ["vazao", "vazao_calc", "chuva_media"]].copy()
            metricas = calcular_indices_estatisticos(sub["vazao_calc"], sub["vazao"])
            metricas["intervalo"] = linha.get("intervalo", f"{inicio}-{fim}")
            metricas["inicio"] = inicio
            metricas["fim"] = fim
            linhas.append(metricas)

            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(111)
            lns1 = ax.plot(sub.index + 1, sub["vazao"], label="Qobs m³/s", color="red")
            lns2 = ax.plot(sub.index + 1, sub["vazao_calc"], label="Qcalc m³/s", color="blue")
            ax2 = ax.twinx()
            lns3 = ax2.plot(sub.index + 1, -1 * sub["chuva_media"], label="chuva", color="green")
            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)
            ax.set_xlabel("Dias corridos")
            ax.set_ylabel("Vazao")
            ax2.set_ylabel("Chuva")
            ax.set_title(f"CAWM: intervalo {inicio} a {fim}")
            pp.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        inicio = int(intervalos["inicio"].min())
        fim = int(intervalos["fim"].max())
        mascara = (df.index + 1 >= inicio) & (df.index + 1 <= fim)
        sub = df.loc[mascara, ["vazao", "vazao_calc", "chuva_media"]].copy()
        metricas = calcular_indices_estatisticos(sub["vazao_calc"], sub["vazao"])
        metricas["intervalo"] = "total"
        metricas["inicio"] = inicio
        metricas["fim"] = fim
        linhas.append(metricas)

    df_saida = pd.DataFrame(linhas, columns=["intervalo", "inicio", "fim", "Nash", "Nash_log", "Nash_sqrt", "RMS", "Pbias"])
    df_saida.to_csv(caminho_csv_saida, index=False, float_format="%.4f")
    return df_saida


def mapear_e_verificar_bacias(pastas_base):
    """Varre as pastas base e retorna apenas as bacias com arquivos completos."""
    bacias_validas = {}

    for pasta_base in pastas_base:
        if not os.path.exists(pasta_base):
            print(f"⚠️ Pasta base não encontrada: {pasta_base}")
            continue

        print(f"\n📁 Analisando bacias em: {os.path.basename(os.path.dirname(pasta_base))}")

        sub_bacias = [entry.path for entry in os.scandir(pasta_base) if entry.is_dir()]

        for caminho_bacia in sub_bacias:
            nome_bacia = os.path.basename(caminho_bacia)
            
            arq_intervalo = os.path.join(caminho_bacia, "intervalo.csv")
            arq_precip = os.path.join(caminho_bacia, "precipitacao.csv")
            arq_evap = os.path.join(caminho_bacia, "evaporacao.csv")
            arq_pao = os.path.join(caminho_bacia, "pao.csv")

            pasta_shapes = os.path.join(caminho_bacia, "shapes")
            arquivos_shp = glob.glob(os.path.join(pasta_shapes, "*.shp"))
            arq_shape = arquivos_shp[0] if arquivos_shp else None

            faltando = []
            if not os.path.exists(arq_intervalo):
                faltando.append("intervalo.csv")
            if not os.path.exists(arq_precip):
                faltando.append("precipitacao.csv")
            if not os.path.exists(arq_evap):
                faltando.append("evaporacao.csv")
            if not os.path.exists(arq_pao):
                faltando.append("pao.csv")
            if not arq_shape:
                faltando.append("Arquivo .shp na pasta shapes/")

            if faltando:
                print(f"  ❌ {nome_bacia}: Ignorado. Faltam -> {', '.join(faltando)}")
            else:
                print(f"  ✅ {nome_bacia}: Todos os arquivos encontrados!")
                bacias_validas[nome_bacia] = {
                    "intervalo": arq_intervalo,
                    "precipitacao": arq_precip,
                    "evaporacao": arq_evap,
                    "vazao": arq_pao,
                    "shape": arq_shape,
                    "pasta_bacia" : caminho_bacia
                }

    return bacias_validas

def carregar_series_temporais(nome_bacia, caminhos, xp):
    """
    Versão 2.1: Correção do erro 'Series object has no attribute month'
    e ajuste para leitura robusta de colunas.
    """
    try:
        # 1. Lendo o intervalo de calibração
        df_intervalo = pd.read_csv(caminhos["intervalo"], decimal='.', float_precision='high')
        dia_inicio = int(df_intervalo['inicio'].iloc[0])
        dia_fim = int(df_intervalo['fim'].iloc[0])
        
        # 2. Lendo as séries temporais (Proteção contra ponto/vírgula decimal)
        df_pao = pd.read_csv(caminhos["vazao"], decimal='.', thousands=None, float_precision='high')
        df_precip = pd.read_csv(caminhos["precipitacao"], decimal='.', thousands=None, float_precision='high')
        df_evap = pd.read_csv(caminhos["evaporacao"], header=None, decimal='.',thousands=None, float_precision='high')
        
        # 3. Localizando as colunas de dados (Ignorando a coluna de data)
        def _pick_value_col(df):
            for c in df.columns:
                low = c.lower()
                if not any(k in low for k in ('data', 'date', 'dia')):
                    return c
            # fallback: primeiro numérico, senão a primeira coluna disponível
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            return numeric[0] if numeric else df.columns[0]

        col_vazao = _pick_value_col(df_pao)
        col_chuva = _pick_value_col(df_precip)
        # 4. Tratamento de Datas e alinhamento entre vazão e precipitação
        col_date_pao = df_pao.columns[0]
        col_date_precip = df_precip.columns[0]
        df_pao['_date_'] = pd.to_datetime(df_pao[col_date_pao], dayfirst=False, errors='coerce')
        df_precip['_date_'] = pd.to_datetime(df_precip[col_date_precip], dayfirst=False, errors='coerce')

        # Merge inner para garantir alinhamento; se houver discrepância, avisamos
        df_merged = pd.merge(df_pao[["_date_", col_vazao]], df_precip[["_date_", col_chuva]], on="_date_", how="inner")
        if len(df_merged) != len(df_pao) or len(df_merged) != len(df_precip):
            print(f"  ⚠️ {nome_bacia}: discrepância de datas entre vazão ({len(df_pao)}) e precipitação ({len(df_precip)}). Usando interseção de {len(df_merged)} dias.")

        if df_merged['_date_'].isnull().any():
            df_merged = df_merged.dropna(subset=['_date_'])

        # Mapeia a evaporação mensal (1-12) para cada dia da série combinada
        meses = df_merged['_date_'].dt.month.astype(int)
        dict_evap = dict(zip(df_evap.iloc[:, 0].astype(int), df_evap.iloc[:, 1]))
        evaporacao_diaria = meses.map(dict_evap)

        # 5. Conversão para Array (GPU ou CPU) com a série alinhada
        vazao_obs = xp.asarray(df_merged[col_vazao].fillna(0.0).values, dtype=xp.float32)
        chuva = xp.asarray(df_merged[col_chuva].fillna(0.0).values, dtype=xp.float32)
        evaporacao = xp.asarray(evaporacao_diaria.fillna(0.0).values, dtype=xp.float32)

        # Usar a coluna de data unificada criada pelo merge
        datas_texto = df_merged['_date_'].dt.strftime('%Y-%m-%d').values

        return vazao_obs, chuva, evaporacao, datas_texto, df_intervalo
        
    except Exception as e:
        print(f"  ❌ Erro ao carregar séries de {nome_bacia}: {e}")
        return None, None, None, None, None


def validar_bacias_antes_pso(dicionario_bacias):
    """
    Verifica rapidamente cada bacia para garantir que os arquivos CSV
    possuem colunas esperadas, que as datas fazem sentido e que a
    interseção entre vazão e precipitação não é vazia.
    Retorna (validas, invalidas) onde cada um é um dict nome_bacia->motivo/dados.
    """
    validas = {}
    invalidas = {}

    for nome_bacia, caminhos in dicionario_bacias.items():
        motivo = []
        try:
            # Verifica existência dos arquivos
            for chave in ("intervalo", "precipitacao", "evaporacao", "vazao"):
                if chave not in caminhos or not os.path.exists(caminhos[chave]):
                    motivo.append(f"Arquivo faltando: {chave}")

            if motivo:
                invalidas[nome_bacia] = {'motivo': '; '.join(motivo)}
                continue

            # Leitura leve com pandas (sem conversão para xp)
            df_intervalo = pd.read_csv(caminhos['intervalo'], decimal='.', float_precision='high')
            df_pao = pd.read_csv(caminhos['vazao'], decimal='.', thousands=None, float_precision='high')
            df_precip = pd.read_csv(caminhos['precipitacao'], decimal='.', thousands=None, float_precision='high')
            df_evap = pd.read_csv(caminhos['evaporacao'], header=None, decimal='.', thousands=None, float_precision='high')

            # Identifica colunas de valor
            def _pick_value_col_df(df):
                for c in df.columns:
                    low = c.lower()
                    if not any(k in low for k in ('data', 'date', 'dia')):
                        return c
                numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                return numeric[0] if numeric else df.columns[0]

            col_vazao = _pick_value_col_df(df_pao)
            col_chuva = _pick_value_col_df(df_precip)
            col_date_pao = df_pao.columns[0]
            col_date_precip = df_precip.columns[0]

            # Parse de datas
            df_pao['_date_'] = pd.to_datetime(df_pao[col_date_pao], dayfirst=False, errors='coerce')
            df_precip['_date_'] = pd.to_datetime(df_precip[col_date_precip], dayfirst=False, errors='coerce')

            if df_pao['_date_'].isnull().all():
                motivo.append('Todas as datas de vazão inválidas')
            if df_precip['_date_'].isnull().all():
                motivo.append('Todas as datas de precipitação inválidas')

            # Interseção
            df_merged = pd.merge(df_pao[['_date_', col_vazao]], df_precip[['_date_', col_chuva]], on='_date_', how='inner')
            if df_merged.empty:
                motivo.append('Interseção vazia entre vazão e precipitação')

            # Evap: checar se há meses 1..12
            meses_evap = set(df_evap.iloc[:, 0].astype(int).unique())
            if not meses_evap.intersection(set(range(1, 13))):
                motivo.append('Evaporação sem meses 1..12')

            # Intervalo de calibração
            try:
                inicio = int(df_intervalo['inicio'].iloc[0])
                fim = int(df_intervalo['fim'].iloc[0])
                if inicio < 0 or fim <= inicio:
                    motivo.append('Intervalo inválido (inicio/fim)')
            except Exception:
                motivo.append('Intervalo CSV inválido')

            if motivo:
                invalidas[nome_bacia] = {'motivo': '; '.join(motivo)}
            else:
                validas[nome_bacia] = {'n_dias': len(df_merged)}

        except Exception as e:
            invalidas[nome_bacia] = {'motivo': f'Erro leitura: {e}'}

    return validas, invalidas

def extrair_parametros_shapes(dicionario_bacias):
    bacias_com_parametros = {}
    print("\n🗺️ Extraindo parâmetros (Cascata: Shapefile -> CSV -> Excel -> Tolerância)...")

    for nome_bacia, caminhos in dicionario_bacias.items():
        caminho_shp = caminhos["shape"]
        pasta_bacia = caminhos["pasta_bacia"]
        
        try:
            # =================================================================
            # 1. TENTATIVA DE LEITURA DO SHAPEFILE
            # =================================================================
            gdf = gpd.read_file(caminho_shp)
            atributos = gdf.iloc[0]
            colunas_shape = [c.lower() for c in gdf.columns]
            
            def buscar_coluna_shape(opcoes, tipo=float):
                for op in opcoes:
                    if op in colunas_shape:
                        col_exata = gdf.columns[colunas_shape.index(op)]
                        valor = atributos[col_exata]
                        if pd.notna(valor):
                            return tipo(valor)
                return None

            params = {}
            
            # Cálculo de Área
            area_km2 = buscar_coluna_shape(["area_km", "area_km2", "area", "areakm", "areakm2"])
            if area_km2 is None or area_km2 < 0.1:
                if gdf.crs is None:
                    gdf.set_crs(epsg=4674, inplace=True)
                gdf_proj = gdf.to_crs(epsg=5880)
                area_km2 = gdf_proj.geometry.area.sum() / 1_000_000.0
                
                if area_km2 < 0.1:
                    area_km2 = None  # Marca como None para tentar resgate no CSV depois
            
            params["area_km2"] = area_km2
            if area_km2 < 6000:
                params["k"] = 0.3745 * (area_km2 ** -0.489) + 0.0146
            elif area_km2 <= 60000:
                params["k"] = 34.343 * (area_km2 ** -0.853)
            else:
                params["k"] = 0.0028

            # Extração Inicial do Shapefile
            params["a"] = buscar_coluna_shape(["a", "evap_a"])
            params["expo_perdas"] = buscar_coluna_shape(["expo_perdas", "exp_perdas", "expoente_perdas", "p"])
            params["beta"] = buscar_coluna_shape(["beta"])
            params["kg"] = buscar_coluna_shape(["kg"])
            params["rio"] = buscar_coluna_shape(["rio", "tipo"], int)
            params["submax"] = buscar_coluna_shape(["submax", "s"])
            params["gmax"] = buscar_coluna_shape(["gmax"])
            params["reserva_solo_inicial"] = buscar_coluna_shape(["reserva_solo_inicial", "res_solo_ini", "wm", "reser_solo"])
            params["profundo_inicial"] = buscar_coluna_shape(["profundo_inicial", "prof_ini", "profundo"])
            params["s3_inicial"] = buscar_coluna_shape(["reserva", "reserva_calha", "s3", "rese_calha"])
            
            # Constantes fixas
            params["s1_inicial"] = 0.0
            params["s2_inicial"] = 0.0
            params["b"] = 1.666666667
            params["T"] = 86400.0

            nomes_exibicao = {
                "a": "a (Evap)", "expo_perdas": "Expoente(p)", "beta": "Beta", "kg": "Kg",
                "rio": "Tipo Rio", "submax": "SUBmax", "gmax": "Gmax", 
                "reserva_solo_inicial": "Wm (Res. Inicial)", "profundo_inicial": "Profundo Ini.", 
                "s3_inicial": "S3 (Calha Ini.)"
            }

            # =================================================================
            # 2. RESGATE VIA parametros_de_entrada.csv
            # =================================================================
            faltando_no_shape = [k for k, v in params.items() if v is None]
            
            if faltando_no_shape:
                caminho_csv_params = os.path.join(pasta_bacia, "parametros_de_entrada.csv")
                
                if os.path.exists(caminho_csv_params):
                    try:
                        df_csv = pd.read_csv(caminho_csv_params, header=None, sep=None, engine='python')
                        
                        termos_csv = {
                            "a": ["mult"],
                            "expo_perdas": ["coeficiente_do_expoente"],
                            "beta": ["beta"],
                            "kg": ["fração", "kg", "k"],
                            "rio": ["rio"],
                            "submax": ["submax"],
                            "gmax": ["gmax"],
                            "reserva_solo_inicial": ["reserv_solo"],
                            "profundo_inicial": ["profundo_corrigido"],
                            "s3_inicial": ["reserva_na_calha"],
                            "area_km2": ["Area_subbacia", "area"]
                        }
                        
                        for idx, row in df_csv.iterrows():
                            if pd.isna(row[0]) or pd.isna(row[1]):
                                continue
                                
                            chave_limpa = str(row[0]).lower().replace('=', '').strip()
                            
                            try:
                                valor_limpo = float(str(row[1]).replace(',', '.'))
                            except ValueError:
                                continue
                                
                            for var_chave in faltando_no_shape:
                                if params[var_chave] is None:
                                    termos_busca = termos_csv.get(var_chave, [])
                                    achou = False
                                    for termo in termos_busca:
                                        # Normaliza o termo para lowercase para comparação
                                        termo_lower = termo.lower()
                                        if termo_lower == "k" and chave_limpa == "k":
                                            achou = True
                                        elif termo_lower != "k" and termo_lower in chave_limpa:
                                            achou = True
                                            
                                    if achou:
                                        if var_chave == "rio":
                                            valor_limpo = int(valor_limpo)
                                        params[var_chave] = valor_limpo
                                        # Imprime na tela o resgate
                                        print(f"      📄 CSV Resgate: '{nomes_exibicao.get(var_chave, var_chave)}' -> {valor_limpo}")
                    except Exception as e:
                        print(f"      ⚠️ Falha ao ler parametros_de_entrada.csv: {e}")

            # =================================================================
            # 3. RESGATE VIA EXCEL (.xlsm / .xlsx) 
            # =================================================================
            ainda_faltando = [k for k, v in params.items() if v is None]
            
            if ainda_faltando:
                termos_excel = {
                    "a": ["taxa de evapotranspiração", "parâmetro a", "multiplicador da taxa", "coeficiente da função de evapotranspiração"],
                    "expo_perdas": ["expoente da função de perdas", "coeficiente do expoente das perdas"],
                    "beta": ["beta", "parâmetro de infiltração", "parâmetro de recarga"],
                    "kg": ["fração da percolação", "kg", "parâmetro de percolação"],
                    "rio": ["rio 0-temporário", "rios temporários", "tipo rio"],
                    "submax": ["capacidade máxima de armazenamento do solo", "capacidade de armazenamento no solo", "submax"],
                    "gmax": ["profundidade máxima do reservatório profundo", "capacidade de armazenamento percolação profunda", "gmax"],
                    "reserva_solo_inicial": ["reserva de solo", "wm", "reserva no solo inicial"],
                    "profundo_inicial": ["profundo corrigido inicial", "reservatório profundo", "inicial", "profundo corrigido"],
                    "s3_inicial": ["reserva na calha", "s3"],
                    "area_km2": ["área", "area_km2", "area_subbacia", "área da bacia"]
                }

                arquivos_excel = glob.glob(os.path.join(pasta_bacia, "*.xlsm")) + glob.glob(os.path.join(pasta_bacia, "*.xlsx"))
                
                if arquivos_excel:
                    arquivo_alvo = arquivos_excel[0]
                    try:
                        df_excel = pd.read_excel(arquivo_alvo, sheet_name=0, header=None)
                        
                        def buscar_valor_na_matriz(termos_busca):
                            # Tenta primeiro procurar na coluna A (com B ao lado)
                            for idx, row in df_excel.iterrows():
                                col_a = row.iloc[0] if len(row) > 0 else None
                                col_b = row.iloc[1] if len(row) > 1 else None
                                
                                if pd.notna(col_a) and isinstance(col_a, str):
                                    col_a_lower = col_a.lower()
                                    if any(termo.lower() in col_a_lower for termo in termos_busca):
                                        if pd.notna(col_b) and isinstance(col_b, (int, float)):
                                            return float(col_b)
                            
                            # Se não encontrar em A-B, tenta na mesma linha em outras colunas
                            for idx, row in df_excel.iterrows():
                                for col_idx, cell in enumerate(row):
                                    if pd.notna(cell) and isinstance(cell, str):
                                        cell_lower = cell.lower()
                                        if any(termo.lower() in cell_lower for termo in termos_busca):
                                            for next_col in range(col_idx + 1, len(row)):
                                                if pd.notna(row[next_col]) and isinstance(row[next_col], (int, float)):
                                                    return float(row[next_col])
                            return None

                        for var_chave in ainda_faltando:
                            valor_resgatado = buscar_valor_na_matriz(termos_excel[var_chave])
                            if valor_resgatado is not None:
                                if var_chave == "rio":
                                    valor_resgatado = int(valor_resgatado)
                                # Se beta for 0, usar o valor padrão de 2
                                if var_chave == "beta" and valor_resgatado == 0:
                                    valor_resgatado = 2
                                params[var_chave] = valor_resgatado
                                print(f"      🛟 Excel Resgate: '{nomes_exibicao.get(var_chave, var_chave)}' -> {valor_resgatado}")

                    except Exception as e:
                        print(f"      ⚠️ Falha ao ler o arquivo Excel: {e}")

            # =================================================================
            # 4. VALIDAÇÃO FINAL DE ÁREA
            # =================================================================
            if params["area_km2"] is None or params["area_km2"] < 0.1:
                raise ValueError(f"Área não encontrada ou inválida ({params.get('area_km2', 'None')})")

            # =================================================================
            # 5. TOLERÂNCIA E VALORES PADRÃO (Plano D)
            # =================================================================
            # Se depois de tudo o Kg ainda faltar, assumimos 1.0
            if params["kg"] is None:
                params["kg"] = 1.0
                print(f"      🔧 Tolerância Aplicada: 'Kg' forçado para 1.0")
            
            # Se depois de tudo o Beta ainda faltar, assumimos 2.0
            if params["beta"] is None:
                params["beta"] = 2.0
                print(f"      🔧 Tolerância Aplicada: 'Beta' forçado para 2.0")

            # =================================================================
            # 6. VEREDITO FINAL DA BACIA
            # =================================================================
            resultado_final_faltando = [nomes_exibicao[k] for k, v in params.items() if v is None]
            
            if resultado_final_faltando:
                raise ValueError(f"Dados não encontrados (Shape/CSV/Excel): [{', '.join(resultado_final_faltando)}]")
            else:
                dicionario_bacias[nome_bacia]["parametros"] = params
                bacias_com_parametros[nome_bacia] = params
                print(f"  ✅ {nome_bacia}: Aprovada e Carregada! (Área: {params['area_km2']:.1f}km²)")
            
        except Exception as e:
            print(f"  ❌ {nome_bacia} IGNORADA: {e}")
            
    return bacias_com_parametros
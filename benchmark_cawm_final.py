import time
import gc
from typing import Any
import numpy as np
import pandas as pd

cp: Any = None
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False
    print("Aviso: CuPy não encontrado. Rodando em NumPy (Matricial CPU).")

jax: Any = None
jnp: Any = None
try:
    import jax  
    import jax.numpy as jnp  
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False
    print("Aviso: JAX não encontrado. O método GPU_JAX ficará indisponível.")

from db_init import initialize_db
from db_models import Bacia, PrecipitationDaily, EvaporationMonthly, FlowDaily, CalibrationPeriod

# Constante para seleção da Função Objetivo (FO)
# Opções suportadas: 'fo1' (padrão), 'fo2'/'composite' (NSE + NSE_sqrt), 'kge'/'fo_kge'
FO_SELECTION = 'fo1'


def limpar_memoria_gpu():
    if not HAS_GPU:
        return
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

# ==============================================================================
# 1. FUNÇÃO DE EXTRAÇÃO DE DADOS POR CALIBRATION_ID
# ==============================================================================
def extrair_dados_por_id(calib_id: int):
    session = initialize_db()
    if isinstance(session, tuple):
        for item in session:
            if hasattr(item, 'query'):
                session = item
                break
            elif callable(item): 
                session = item()
                if hasattr(session, 'query'):
                    break
    elif callable(session):
        session = session()

    periodo = session.query(CalibrationPeriod).filter_by(id=calib_id).first()
    if not periodo:
        raise ValueError(f"Calibration ID {calib_id} não encontrado.")
        
    bacia = session.query(Bacia).filter_by(id=periodo.bacia_id).first()
    
    nome_bacia = bacia.nome
    area_km2 = bacia.area_km2
    SUBmax = bacia.submax if bacia.submax is not None else 1000.0
    a_param = 1.0 
    
    calib_start = pd.Timestamp(periodo.calib_start)
    calib_end = pd.Timestamp(periodo.calib_end)
    valid_start = pd.Timestamp(getattr(periodo, "valid_start", None) or getattr(periodo, "val_start", None))
    valid_end = pd.Timestamp(getattr(periodo, "valid_end", None) or getattr(periodo, "val_end", None))
    metodo = getattr(periodo, "metodo", None) or getattr(periodo, "method", None)

    precip = session.query(PrecipitationDaily).filter_by(bacia_id=bacia.id).order_by(PrecipitationDaily.data).all()
    vazao = session.query(FlowDaily).filter_by(bacia_id=bacia.id).order_by(FlowDaily.data).all()
    evap_mensal = session.query(EvaporationMonthly).filter_by(bacia_id=bacia.id).all()
    session.close()
    
    df_chuva = pd.DataFrame({"data": [p.data for p in precip], "chuva": [float(p.valor) for p in precip]})
    df_vazao = pd.DataFrame({"data": [q.data for q in vazao], "vazao": [float(q.valor) for q in vazao]})
    df = pd.merge(df_vazao, df_chuva, on="data", how="inner").sort_values("data")
    
    mapa_evap = {int(e.mes): float(e.valor) for e in evap_mensal}
    df["data_dt"] = pd.to_datetime(df["data"])
    df["mes"] = df["data_dt"].dt.month
    df["evap"] = df["mes"].map(mapa_evap).fillna(5.0)
    
    mask_calib = ((df["data_dt"] >= calib_start) & (df["data_dt"] <= calib_end)).to_numpy()
    mask_valid = ((df["data_dt"] >= valid_start) & (df["data_dt"] <= valid_end)).to_numpy()
    
    P = df["chuva"].to_numpy(dtype=np.float32)
    E = df["evap"].to_numpy(dtype=np.float32)
    Q_obs = df["vazao"].to_numpy(dtype=np.float32)
    
    return P, E, Q_obs, mask_calib, mask_valid, area_km2, SUBmax, a_param, nome_bacia, metodo

# ==============================================================================
# 2. MOTOR DO CAWM VETORIZADO PARA O PSO
# ==============================================================================
def simular_cawm_vetorizado(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, ks_array, expo_array, fo='fo1'):
    dias = len(P)
    particulas = len(ks_array)
    b = 1.666666667
    T_sec = 86400.0
    
    if area < 6000: k_calha = 0.3745 * (area**-0.489) + 0.0146
    elif area <= 60000: k_calha = 34.343 * (area**-0.853)
    else: k_calha = 0.0028
        
    F_conversao = (area * 1000000.0 / T_sec) / 1000.0
    mask_xp = xp.asarray(mask_calib)
    Q_obs_masked = Q_obs[mask_xp]
    vol_obs_mm = xp.sum(Q_obs_masked) / F_conversao
    n_masked = int(xp.sum(mask_xp))

    def executar_passagem(calcular_erro: bool = False, kl_array=None):
        ret_corrig = xp.zeros(particulas, dtype=xp.float32)
        reserv_solo_corrig = xp.zeros(particulas, dtype=xp.float32)
        S3 = xp.zeros(particulas, dtype=xp.float32)

        soma_C_masked = xp.zeros(particulas, dtype=xp.float32)
        soma_C_expo_masked = xp.zeros(particulas, dtype=xp.float32)
        numerador_nash = xp.zeros(particulas, dtype=xp.float32)
        soma_qcalc_masked = xp.zeros(particulas, dtype=xp.float32)

        # Acumuladores para estatísticas evitando armazenar matriz completa (otimização de memória)
        qcalc_masked = None
        sum_qcalc = None
        sum_qcalc2 = None
        sum_qcalc_qobs = None
        sum_sqrtqcalc_sqrtqobs = None
        if calcular_erro and fo in ('fo2', 'composite'):
            sum_qcalc = xp.zeros(particulas, dtype=xp.float32)
            sum_qcalc2 = xp.zeros(particulas, dtype=xp.float32)
            sum_qcalc_qobs = xp.zeros(particulas, dtype=xp.float32)
            sum_sqrtqcalc_sqrtqobs = xp.zeros(particulas, dtype=xp.float32)
        elif calcular_erro and fo in ('kge', 'fo_kge'):
            sum_qcalc = xp.zeros(particulas, dtype=xp.float32)
            sum_qcalc2 = xp.zeros(particulas, dtype=xp.float32)
            sum_qcalc_qobs = xp.zeros(particulas, dtype=xp.float32)

        idx_mask = 0
        for d in range(dias):
            P_d, E_d = P[d], E[d]
            evap_inicial = xp.where(ret_corrig + P_d >= E_d, E_d, ret_corrig + P_d)
            ret_corrig = xp.maximum(ret_corrig + P_d - evap_inicial, 0.0)
            evap_n_atendida = E_d - evap_inicial

            Pn_pos = xp.maximum(P_d - evap_inicial, 0.0)
            hiperb = xp.tanh(Pn_pos / SUBmax)
            reserv_solo = xp.minimum(reserv_solo_corrig, SUBmax)
            Sub = reserv_solo / SUBmax

            Ps = (SUBmax * (1.0 - Sub**2) * hiperb) / (1.0 + Sub * hiperb + 1e-9)
            escoamento = P_d - evap_inicial - Ps

            E_comp = (1.0 - xp.exp(-a_param * (reserv_solo / SUBmax))) * evap_n_atendida
            RE = xp.minimum(xp.minimum(evap_n_atendida, reserv_solo), E_comp)
            Solo = xp.maximum(reserv_solo - RE, 0.0)
            rec_rio = ks_array * Solo

            S1 = xp.maximum(S3 + escoamento + rec_rio, 0.0)
            C = xp.minimum(k_calha * (S1 ** b), S1)
            S3 = S1 - C
            reserv_solo_corrig = xp.maximum(Solo + Ps - rec_rio, 0.0)

            C_expo = C ** expo_array
            if mask_calib[d]:
                soma_C_masked = soma_C_masked + C
                soma_C_expo_masked = soma_C_expo_masked + C_expo
                if calcular_erro:
                    Q_calc_d = (C - xp.minimum(kl_array * C_expo, C)) * F_conversao
                    soma_qcalc_masked = soma_qcalc_masked + Q_calc_d
                    # Acumula estatísticas por partícula para FO alternativa (sem armazenar matriz completa)
                    if sum_qcalc is not None:
                        sum_qcalc = sum_qcalc + Q_calc_d
                        sum_qcalc2 = sum_qcalc2 + (Q_calc_d * Q_calc_d)
                        sum_qcalc_qobs = sum_qcalc_qobs + (Q_calc_d * Q_obs_masked[idx_mask])
                    if sum_sqrtqcalc_sqrtqobs is not None:
                        sum_sqrtqcalc_sqrtqobs = sum_sqrtqcalc_sqrtqobs + xp.sqrt(xp.maximum(Q_calc_d, 0.0)) * xp.sqrt(xp.maximum(Q_obs_masked[idx_mask], 0.0))
                    erro_d = Q_obs_masked[idx_mask] - Q_calc_d
                    numerador_nash = numerador_nash + (erro_d ** 2)
                    idx_mask += 1

        if calcular_erro:
            denominador_nash = xp.sum((Q_obs_masked - xp.mean(Q_obs_masked))**2) + 1e-9
            nash_array = 1.0 - (numerador_nash / denominador_nash)
            # Retorna também os acumuladores para cálculo de FO sem a matriz completa
            return nash_array, soma_qcalc_masked, (sum_qcalc, sum_qcalc2, sum_qcalc_qobs, sum_sqrtqcalc_sqrtqobs)

        return soma_C_masked, soma_C_expo_masked, None

    soma_C_masked, soma_C_expo_masked, _ = executar_passagem(calcular_erro=False)
    kl_array = xp.maximum((soma_C_masked - vol_obs_mm) / (soma_C_expo_masked + 1e-9), 0.0)
    nash_array, soma_qcalc_masked, accumulators = executar_passagem(calcular_erro=True, kl_array=kl_array)

    # Seleciona a função objetivo conforme solicitado (usando acumuladores quando possível)
    if fo in ('fo1', 'default'):
        fo_array = calcular_fo(xp, nash_array, soma_qcalc_masked, xp.sum(Q_obs_masked))
    elif fo in ('fo2', 'composite'):
        sums = accumulators
        if sums[0] is None:
            fo_array = calcular_fo(xp, nash_array, soma_qcalc_masked, xp.sum(Q_obs_masked))
        else:
            sum_qcalc, sum_qcalc2, sum_qcalc_qobs, sum_sqrtqcalc_sqrtqobs = sums
            # Denominadores e constantes
            media_obs = xp.mean(Q_obs_masked)
            denom_nse = xp.sum((Q_obs_masked - media_obs)**2) + 1e-9

            # NSE clássico já disponível como nash_array (a partir do numerador acumulado), mas recalculamos por consistência
            numerador_nse = sum_qcalc2 - 2.0 * sum_qcalc_qobs + xp.sum(Q_obs_masked * Q_obs_masked)
            nse = 1.0 - (numerador_nse / denom_nse)

            # NSE raiz quadrada
            qobs_sqrt = xp.sqrt(xp.maximum(Q_obs_masked, 0.0))
            media_obs_sqrt = xp.mean(qobs_sqrt)
            denom_sqrt = xp.sum((qobs_sqrt - media_obs_sqrt)**2) + 1e-9
            numerador_sqrt = sum_qcalc - 2.0 * sum_sqrtqcalc_sqrtqobs + xp.sum(Q_obs_masked)
            nse_sqrt = 1.0 - (numerador_sqrt / denom_sqrt)

            fo_array = (nse * 0.5) + (nse_sqrt * 0.5)
    elif fo in ('kge', 'fo_kge'):
        sums = accumulators
        if sums[0] is None:
            fo_array = calcular_fo(xp, nash_array, soma_qcalc_masked, xp.sum(Q_obs_masked))
        else:
            sum_qcalc, sum_qcalc2, sum_qcalc_qobs, _ = sums
            n = float(n_masked)
            media_obs = xp.mean(Q_obs_masked)
            std_obs = xp.std(Q_obs_masked) + 1e-9

            media_sim = sum_qcalc / n
            var_sim = (sum_qcalc2 / n) - (media_sim ** 2)
            std_sim = xp.sqrt(xp.maximum(var_sim, 1e-12))
            cov = (sum_qcalc_qobs / n) - (media_sim * media_obs)
            r = cov / (std_sim * std_obs + 1e-9)
            alpha = std_sim / (std_obs + 1e-9)
            beta = (sum_qcalc / n) / (media_obs + 1e-9)
            fo_array = 1.0 - xp.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    else:
        fo_array = calcular_fo(xp, nash_array, soma_qcalc_masked, xp.sum(Q_obs_masked))

    return fo_array, nash_array


def calcular_fo(xp, nash_array, soma_qcalc_masked, soma_qobs_masked, eps=1e-9, scale=1e6):
    """Calcula o FO a partir do NSE (nash_array) e das somas de Q_calc e Q_obs.

    FO = (NSE) / (|sum(Q_calc_masked) - sum(Q_obs_masked)| + eps) * scale
    Usa operações do backend `xp` (numpy ou cupy).
    """
    soma_abs = xp.abs(soma_qcalc_masked - soma_qobs_masked) + eps
    return (nash_array / soma_abs) * scale

def calcular_fo2(xp, qcalc_masked, qobs_masked, eps=1e-9):
    """
    Calcula a FO Composta: 50% NSE Clássico + 50% NSE Raiz Quadrada.
    qcalc_masked: Matriz 2D (particulas, dias) - Vazões simuladas
    qobs_masked: Array 1D (dias) - Vazões observadas
    Usa operações do backend `xp` (numpy ou cupy).
    """
    # Média da série observada
    media_obs = xp.mean(qobs_masked)
    
    # ==========================================
    # 1. Cálculo do NSE Clássico (Foco nos picos)
    # ==========================================
    numerador_nse = xp.sum((qcalc_masked - qobs_masked)**2, axis=1)
    denominador_nse = xp.sum((qobs_masked - media_obs)**2) + eps
    nse = 1.0 - (numerador_nse / denominador_nse)
    
    # ==========================================
    # 2. Cálculo do NSE Raiz Quadrada (Foco nas médias/recessão)
    # ==========================================
    # O xp.maximum garante que pequenos erros de precisão do float não gerem raiz de negativo
    qcalc_sqrt = xp.sqrt(xp.maximum(qcalc_masked, 0.0))
    qobs_sqrt = xp.sqrt(xp.maximum(qobs_masked, 0.0))
    media_obs_sqrt = xp.mean(qobs_sqrt)
    
    numerador_sqrt = xp.sum((qcalc_sqrt - qobs_sqrt)**2, axis=1)
    denominador_sqrt = xp.sum((qobs_sqrt - media_obs_sqrt)**2) + eps
    nse_sqrt = 1.0 - (numerador_sqrt / denominador_sqrt)
    
    # ==========================================
    # 3. Retorno da Função Objetivo Composta
    # ==========================================
    # O PSO vai tentar maximizar esta média. 
    # O modelo é forçado a ser "bom nas cheias" E "bom na recessão".
    return (nse * 0.5) + (nse_sqrt * 0.5)

def calcular_fo_kge(xp, qcalc_masked, qobs_masked, eps=1e-9):
    """
    Calcula o Kling-Gupta Efficiency (KGE) adaptado para GPU.
    qcalc_masked: Matriz 2D (particulas x dias) -> A vazão simulada
    qobs_masked: Array 1D (dias) -> A vazão real observada
    """
    # 1. Média e Desvio Padrão Observado
    media_obs = xp.mean(qobs_masked)
    std_obs = xp.std(qobs_masked) + eps

    # 2. Média e Desvio Padrão Simulado (calculado por linha/partícula)
    media_sim = xp.mean(qcalc_masked, axis=1, keepdims=True)
    std_sim = xp.std(qcalc_masked, axis=1) + eps

    # 3. Correlação Linear de Pearson (r)
    # Mostra se o modelo sobe e desce no momento certo
    covariancia = xp.mean((qcalc_masked - media_sim) * (qobs_masked - media_obs), axis=1)
    r = covariancia / (std_sim * std_obs)

    # 4. Razão de Variabilidade (Alpha)
    # Evita que o modelo subestime ou superestime os picos em geral
    alpha = std_sim / std_obs

    # 5. Razão de Viés (Beta)
    # Garante o balanço do volume (que o seu Kl já faz, mas mantemos pela matemática)
    beta = xp.mean(qcalc_masked, axis=1) / (media_obs + eps)

    # 6. Cálculo Final do KGE
    # Calcula a distância euclidiana para o ponto perfeito (r=1, alpha=1, beta=1)
    kge = 1.0 - xp.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)

    # O PSO tentará aproximar este valor de 1.0
    return kge

# ==============================================================================
# 3. MOTOR FORWARD (Gera a série contínua para as Estatísticas da Professora)
# ==============================================================================
def simular_forward_determinista(P, E, Q_obs, mask_calib, area, SUBmax, a_param, best_ks, best_expo):
    dias = len(P)
    b = 1.666666667
    T_sec = 86400.0
    k_calha = 0.3745 * (area**-0.489) + 0.0146 if area < 6000 else (34.343 * (area**-0.853) if area <= 60000 else 0.0028)
    F_conversao = (area * 1000000.0 / T_sec) / 1000.0
    
    ret_corrig, reserv_solo_corrig, S3 = 0.0, 0.0, 0.0
    C_hist = np.zeros(dias, dtype=np.float32)
    C_expo_hist = np.zeros(dias, dtype=np.float32)

    for d in range(dias):
        P_d, E_d = P[d], E[d]
        evap_inicial = E_d if ret_corrig + P_d >= E_d else ret_corrig + P_d
        ret_corrig = max(ret_corrig + P_d - evap_inicial, 0.0)
        evap_n_atendida = E_d - evap_inicial

        Pn_pos = max(P_d - evap_inicial, 0.0)
        hiperb = np.tanh(Pn_pos / SUBmax)
        reserv_solo = min(reserv_solo_corrig, SUBmax)
        Sub = reserv_solo / SUBmax
        
        Ps = (SUBmax * (1.0 - Sub**2) * hiperb) / (1.0 + Sub * hiperb + 1e-9)
        escoamento = P_d - evap_inicial - Ps

        E_comp = (1.0 - np.exp(-a_param * (reserv_solo / SUBmax))) * evap_n_atendida
        RE = min(min(evap_n_atendida, reserv_solo), E_comp)
        Solo = max(reserv_solo - RE, 0.0)
        rec_rio = best_ks * Solo

        S1 = max(S3 + escoamento + rec_rio, 0.0)
        C = min(k_calha * (S1 ** b), S1)
        S3 = S1 - C  
        reserv_solo_corrig = max(Solo + Ps - rec_rio, 0.0)

        C_hist[d] = C
        C_expo_hist[d] = C ** best_expo

    # Kl Analítico calculado na calibração
    Q_obs_masked = Q_obs[mask_calib]
    vol_obs_mm = np.sum(Q_obs_masked) / F_conversao
    soma_C_masked = np.sum(C_hist[mask_calib])
    soma_C_expo_masked = np.sum(C_expo_hist[mask_calib])
    kl_calc = max((soma_C_masked - vol_obs_mm) / (soma_C_expo_masked + 1e-9), 0.0)

    # Q_calc para série inteira
    Q_calc_total = np.zeros(dias, dtype=np.float32)
    for d in range(dias):
        perdas = min(kl_calc * C_expo_hist[d], C_hist[d])
        Q_calc_total[d] = (C_hist[d] - perdas) * F_conversao
    
    return Q_calc_total, kl_calc

# ==============================================================================
# 4. ALGORITMOS DE OTIMIZAÇÃO (Legado, CPU Vetorizado e GPU Vetorizado)
# ==============================================================================
def pso_escalar_legado(P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2, fo='fo1'):
    X = np.random.rand(particulas, 2)
    X[:, 0] = X[:, 0] * 1.0
    X[:, 1] = X[:, 1] * 2.5 + 0.5
    V = np.zeros((particulas, 2))
    Pbest = np.copy(X)
    Pbest_nash = np.full(particulas, -9999.0)
    Gbest = np.zeros(2)
    Gbest_nash = -9999.0
    Gbest_nse = -9999.0
    it_sem_melhora = 0
    historico = []
    iteracao_convergencia = 0
    
    for it in range(iteracoes):
        houve_melhora = False
        nash_array = np.zeros(particulas)
        nse_array = np.zeros(particulas)
        for i in range(particulas):
            fo_val, nse_val = simular_cawm_vetorizado(np, P, E, Q_obs, mask_calib, area, SUBmax, a_param, np.array([X[i, 0]]), np.array([X[i, 1]]), fo=fo)
            nash_array[i] = round(float(fo_val[0]), 3)
            nse_array[i] = round(float(nse_val[0]), 3)
            if nash_array[i] > Pbest_nash[i]:
                Pbest_nash[i] = nash_array[i]
                Pbest[i] = X[i]
                if nash_array[i] > Gbest_nash:
                    Gbest_nash = nash_array[i]
                    Gbest_nse = nse_array[i]
                    Gbest = np.copy(X[i])
                    houve_melhora = True
                    iteracao_convergencia = it + 1
        
        it_sem_melhora = 0 if houve_melhora else it_sem_melhora + 1
        historico.append(Gbest_nash)
        print(f"      Fase 2 [Legado]: Iteração [{it+1:02d}/{iteracoes}] | FO: {Gbest_nash:8.3f} | NSE: {Gbest_nse:8.3f}", end="\r")
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break
            
        for i in range(particulas):
            r1, r2 = np.random.rand(2)
            V[i] = (w * V[i]) + (c1 * r1 * (Pbest[i] - X[i])) + (c2 * r2 * (Gbest - X[i]))
            X[i] = X[i] + V[i]
            X[i, 0] = np.clip(X[i, 0], 0.0, 1.0)
            X[i, 1] = np.clip(X[i, 1], 0.5, 3.0)
            
    return Gbest_nash, Gbest_nse, historico, float(Gbest[0]), float(Gbest[1]), iteracao_convergencia

def pso_vetorizado(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2, fo='fo1'):
    P_xp, E_xp, Q_obs_xp = xp.asarray(P), xp.asarray(E), xp.asarray(Q_obs)
    X = xp.random.rand(particulas, 2).astype(xp.float32)
    X[:, 0] = X[:, 0] * 1.0
    X[:, 1] = X[:, 1] * 2.5 + 0.5
    V = xp.zeros((particulas, 2), dtype=xp.float32)
    Pbest = xp.copy(X)
    Pbest_nash = xp.full(particulas, -9999.0, dtype=xp.float32)
    Gbest = xp.zeros(2, dtype=xp.float32)
    Gbest_nash = -9999.0
    Gbest_nse = -9999.0
    it_sem_melhora = 0
    historico = []
    iteracao_convergencia = 0
    
    for it in range(iteracoes):
        fo_array, nse_array = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X[:, 0], X[:, 1], fo=fo)
        nash_array = xp.round(fo_array, 3)
        nse_array = xp.round(nse_array, 3)
        melhorias = nash_array > Pbest_nash
        Pbest_nash = xp.where(melhorias, nash_array, Pbest_nash)
        Pbest = xp.where(melhorias[:, None], X, Pbest)
        idx_max = xp.argmax(Pbest_nash)
        max_nash = float(Pbest_nash[idx_max])
        
        if max_nash > Gbest_nash:
            Gbest_nash = max_nash
            Gbest_nse = float(nse_array[idx_max])
            Gbest = xp.copy(Pbest[idx_max])
            it_sem_melhora = 0
            iteracao_convergencia = it + 1
        else:
            it_sem_melhora += 1
            
        historico.append(Gbest_nash)
        print(f"      Fase 2 [Matricial]: Iteração [{it+1:02d}/{iteracoes}] | FO: {Gbest_nash:8.3f} | NSE: {Gbest_nse:8.3f}      ", end="\r")
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break
            
        R1, R2 = xp.random.rand(particulas, 2).astype(xp.float32), xp.random.rand(particulas, 2).astype(xp.float32)
        V = (w * V) + (c1 * R1 * (Pbest - X)) + (c2 * R2 * (Gbest - X))
        X = X + V
        X[:, 0] = xp.clip(X[:, 0], 0.0, 1.0)
        X[:, 1] = xp.clip(X[:, 1], 0.5, 3.0)
        
    return Gbest_nash, Gbest_nse, historico, float(Gbest[0]), float(Gbest[1]), iteracao_convergencia


if HAS_JAX:
    def _simular_cawm_jax_core(P, E, Q_obs, mask_calib, area, SUBmax, a_param, ks_array, expo_array, fo='fo1'):
        P = jnp.asarray(P, dtype=jnp.float32)
        E = jnp.asarray(E, dtype=jnp.float32)
        Q_obs = jnp.asarray(Q_obs, dtype=jnp.float32)
        mask_calib = jnp.asarray(mask_calib, dtype=bool)
        ks_array = jnp.asarray(ks_array, dtype=jnp.float32)
        expo_array = jnp.asarray(expo_array, dtype=jnp.float32)

        b = 1.666666667
        T_sec = 86400.0

        if area < 6000:
            k_calha = 0.3745 * (area**-0.489) + 0.0146
        elif area <= 60000:
            k_calha = 34.343 * (area**-0.853)
        else:
            k_calha = 0.0028

        F_conversao = (area * 1000000.0 / T_sec) / 1000.0
        mask_float = mask_calib.astype(jnp.float32)
        qobs_masked = jnp.where(mask_calib, Q_obs, 0.0)
        soma_qobs_masked = jnp.sum(qobs_masked)
        soma_qobs2_masked = jnp.sum(qobs_masked * qobs_masked)
        qobs_sqrt = jnp.sqrt(jnp.maximum(Q_obs, 0.0))
        qobs_sqrt_masked = jnp.where(mask_calib, qobs_sqrt, 0.0)
        soma_qobs_sqrt_masked = jnp.sum(qobs_sqrt_masked)
        soma_qobs_sqrt2_masked = jnp.sum(qobs_sqrt_masked * qobs_sqrt_masked)
        n_masked = jnp.sum(mask_float)

        def passagem_sem_erro(carry, inputs):
            ret_corrig, reserv_solo_corrig, S3, soma_C_masked, soma_C_expo_masked = carry
            P_d, E_d, mask_d = inputs

            evap_inicial = jnp.minimum(ret_corrig + P_d, E_d)
            ret_corrig = jnp.maximum(ret_corrig + P_d - evap_inicial, 0.0)
            evap_n_atendida = E_d - evap_inicial

            Pn_pos = jnp.maximum(P_d - evap_inicial, 0.0)
            hiperb = jnp.tanh(Pn_pos / SUBmax)
            reserv_solo = jnp.minimum(reserv_solo_corrig, SUBmax)
            Sub = reserv_solo / SUBmax

            Ps = (SUBmax * (1.0 - Sub**2) * hiperb) / (1.0 + Sub * hiperb + 1e-9)
            escoamento = P_d - evap_inicial - Ps

            E_comp = (1.0 - jnp.exp(-a_param * (reserv_solo / SUBmax))) * evap_n_atendida
            RE = jnp.minimum(jnp.minimum(evap_n_atendida, reserv_solo), E_comp)
            Solo = jnp.maximum(reserv_solo - RE, 0.0)
            rec_rio = ks_array * Solo

            S1 = jnp.maximum(S3 + escoamento + rec_rio, 0.0)
            C = jnp.minimum(k_calha * (S1 ** b), S1)
            S3 = S1 - C
            reserv_solo_corrig = jnp.maximum(Solo + Ps - rec_rio, 0.0)

            C_expo = C ** expo_array
            mask_d = mask_d.astype(jnp.float32)
            soma_C_masked = soma_C_masked + (mask_d * C)
            soma_C_expo_masked = soma_C_expo_masked + (mask_d * C_expo)

            return (ret_corrig, reserv_solo_corrig, S3, soma_C_masked, soma_C_expo_masked), None

        carry_inicial = (
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
        )
        (_, _, _, soma_C_masked, soma_C_expo_masked), _ = jax.lax.scan(
            passagem_sem_erro,
            carry_inicial,
            (P, E, mask_calib),
        )

        vol_obs_mm = jnp.sum(qobs_masked) / F_conversao
        kl_array = jnp.maximum((soma_C_masked - vol_obs_mm) / (soma_C_expo_masked + 1e-9), 0.0)

        def passagem_com_erro(carry, inputs):
            ret_corrig, reserv_solo_corrig, S3, soma_C_masked, soma_C_expo_masked, numerador_nash, soma_qcalc_masked, sum_qcalc, sum_qcalc2, sum_qcalc_qobs, sum_sqrtqcalc_sqrtqobs = carry
            P_d, E_d, mask_d, Q_obs_d = inputs

            evap_inicial = jnp.minimum(ret_corrig + P_d, E_d)
            ret_corrig = jnp.maximum(ret_corrig + P_d - evap_inicial, 0.0)
            evap_n_atendida = E_d - evap_inicial

            Pn_pos = jnp.maximum(P_d - evap_inicial, 0.0)
            hiperb = jnp.tanh(Pn_pos / SUBmax)
            reserv_solo = jnp.minimum(reserv_solo_corrig, SUBmax)
            Sub = reserv_solo / SUBmax

            Ps = (SUBmax * (1.0 - Sub**2) * hiperb) / (1.0 + Sub * hiperb + 1e-9)
            escoamento = P_d - evap_inicial - Ps

            E_comp = (1.0 - jnp.exp(-a_param * (reserv_solo / SUBmax))) * evap_n_atendida
            RE = jnp.minimum(jnp.minimum(evap_n_atendida, reserv_solo), E_comp)
            Solo = jnp.maximum(reserv_solo - RE, 0.0)
            rec_rio = ks_array * Solo

            S1 = jnp.maximum(S3 + escoamento + rec_rio, 0.0)
            C = jnp.minimum(k_calha * (S1 ** b), S1)
            S3 = S1 - C
            reserv_solo_corrig = jnp.maximum(Solo + Ps - rec_rio, 0.0)

            C_expo = C ** expo_array
            Q_calc_d = (C - jnp.minimum(kl_array * C_expo, C)) * F_conversao
            mask_d = mask_d.astype(jnp.float32)

            soma_C_masked = soma_C_masked + (mask_d * C)
            soma_C_expo_masked = soma_C_expo_masked + (mask_d * C_expo)
            soma_qcalc_masked = soma_qcalc_masked + (mask_d * Q_calc_d)
            sum_qcalc = sum_qcalc + (mask_d * Q_calc_d)
            sum_qcalc2 = sum_qcalc2 + (mask_d * Q_calc_d * Q_calc_d)
            sum_qcalc_qobs = sum_qcalc_qobs + (mask_d * Q_calc_d * Q_obs_d)
            sum_sqrtqcalc_sqrtqobs = sum_sqrtqcalc_sqrtqobs + (mask_d * jnp.sqrt(jnp.maximum(Q_calc_d, 0.0)) * jnp.sqrt(jnp.maximum(Q_obs_d, 0.0)))
            erro_d = Q_obs_d - Q_calc_d
            numerador_nash = numerador_nash + (mask_d * erro_d * erro_d)

            return (ret_corrig, reserv_solo_corrig, S3, soma_C_masked, soma_C_expo_masked, numerador_nash, soma_qcalc_masked, sum_qcalc, sum_qcalc2, sum_qcalc_qobs, sum_sqrtqcalc_sqrtqobs), None

        carry_inicial = (
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
            jnp.zeros_like(ks_array),
        )
        (_, _, _, _, _, numerador_nash, soma_qcalc_masked, sum_qcalc, sum_qcalc2, sum_qcalc_qobs, sum_sqrtqcalc_sqrtqobs), _ = jax.lax.scan(
            passagem_com_erro,
            carry_inicial,
            (P, E, mask_calib, Q_obs),
        )

        n_masked = jnp.maximum(n_masked, 1.0)
        denominador_nash = (soma_qobs2_masked - (soma_qobs_masked * soma_qobs_masked) / n_masked) + 1e-9
        nash_array = 1.0 - (numerador_nash / denominador_nash)

        if fo in ('fo1', 'default'):
            fo_array = calcular_fo(jnp, nash_array, soma_qcalc_masked, soma_qobs_masked)
        elif fo in ('fo2', 'composite'):
            numerador_nse = sum_qcalc2 - (2.0 * sum_qcalc_qobs) + soma_qobs2_masked
            nse = 1.0 - (numerador_nse / denominador_nash)

            numerador_sqrt = sum_qcalc - (2.0 * sum_sqrtqcalc_sqrtqobs) + soma_qobs_sqrt_masked
            denominador_sqrt = (soma_qobs_sqrt2_masked - (soma_qobs_sqrt_masked * soma_qobs_sqrt_masked) / n_masked) + 1e-9
            nse_sqrt = 1.0 - (numerador_sqrt / denominador_sqrt)
            fo_array = (nse * 0.5) + (nse_sqrt * 0.5)
        elif fo in ('kge', 'fo_kge'):
            media_obs = soma_qobs_masked / n_masked
            media_sim = sum_qcalc / n_masked
            var_sim = (sum_qcalc2 / n_masked) - (media_sim ** 2)
            var_obs = (soma_qobs2_masked / n_masked) - (media_obs ** 2)
            std_sim = jnp.sqrt(jnp.maximum(var_sim, 1e-12))
            std_obs = jnp.sqrt(jnp.maximum(var_obs, 1e-12)) + 1e-9
            cov = (sum_qcalc_qobs / n_masked) - (media_sim * media_obs)
            r = cov / (std_sim * std_obs + 1e-9)
            alpha = std_sim / (std_obs + 1e-9)
            beta = (sum_qcalc / n_masked) / (media_obs + 1e-9)
            fo_array = 1.0 - jnp.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
        else:
            fo_array = calcular_fo(jnp, nash_array, soma_qcalc_masked, soma_qobs_masked)

        return fo_array, nash_array

    simular_cawm_jax = jax.jit(
        _simular_cawm_jax_core,
        static_argnames=('area', 'SUBmax', 'a_param', 'fo'),
    )
else:
    def simular_cawm_jax(*args, **kwargs):
        raise ImportError("JAX não está disponível. Instale JAX para usar GPU_JAX.")


def pso_vetorizado_jax(P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2, fo='fo1'):
    if not HAS_JAX:
        raise ImportError("JAX não está disponível. Instale JAX para usar GPU_JAX.")

    key = jax.random.PRNGKey(int(time.time() * 1e6) % (2**31 - 1))
    X = jax.random.uniform(key, (particulas, 2), dtype=jnp.float32)
    X = X.at[:, 0].set(X[:, 0] * 1.0)
    X = X.at[:, 1].set(X[:, 1] * 2.5 + 0.5)
    V = jnp.zeros((particulas, 2), dtype=jnp.float32)
    Pbest = jnp.array(X)
    Pbest_nash = jnp.full(particulas, -9999.0, dtype=jnp.float32)
    Gbest = jnp.zeros(2, dtype=jnp.float32)
    Gbest_nash = -9999.0
    Gbest_nse = -9999.0
    it_sem_melhora = 0
    historico = []
    iteracao_convergencia = 0

    for it in range(iteracoes):
        fo_array, nse_array = simular_cawm_jax(P, E, Q_obs, mask_calib, area=area, SUBmax=SUBmax, a_param=a_param, ks_array=X[:, 0], expo_array=X[:, 1], fo=fo)
        nash_array = jnp.round(fo_array, 3)
        nse_array = jnp.round(nse_array, 3)
        melhorias = nash_array > Pbest_nash
        Pbest_nash = jnp.where(melhorias, nash_array, Pbest_nash)
        Pbest = jnp.where(melhorias[:, None], X, Pbest)
        idx_max = int(jnp.argmax(Pbest_nash))
        max_nash = float(Pbest_nash[idx_max])

        if max_nash > Gbest_nash:
            Gbest_nash = max_nash
            Gbest_nse = float(nse_array[idx_max])
            Gbest = jnp.array(Pbest[idx_max])
            it_sem_melhora = 0
            iteracao_convergencia = it + 1
        else:
            it_sem_melhora += 1

        historico.append(Gbest_nash)
        print(f"      Fase 2 [GPU_JAX]: Iteração [{it+1:02d}/{iteracoes}] | FO: {Gbest_nash:8.3f} | NSE: {Gbest_nse:8.3f}      ", end="\r")
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break

        key, r1_key, r2_key = jax.random.split(key, 3)
        R1 = jax.random.uniform(r1_key, (particulas, 2), dtype=jnp.float32)
        R2 = jax.random.uniform(r2_key, (particulas, 2), dtype=jnp.float32)
        V = (w * V) + (c1 * R1 * (Pbest - X)) + (c2 * R2 * (Gbest - X))
        X = X + V
        X = X.at[:, 0].set(jnp.clip(X[:, 0], 0.0, 1.0))
        X = X.at[:, 1].set(jnp.clip(X[:, 1], 0.5, 3.0))

    return Gbest_nash, Gbest_nse, historico, float(np.asarray(Gbest[0])), float(np.asarray(Gbest[1])), iteracao_convergencia

def pso_mega_tensor_grid_50(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, hiperparametros, iteracoes, paciencia, fo='fo1'):
    # Executa a busca na GPU com 50 partículas simultaneamente para todas as 12 combinações (Apenas 1 rodada)
    Nc = len(hiperparametros)
    total_swarms = Nc
    total_particulas = total_swarms * particulas
    P_xp, E_xp, Q_obs_xp = xp.asarray(P), xp.asarray(E), xp.asarray(Q_obs)
    
    W = xp.zeros((total_swarms, 1, 1), dtype=xp.float32)
    C1 = xp.zeros((total_swarms, 1, 1), dtype=xp.float32)
    C2 = xp.zeros((total_swarms, 1, 1), dtype=xp.float32)
    
    for i, hp in enumerate(hiperparametros):
        W[i, 0, 0], C1[i, 0, 0], C2[i, 0, 0] = hp['w'], hp['c1'], hp['c2']
        
    X = xp.random.rand(total_swarms, particulas, 2).astype(xp.float32)
    X[:, :, 0] = X[:, :, 0] * 1.0
    X[:, :, 1] = X[:, :, 1] * 2.5 + 0.5
    V = xp.zeros((total_swarms, particulas, 2), dtype=xp.float32)
    Pbest = xp.copy(X)
    Pbest_nash = xp.full((total_swarms, particulas), -9999.0, dtype=xp.float32)
    Gbest = xp.zeros((total_swarms, 2), dtype=xp.float32)
    Gbest_nash = xp.full(total_swarms, -9999.0, dtype=xp.float32)
    it_sem_melhora = np.zeros(total_swarms, dtype=int)
    
    for it in range(iteracoes):
        X_flat = X.reshape(total_particulas, 2)
        nash_flat, _ = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X_flat[:, 0], X_flat[:, 1], fo=fo)
        nash_array = nash_flat.reshape(total_swarms, particulas)
        melhorias = nash_array > Pbest_nash
        Pbest_nash = xp.where(melhorias, nash_array, Pbest_nash)
        Pbest = xp.where(melhorias[:, :, None], X, Pbest)
        max_n = xp.max(Pbest_nash, axis=1)
        idx_max = xp.argmax(Pbest_nash, axis=1)
        
        for s in range(total_swarms):
            if max_n[s] > Gbest_nash[s]:
                Gbest_nash[s] = max_n[s]
                Gbest[s] = Pbest[s, int(idx_max[s])]
                it_sem_melhora[s] = 0
            else:
                it_sem_melhora[s] += 1
                
        melhor_global = float(xp.max(Gbest_nash))
        print(f"      Grid GPU (50P): Iteração {it+1:02d}/{iteracoes} | Combinações: {Nc:02d} | Melhor FO: {melhor_global:8.5f}", end="\r")
        if np.all(it_sem_melhora >= paciencia) or it == iteracoes - 1:
            print()
            break
            
        R1, R2 = xp.random.rand(total_swarms, particulas, 2).astype(xp.float32), xp.random.rand(total_swarms, particulas, 2).astype(xp.float32)
        V = (W * V) + (C1 * R1 * (Pbest - X)) + (C2 * R2 * (Gbest[:, None, :] - X))
        X = X + V
        X[:, :, 0] = xp.clip(X[:, :, 0], 0.0, 1.0)
        X[:, :, 1] = xp.clip(X[:, :, 1], 0.5, 3.0)
        
    return xp.asnumpy(Gbest_nash) if HAS_GPU else Gbest_nash

def pso_mega_tensor_grid_2000(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, hiperparametros, iteracoes, paciencia, fo='fo1'):
    # Executa a busca na GPU com todas as combinações em paralelo, igual ao grid de 50P.
    # Se a memória não aguentar, o erro vai aparecer aqui de forma explícita.
    Nc = len(hiperparametros)
    total_swarms = Nc
    total_particulas = total_swarms * particulas
    P_xp, E_xp, Q_obs_xp = xp.asarray(P), xp.asarray(E), xp.asarray(Q_obs)
    W = xp.zeros((total_swarms, 1, 1), dtype=xp.float32)
    C1 = xp.zeros((total_swarms, 1, 1), dtype=xp.float32)
    C2 = xp.zeros((total_swarms, 1, 1), dtype=xp.float32)

    for i, hp in enumerate(hiperparametros):
        W[i, 0, 0], C1[i, 0, 0], C2[i, 0, 0] = hp['w'], hp['c1'], hp['c2']

    X = xp.random.rand(total_swarms, particulas, 2).astype(xp.float32)
    X[:, :, 0] = X[:, :, 0] * 1.0
    X[:, :, 1] = X[:, :, 1] * 2.5 + 0.5
    V = xp.zeros((total_swarms, particulas, 2), dtype=xp.float32)
    Pbest = xp.copy(X)
    Pbest_nash = xp.full((total_swarms, particulas), -9999.0, dtype=xp.float32)
    Gbest = xp.zeros((total_swarms, 2), dtype=xp.float32)
    Gbest_nash = xp.full(total_swarms, -9999.0, dtype=xp.float32)
    it_sem_melhora = np.zeros(total_swarms, dtype=int)

    for it in range(iteracoes):
        X_flat = X.reshape(total_particulas, 2)
        nash_flat, _ = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X_flat[:, 0], X_flat[:, 1], fo=fo)
        nash_array = nash_flat.reshape(total_swarms, particulas)
        melhorias = nash_array > Pbest_nash
        Pbest_nash = xp.where(melhorias, nash_array, Pbest_nash)
        Pbest = xp.where(melhorias[:, :, None], X, Pbest)
        max_n = xp.max(Pbest_nash, axis=1)
        idx_max = xp.argmax(Pbest_nash, axis=1)

        for s in range(total_swarms):
            if max_n[s] > Gbest_nash[s]:
                Gbest_nash[s] = max_n[s]
                Gbest[s] = Pbest[s, int(idx_max[s])]
                it_sem_melhora[s] = 0
            else:
                it_sem_melhora[s] += 1

        melhor_global = float(xp.max(Gbest_nash))
        print(f"      Grid GPU (2000P): Iteração {it+1:02d}/{iteracoes} | Combinações: {Nc:02d} | Melhor FO: {melhor_global:8.5f}", end="\r")
        if np.all(it_sem_melhora >= paciencia) or it == iteracoes - 1:
            print()
            break

        R1, R2 = xp.random.rand(total_swarms, particulas, 2).astype(xp.float32), xp.random.rand(total_swarms, particulas, 2).astype(xp.float32)
        V = (W * V) + (C1 * R1 * (Pbest - X)) + (C2 * R2 * (Gbest[:, None, :] - X))
        X = X + V
        X[:, :, 0] = xp.clip(X[:, :, 0], 0.0, 1.0)
        X[:, :, 1] = xp.clip(X[:, :, 1], 0.5, 3.0)

    return xp.asnumpy(Gbest_nash) if HAS_GPU else Gbest_nash

def pso_mega_tensor_grid_repetido(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, hiperparametros, iteracoes, paciencia, repeticoes, processos_paralelos, fo='fo1'):
    # Executa várias buscas completas em lotes paralelos na GPU e devolve a média da FO/NSE por combinação.
    Nc = len(hiperparametros)
    P_xp, E_xp, Q_obs_xp = xp.asarray(P), xp.asarray(E), xp.asarray(Q_obs)

    soma_fo = xp.zeros(Nc, dtype=xp.float32)
    soma_nse = xp.zeros(Nc, dtype=xp.float32)
    total_repeticoes = 0

    for inicio in range(0, repeticoes, processos_paralelos):
        lote = min(processos_paralelos, repeticoes - inicio)

        W = xp.zeros((lote, Nc, 1, 1), dtype=xp.float32)
        C1 = xp.zeros((lote, Nc, 1, 1), dtype=xp.float32)
        C2 = xp.zeros((lote, Nc, 1, 1), dtype=xp.float32)

        for i, hp in enumerate(hiperparametros):
            W[:, i, 0, 0] = hp['w']
            C1[:, i, 0, 0] = hp['c1']
            C2[:, i, 0, 0] = hp['c2']

        X = xp.random.rand(lote, Nc, particulas, 2).astype(xp.float32)
        X[:, :, :, 0] = X[:, :, :, 0] * 1.0
        X[:, :, :, 1] = X[:, :, :, 1] * 2.5 + 0.5
        V = xp.zeros((lote, Nc, particulas, 2), dtype=xp.float32)
        Pbest = xp.copy(X)
        Pbest_nash = xp.full((lote, Nc, particulas), -9999.0, dtype=xp.float32)
        Gbest = xp.zeros((lote, Nc, 2), dtype=xp.float32)
        Gbest_nash = xp.full((lote, Nc), -9999.0, dtype=xp.float32)
        Gbest_nse = xp.full((lote, Nc), -9999.0, dtype=xp.float32)
        it_sem_melhora = xp.zeros((lote, Nc), dtype=int)

        for it in range(iteracoes):
            X_flat = X.reshape(lote * Nc * particulas, 2)
            fo_flat, nse_flat = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X_flat[:, 0], X_flat[:, 1], fo=fo)
            fo_array = fo_flat.reshape(lote, Nc, particulas)
            nse_array = nse_flat.reshape(lote, Nc, particulas)

            melhorias = fo_array > Pbest_nash
            Pbest_nash = xp.where(melhorias, fo_array, Pbest_nash)
            Pbest = xp.where(melhorias[:, :, :, None], X, Pbest)

            max_n = xp.max(Pbest_nash, axis=2)
            idx_max = xp.argmax(Pbest_nash, axis=2)

            for l in range(lote):
                for s in range(Nc):
                    if max_n[l, s] > Gbest_nash[l, s]:
                        Gbest_nash[l, s] = max_n[l, s]
                        Gbest_nse[l, s] = nse_array[l, s, int(idx_max[l, s])]
                        Gbest[l, s] = Pbest[l, s, int(idx_max[l, s])]
                        it_sem_melhora[l, s] = 0
                    else:
                        it_sem_melhora[l, s] += 1

            melhor_global = float(xp.max(Gbest_nash))
            print(
                f"      Grid GPU {inicio + 1:02d}-{inicio + lote:02d}/{repeticoes}: Iteração {it + 1:02d}/{iteracoes} | "
                f"Combinações: {Nc:02d} | Melhor FO: {melhor_global:8.5f}",
                end="\r",
            )
            if xp.all(it_sem_melhora >= paciencia).item() or it == iteracoes - 1:
                print()
                break

            R1, R2 = xp.random.rand(lote, Nc, particulas, 2).astype(xp.float32), xp.random.rand(lote, Nc, particulas, 2).astype(xp.float32)
            V = (W * V) + (C1 * R1 * (Pbest - X)) + (C2 * R2 * (Gbest[:, :, None, :] - X))
            X = X + V
            X[:, :, :, 0] = xp.clip(X[:, :, :, 0], 0.0, 1.0)
            X[:, :, :, 1] = xp.clip(X[:, :, :, 1], 0.5, 3.0)

        soma_fo = soma_fo + xp.sum(Gbest_nash, axis=0)
        soma_nse = soma_nse + xp.sum(Gbest_nse, axis=0)
        total_repeticoes += lote

    return soma_fo / total_repeticoes, soma_nse / total_repeticoes

# ==============================================================================
# 5. CALCULADORA DE MÉTRICAS ESTATÍSTICAS
# ==============================================================================
def calcular_metricas(Q_obs, Q_calc):
    eps = 0.01#1e-6 
    den_nse = np.sum((Q_obs - np.mean(Q_obs))**2) + eps
    nse = 1 - (np.sum((Q_obs - Q_calc)**2) / den_nse)
    
    Q_obs_log = np.log(Q_obs + eps)
    Q_calc_log = np.log(Q_calc + eps)
    nse_log = 1 - (np.sum((Q_obs_log - Q_calc_log)**2) / (np.sum((Q_obs_log - np.mean(Q_obs_log))**2) + eps))

    obs_legacy = np.nan_to_num(np.asarray(Q_obs, dtype=np.float64))
    calc_legacy = np.nan_to_num(np.asarray(Q_calc, dtype=np.float64))
    obs_media = np.mean(obs_legacy)
    if (obs_legacy <= 0).any() or (calc_legacy <= 0).any():
        sim_log = obs_legacy
        obs_log = calc_legacy
    else:
        sim_log = np.log10(obs_legacy)
        obs_log = np.log10(calc_legacy)
    nse_log_legacy = 1 - (np.sum((sim_log - obs_log)**2) / (np.sum((obs_log - obs_media)**2)))

    Q_obs_sqrt = np.sqrt(Q_obs)
    Q_calc_sqrt = np.sqrt(Q_calc)
    nse_sqrt = 1 - (np.sum((Q_obs_sqrt - Q_calc_sqrt)**2) / (np.sum((Q_obs_sqrt - np.mean(Q_obs_sqrt))**2) + eps))
    
    pbias = (np.sum(Q_obs - Q_calc) / (np.sum(Q_obs) + eps)) * 100
    rmse = np.sqrt(np.mean((Q_obs - Q_calc)**2))
    
    return nse, nse_log, nse_log_legacy, nse_sqrt, pbias, rmse

# ==============================================================================
# MOTOR ORQUESTRADOR: O BENCHMARK DEFINITIVO DO ARTIGO
# ==============================================================================
def executar_benchmark_pajeu():
    # Os IDs que a professora pediu
    ids_alvo = [25, 27, 29, 31, 33]
    
    hiperparametros = [
    # --- GRUPO 1: PONTOS DE CONVERGÊNCIA TEÓRICA (OS MAIS USADOS) ---
    {"w": 0.7298, "c1": 1.49618, "c2": 1.49618},  # 1. Fator de Constrição de Clerc (O melhor e mais famoso)
    {"w": 0.6000, "c1": 1.70000, "c2": 1.70000},  # 2. Configuração popular de estabilidade estática
    {"w": 0.7000, "c1": 2.00000, "c2": 2.00000},  # 3. Padrão clássico (Soma c1+c2 = 4 com inércia moderada)

    # --- GRUPO 2: EQUILÍBRIO COMPORTAMENTAL (STANDARD) ---
    {"w": 0.5000, "c1": 1.50000, "c2": 1.50000},  # 4. Equilíbrio perfeito de forças (Média inércia)
    {"w": 0.8000, "c1": 1.50000, "c2": 1.50000},  # 5. Equilíbrio perfeito (Alta inércia para evitar mínimos locais)
    {"w": 0.4000, "c1": 2.00000, "c2": 2.00000},  # 6. Foco em convergência rápida e ajuste fino local

    # --- GRUPO 3: FOCO COGNITIVO / INDIVIDUAL (EXPLORAÇÃO) ---
    {"w": 0.6000, "c1": 2.00000, "c2": 1.00000},  # 7. Alta busca individual, evita seguir o grupo cegamente
    {"w": 0.7000, "c1": 2.50000, "c2": 0.50000},  # 8. Extremo cognitivo (Ótimo para funções multimodais complexas)
    {"w": 0.8000, "c1": 2.00000, "c2": 1.00000},  # 9. Alta inércia com viés de auto-aprendizado

    # --- GRUPO 4: FOCO SOCIAL / COLETIVO (CONVERGÊNCIA) ---
    {"w": 0.6000, "c1": 1.00000, "c2": 2.00000},  # 10. Alta atração pelo melhor global (Convergência rápida)
    {"w": 0.4000, "c1": 1.00000, "c2": 2.50000},  # 11. Extremo social com baixa inércia (Busca agressiva)
    {"w": 0.7000, "c1": 1.00000, "c2": 2.00000},   # 12. Atração social com maior capacidade de escape
    # --- GRUPO 5: CONFIGURAÇÃO USADA ORIGINALMENTE NO ARTIGO (APENAS PARA COMPARAÇÃO) ---
    {"w": 0.8000, "c1": 1.00000, "c2": 2.00000},
    {"w": 0.9000, "c1": 1.20000, "c2": 1.20000},  # 14. Alta Exploração Hidrológica (Evita convergência prematura em bacias complexas)
    {"w": 0.7298, "c1": 2.04700, "c2": 2.04700}   # 15. Parametrização Alternativa de Clerc & Kennedy (Alta aceleração controlada)  
]
    
    xp = cp if HAS_GPU else np
    iteracoes_grid = 20
    paciencia_grid = 5
    iteracoes_fase2 = 100
    paciencia_fase2 = 10
    repeticoes_fase1 = 10
    processos_paralelos_fase1 = 20
    
    tabela_estatisticas = []
    tabela_benchmark = []
    dados_convergencia = []

    print("\n" + "="*90)
    print(" BENCHMARK E VALIDAÇÃO ESTATÍSTICA: BACIA DO PAJEÚ (PE)")
    print("="*90)

    for calib_id in ids_alvo:
        print(f"\n▶ INICIANDO CALIBRATION_ID: {calib_id}")
        try:
            P, E, Q_obs, mask_calib, mask_valid, area, SUBmax, a_param, nome_bacia, metodo = extrair_dados_por_id(calib_id)
        except Exception as e:
            print(f"Erro ao extrair ID {calib_id}: {e}")
            continue
        try:
            print(f"  Bacia: {nome_bacia} | Método: {metodo} | Calibração: {mask_calib.sum()} dias")

            # -------------------------------------------------------------------
            # FASE 1: GRID SEARCH (Acha melhores parâmetros para 50P e 2000P)
            # -------------------------------------------------------------------
            print("  [Fase 1] Grid Search Rápido na GPU...")
            print(f"    Repetições: {repeticoes_fase1} | Paralelismo: {processos_paralelos_fase1} | Iterações: {iteracoes_grid} | Paciencia: {paciencia_grid}")
            t_calib_inicio = time.perf_counter()
            nashes_50, nse_50 = pso_mega_tensor_grid_repetido(
                xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, 50, hiperparametros,
                iteracoes_grid, paciencia_grid, repeticoes_fase1, processos_paralelos_fase1, fo=FO_SELECTION,
            )
            limpar_memoria_gpu()
            nashes_2000, nse_2000 = pso_mega_tensor_grid_repetido(
                xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, 2000, hiperparametros,
                iteracoes_grid, paciencia_grid, repeticoes_fase1, processos_paralelos_fase1, fo=FO_SELECTION,
            )

            idx_50 = int(xp.argmax(nashes_50))
            idx_2000 = int(xp.argmax(nashes_2000))
            melhor_hp_50 = hiperparametros[idx_50]
            melhor_hp_2000 = hiperparametros[idx_2000]
            tempo_calibracao_pso = time.perf_counter() - t_calib_inicio
            print(f"    Melhor Combo (50P): w={melhor_hp_50['w']}, c1={melhor_hp_50['c1']}, c2={melhor_hp_50['c2']} | FO Médio={float(nashes_50[idx_50]):.5f} | NSE Médio={float(nse_50[idx_50]):.5f}")
            print(f"    Melhor Combo (2000P): w={melhor_hp_2000['w']}, c1={melhor_hp_2000['c1']}, c2={melhor_hp_2000['c2']} | FO Médio={float(nashes_2000[idx_2000]):.5f} | NSE Médio={float(nse_2000[idx_2000]):.5f}")
            print(f"    Tempo total de calibração dos parâmetros do PSO: {tempo_calibracao_pso:.2f}s")

            # -------------------------------------------------------------------
            # FASE 2: BENCHMARK (Roda 1x Legado, 1x CPU, 1x GPU para comparar Tempo)
            # -------------------------------------------------------------------
            print("  [Fase 2] Benchmark Comparativo (1 rodada cada)...")
            configuracoes = [
                ("Escalar (Legado)", pso_escalar_legado, None, 50, melhor_hp_50),
                #("Matricial CPU 50P", pso_vetorizado, np, 50, melhor_hp_50),
                #("Matricial GPU 50P", pso_vetorizado, xp, 50, melhor_hp_50),
                ("Matricial CPU", pso_vetorizado, np, 2000, melhor_hp_2000),
                ("Matricial GPU", pso_vetorizado, xp, 2000, melhor_hp_2000)
            ]

            if HAS_JAX:
                configuracoes.append(("GPU_JAX", pso_vetorizado_jax, None, 2000, melhor_hp_2000))

            best_ks_geral, best_expo_geral = 0.0, 0.0

            for nome_metodo, funcao_pso, xp_backend, part, params in configuracoes:
                if xp_backend is None and nome_metodo == "Matricial GPU": continue

                w_opt, c1_opt, c2_opt = params['w'], params['c1'], params['c2']

                t_inicio = time.perf_counter()
                if xp_backend is None:
                    if nome_metodo == "GPU_JAX":
                        nash_final, nse_final, historico, ks_f, expo_f, iteracao_conv = funcao_pso(P, E, Q_obs, mask_calib, area, SUBmax, a_param, part, iteracoes_fase2, paciencia_fase2, w_opt, c1_opt, c2_opt, fo=FO_SELECTION)
                    else:
                        nash_final, nse_final, historico, ks_f, expo_f, iteracao_conv = funcao_pso(P, E, Q_obs, mask_calib, area, SUBmax, a_param, part, iteracoes_fase2, paciencia_fase2, w_opt, c1_opt, c2_opt, fo=FO_SELECTION)
                else:
                    nash_final, nse_final, historico, ks_f, expo_f, iteracao_conv = funcao_pso(xp_backend, P, E, Q_obs, mask_calib, area, SUBmax, a_param, part, iteracoes_fase2, paciencia_fase2, w_opt, c1_opt, c2_opt, fo=FO_SELECTION)
                    if xp_backend == cp: cp.cuda.Stream.null.synchronize()
                t_fim = time.perf_counter()

                tempo_exec = t_fim - t_inicio
                print(f"    -> {nome_metodo}: Tempo = {tempo_exec:.2f}s | FO Final = {nash_final:.4f} | NSE Final = {nse_final:.4f}")

                # Guardamos o Ks e Expo da GPU para o cálculo final das estatísticas
                if nome_metodo == "Matricial GPU":
                    best_ks_geral, best_expo_geral = ks_f, expo_f

                tabela_benchmark.append({
                    "ID": calib_id, "Bacia": nome_bacia, "Método": nome_metodo, "Partículas": part,
                    "Tempo_Calib_PSO(s)": round(tempo_calibracao_pso, 2),
                    "Tempo(s)": round(tempo_exec, 2), "FO": round(nash_final, 4), "NSE": round(nse_final, 4), "Iter_Conv": iteracao_conv
                })
                for it, n_val in enumerate(historico):
                    dados_convergencia.append({"ID": calib_id, "Método": nome_metodo, "Iteração": it+1, "NSE": n_val})

            # -------------------------------------------------------------------
            # FASE 3: ESTATÍSTICAS COMPLETAS (O pedido da Professora)
            # -------------------------------------------------------------------
            print("  [Fase 3] Validando série e calculando métricas...")
            Q_calc_total, kl_calc = simular_forward_determinista(P, E, Q_obs, mask_calib, area, SUBmax, a_param, best_ks_geral, best_expo_geral)

            Q_obs_calib, Q_calc_calib = Q_obs[mask_calib], Q_calc_total[mask_calib]
            Q_obs_valid, Q_calc_valid = Q_obs[mask_valid], Q_calc_total[mask_valid]

            nse_c, nsel_c, nslog_leg_c, nses_c, pbias_c, rmse_c = calcular_metricas(Q_obs_calib, Q_calc_calib)
            nse_v, nsel_v, nslog_leg_v, nses_v, pbias_v, rmse_v = calcular_metricas(Q_obs_valid, Q_calc_valid)

            tabela_estatisticas.append({
                "ID": calib_id, "Bacia": nome_bacia, "Metodo": metodo,
                "Ks": round(best_ks_geral, 4), "Kl": round(kl_calc, 4), "Expo": round(best_expo_geral, 4),
                "NSE_Cal": round(nse_c, 4), "NSE_Log_Cal": round(nsel_c, 4), "NSE_Log_Legado_Cal": round(nslog_leg_c, 4), "NSE_Sqrt_Cal": round(nses_c, 4), "PBIAS_Cal": round(pbias_c, 2), "RMSE_Cal": round(rmse_c, 2),
                "NSE_Val": round(nse_v, 4), "NSE_Log_Val": round(nsel_v, 4), "NSE_Log_Legado_Val": round(nslog_leg_v, 4), "NSE_Sqrt_Val": round(nses_v, 4), "PBIAS_Val": round(pbias_v, 2), "RMSE_Val": round(rmse_v, 2)
            })

            print("  [Fase 3] Resumo do bloco:")
            print(f"    Calibração: NSE={nse_c:.4f} | NSE_Log={nsel_c:.4f} | NSE_Log_Legado={nslog_leg_c:.4f} | NSE_Sqrt={nses_c:.4f} | PBIAS={pbias_c:.2f} | RMSE={rmse_c:.2f}")
            print(f"    Validação : NSE={nse_v:.4f} | NSE_Log={nsel_v:.4f} | NSE_Log_Legado={nslog_leg_v:.4f} | NSE_Sqrt={nses_v:.4f} | PBIAS={pbias_v:.2f} | RMSE={rmse_v:.2f}")
        finally:
            limpar_memoria_gpu()

    # ==========================================================
    # GERAÇÃO DOS CSVs FINAIS
    # ==========================================================
    pd.DataFrame(tabela_benchmark).to_csv("benchmark_tempos_pajeu.csv", index=False)
    pd.DataFrame(dados_convergencia).to_csv("benchmark_convergencia_pajeu.csv", index=False)
    
    df_estat = pd.DataFrame(tabela_estatisticas)
    print("\n" + "="*120)
    print(" TABELA DE DADOS FINAIS - ESTATÍSTICAS DA PROFESSORA")
    print("="*120)
    print(df_estat.to_string(index=False))
    df_estat.to_csv("tabela_pajeu_estatisticas.csv", index=False)
    print("\n[SUCESSO] 3 CSVs gerados (Tempos, Convergência e Tabela Estatística)!")

if __name__ == "__main__":
    executar_benchmark_pajeu()
import time
import gc
import numpy as np
import pandas as pd

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Aviso: CuPy não encontrado. Rodando em NumPy (Matricial CPU).")

from db_init import initialize_db
from db_models import Bacia, PrecipitationDaily, EvaporationMonthly, FlowDaily, CalibrationPeriod


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
def simular_cawm_vetorizado(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, ks_array, expo_array):
    dias = len(P)
    particulas = len(ks_array)
    b = 1.666666667
    T_sec = 86400.0
    
    if area < 6000: k_calha = 0.3745 * (area**-0.489) + 0.0146
    elif area <= 60000: k_calha = 34.343 * (area**-0.853)
    else: k_calha = 0.0028
        
    F_conversao = (area * 1000000.0 / T_sec) / 1000.0
    ret_corrig = xp.zeros(particulas, dtype=xp.float32)
    reserv_solo_corrig = xp.zeros(particulas, dtype=xp.float32)
    S3 = xp.zeros(particulas, dtype=xp.float32)

    C_hist = xp.zeros((dias, particulas), dtype=xp.float32)
    C_expo_hist = xp.zeros((dias, particulas), dtype=xp.float32)

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

        C_hist[d, :] = C
        C_expo_hist[d, :] = C ** expo_array

    mask_xp = xp.asarray(mask_calib)
    Q_obs_masked = Q_obs[mask_xp]
    vol_obs_mm = xp.sum(Q_obs_masked) / F_conversao
    
    C_hist_masked = C_hist[mask_xp, :]
    C_expo_hist_masked = C_expo_hist[mask_xp, :]
    soma_C_masked = xp.sum(C_hist_masked, axis=0)
    soma_C_expo_masked = xp.sum(C_expo_hist_masked, axis=0)
    
    kl_array = xp.maximum((soma_C_masked - vol_obs_mm) / (soma_C_expo_masked + 1e-9), 0.0) 

    Q_calc_hist = xp.zeros((dias, particulas), dtype=xp.float32)
    for d in range(dias):
        perdas = xp.minimum(kl_array * C_expo_hist[d, :], C_hist[d, :])
        Q_calc_hist[d, :] = (C_hist[d, :] - perdas) * F_conversao

    Q_calc_masked = Q_calc_hist[mask_xp, :]
    media_obs = xp.mean(Q_obs_masked)
    denominador_nash = xp.sum((Q_obs_masked - media_obs)**2) + 1e-9
    numerador_nash = xp.sum((xp.expand_dims(Q_obs_masked, 1) - Q_calc_masked)**2, axis=0)
    nash_array = 1.0 - (numerador_nash / denominador_nash)

    return nash_array

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
def pso_escalar_legado(P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2):
    X = np.random.rand(particulas, 2)
    X[:, 0] = X[:, 0] * 1.0
    X[:, 1] = X[:, 1] * 2.5 + 0.5
    V = np.zeros((particulas, 2))
    Pbest = np.copy(X)
    Pbest_nash = np.full(particulas, -9999.0)
    Gbest = np.zeros(2)
    Gbest_nash = -9999.0
    it_sem_melhora = 0
    historico = []
    
    for it in range(iteracoes):
        houve_melhora = False
        nash_array = np.zeros(particulas)
        for i in range(particulas):
            n_val = simular_cawm_vetorizado(np, P, E, Q_obs, mask_calib, area, SUBmax, a_param, np.array([X[i, 0]]), np.array([X[i, 1]]))
            nash_array[i] = n_val[0]
            if nash_array[i] > Pbest_nash[i]:
                Pbest_nash[i] = nash_array[i]
                Pbest[i] = X[i]
                if nash_array[i] > Gbest_nash:
                    Gbest_nash = nash_array[i]
                    Gbest = np.copy(X[i])
                    houve_melhora = True
        
        it_sem_melhora = 0 if houve_melhora else it_sem_melhora + 1
        historico.append(Gbest_nash)
        print(f"      Fase 2 [Legado]: Iteração [{it+1:02d}/{iteracoes}] | NSE: {Gbest_nash:8.5f}", end="\r")
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break
            
        for i in range(particulas):
            r1, r2 = np.random.rand(2)
            V[i] = (w * V[i]) + (c1 * r1 * (Pbest[i] - X[i])) + (c2 * r2 * (Gbest - X[i]))
            X[i] = X[i] + V[i]
            X[i, 0] = np.clip(X[i, 0], 0.0, 1.0)
            X[i, 1] = np.clip(X[i, 1], 0.5, 3.0)
            
    return Gbest_nash, historico, float(Gbest[0]), float(Gbest[1])

def pso_vetorizado(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2):
    P_xp, E_xp, Q_obs_xp = xp.asarray(P), xp.asarray(E), xp.asarray(Q_obs)
    X = xp.random.rand(particulas, 2).astype(xp.float32)
    X[:, 0] = X[:, 0] * 1.0
    X[:, 1] = X[:, 1] * 2.5 + 0.5
    V = xp.zeros((particulas, 2), dtype=xp.float32)
    Pbest = xp.copy(X)
    Pbest_nash = xp.full(particulas, -9999.0, dtype=xp.float32)
    Gbest = xp.zeros(2, dtype=xp.float32)
    Gbest_nash = -9999.0
    it_sem_melhora = 0
    historico = []
    
    for it in range(iteracoes):
        nash_array = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X[:, 0], X[:, 1])
        melhorias = nash_array > Pbest_nash
        Pbest_nash = xp.where(melhorias, nash_array, Pbest_nash)
        Pbest = xp.where(melhorias[:, None], X, Pbest)
        idx_max = xp.argmax(Pbest_nash)
        max_nash = float(Pbest_nash[idx_max])
        
        if max_nash > Gbest_nash:
            Gbest_nash = max_nash
            Gbest = xp.copy(Pbest[idx_max])
            it_sem_melhora = 0
        else:
            it_sem_melhora += 1
            
        historico.append(Gbest_nash)
        print(f"      Fase 2 [Matricial]: Iteração [{it+1:02d}/{iteracoes}] | NSE: {Gbest_nash:8.5f}      ", end="\r")
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break
            
        R1, R2 = xp.random.rand(particulas, 2).astype(xp.float32), xp.random.rand(particulas, 2).astype(xp.float32)
        V = (w * V) + (c1 * R1 * (Pbest - X)) + (c2 * R2 * (Gbest - X))
        X = X + V
        X[:, 0] = xp.clip(X[:, 0], 0.0, 1.0)
        X[:, 1] = xp.clip(X[:, 1], 0.5, 3.0)
        
    return Gbest_nash, historico, float(Gbest[0]), float(Gbest[1])

def pso_mega_tensor_grid_50(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, hiperparametros, iteracoes, paciencia):
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
        nash_flat = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X_flat[:, 0], X_flat[:, 1])
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
        print(f"      Grid GPU (50P): Iteração {it+1:02d}/{iteracoes} | Combinações: {Nc:02d} | Melhor NSE: {melhor_global:8.5f}", end="\r")
        if np.all(it_sem_melhora >= paciencia) or it == iteracoes - 1:
            print()
            break
            
        R1, R2 = xp.random.rand(total_swarms, particulas, 2).astype(xp.float32), xp.random.rand(total_swarms, particulas, 2).astype(xp.float32)
        V = (W * V) + (C1 * R1 * (Pbest - X)) + (C2 * R2 * (Gbest[:, None, :] - X))
        X = X + V
        X[:, :, 0] = xp.clip(X[:, :, 0], 0.0, 1.0)
        X[:, :, 1] = xp.clip(X[:, :, 1], 0.5, 3.0)
        
    return xp.asnumpy(Gbest_nash) if HAS_GPU else Gbest_nash

def pso_mega_tensor_grid_2000(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, hiperparametros, iteracoes, paciencia):
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
        nash_flat = simular_cawm_vetorizado(xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X_flat[:, 0], X_flat[:, 1])
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
        print(f"      Grid GPU (2000P): Iteração {it+1:02d}/{iteracoes} | Combinações: {Nc:02d} | Melhor NSE: {melhor_global:8.5f}", end="\r")
        if np.all(it_sem_melhora >= paciencia) or it == iteracoes - 1:
            print()
            break

        R1, R2 = xp.random.rand(total_swarms, particulas, 2).astype(xp.float32), xp.random.rand(total_swarms, particulas, 2).astype(xp.float32)
        V = (W * V) + (C1 * R1 * (Pbest - X)) + (C2 * R2 * (Gbest[:, None, :] - X))
        X = X + V
        X[:, :, 0] = xp.clip(X[:, :, 0], 0.0, 1.0)
        X[:, :, 1] = xp.clip(X[:, :, 1], 0.5, 3.0)

    return xp.asnumpy(Gbest_nash) if HAS_GPU else Gbest_nash

# ==============================================================================
# 5. CALCULADORA DE MÉTRICAS ESTATÍSTICAS
# ==============================================================================
def calcular_metricas(Q_obs, Q_calc):
    eps = 1e-6 
    den_nse = np.sum((Q_obs - np.mean(Q_obs))**2) + eps
    nse = 1 - (np.sum((Q_obs - Q_calc)**2) / den_nse)
    
    Q_obs_log = np.log(Q_obs + eps)
    Q_calc_log = np.log(Q_calc + eps)
    nse_log = 1 - (np.sum((Q_obs_log - Q_calc_log)**2) / (np.sum((Q_obs_log - np.mean(Q_obs_log))**2) + eps))
    
    Q_obs_sqrt = np.sqrt(Q_obs)
    Q_calc_sqrt = np.sqrt(Q_calc)
    nse_sqrt = 1 - (np.sum((Q_obs_sqrt - Q_calc_sqrt)**2) / (np.sum((Q_obs_sqrt - np.mean(Q_obs_sqrt))**2) + eps))
    
    pbias = (np.sum(Q_obs - Q_calc) / (np.sum(Q_obs) + eps)) * 100
    rmse = np.sqrt(np.mean((Q_obs - Q_calc)**2))
    
    return nse, nse_log, nse_sqrt, pbias, rmse

# ==============================================================================
# MOTOR ORQUESTRADOR: O BENCHMARK DEFINITIVO DO ARTIGO
# ==============================================================================
def executar_benchmark_pajeu():
    # Os IDs que a professora pediu
    ids_alvo = [25, 27, 29, 31, 33]
    
    hiperparametros = [
        {"w": 0.4, "c1": 1.0, "c2": 2.0}, {"w": 0.5, "c1": 1.5, "c2": 1.5},
        {"w": 0.6, "c1": 2.0, "c2": 1.0}, {"w": 0.7, "c1": 1.5, "c2": 2.0},
        {"w": 0.8, "c1": 2.0, "c2": 2.0}, {"w": 0.9, "c1": 1.0, "c2": 1.0},
        {"w": 0.9, "c1": 0.5, "c2": 2.5}, {"w": 0.7, "c1": 2.5, "c2": 0.5},
        {"w": 0.5, "c1": 2.0, "c2": 2.0}, {"w": 0.8, "c1": 1.5, "c2": 1.5},
        {"w": 0.6, "c1": 1.0, "c2": 1.0}, {"w": 0.4, "c1": 2.0, "c2": 1.5}
    ]
    
    xp = cp if HAS_GPU else np
    iteracoes = 20
    paciencia = 10
    
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
            nashes_50 = pso_mega_tensor_grid_50(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, 50, hiperparametros, iteracoes, paciencia)
            limpar_memoria_gpu()
            nashes_2000 = pso_mega_tensor_grid_2000(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, 2000, hiperparametros, iteracoes, paciencia)

            melhor_hp_50 = hiperparametros[np.argmax(nashes_50)]
            melhor_hp_2000 = hiperparametros[np.argmax(nashes_2000)]

            print(f"    Melhor Combo (50P): w={melhor_hp_50['w']}, c1={melhor_hp_50['c1']}, c2={melhor_hp_50['c2']}")
            print(f"    Melhor Combo (2000P): w={melhor_hp_2000['w']}, c1={melhor_hp_2000['c1']}, c2={melhor_hp_2000['c2']}")

            # -------------------------------------------------------------------
            # FASE 2: BENCHMARK (Roda 1x Legado, 1x CPU, 1x GPU para comparar Tempo)
            # -------------------------------------------------------------------
            print("  [Fase 2] Benchmark Comparativo (1 rodada cada)...")
            configuracoes = [
                ("Escalar (Legado)", pso_escalar_legado, None, 50, melhor_hp_50),
                ("Matricial CPU", pso_vetorizado, np, 2000, melhor_hp_2000),
                ("Matricial GPU", pso_vetorizado, xp, 2000, melhor_hp_2000)
            ]

            best_ks_geral, best_expo_geral = 0.0, 0.0

            for nome_metodo, funcao_pso, xp_backend, part, params in configuracoes:
                if xp_backend is None and nome_metodo == "Matricial GPU": continue

                w_opt, c1_opt, c2_opt = params['w'], params['c1'], params['c2']

                t_inicio = time.perf_counter()
                if xp_backend is None:
                    nash_final, historico, ks_f, expo_f = funcao_pso(P, E, Q_obs, mask_calib, area, SUBmax, a_param, part, iteracoes, paciencia, w_opt, c1_opt, c2_opt)
                else:
                    nash_final, historico, ks_f, expo_f = funcao_pso(xp_backend, P, E, Q_obs, mask_calib, area, SUBmax, a_param, part, iteracoes, paciencia, w_opt, c1_opt, c2_opt)
                    if xp_backend == cp: cp.cuda.Stream.null.synchronize()
                t_fim = time.perf_counter()

                tempo_exec = t_fim - t_inicio
                print(f"    -> {nome_metodo}: Tempo = {tempo_exec:.2f}s | NSE Final = {nash_final:.4f}")

                # Guardamos o Ks e Expo da GPU para o cálculo final das estatísticas
                if nome_metodo == "Matricial GPU":
                    best_ks_geral, best_expo_geral = ks_f, expo_f

                tabela_benchmark.append({
                    "ID": calib_id, "Bacia": nome_bacia, "Método": nome_metodo, "Partículas": part,
                    "Tempo(s)": round(tempo_exec, 2), "NSE": round(nash_final, 4)
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

            nse_c, nsel_c, nses_c, pbias_c, rmse_c = calcular_metricas(Q_obs_calib, Q_calc_calib)
            nse_v, nsel_v, nses_v, pbias_v, rmse_v = calcular_metricas(Q_obs_valid, Q_calc_valid)

            tabela_estatisticas.append({
                "ID": calib_id, "Bacia": nome_bacia, "Metodo": metodo,
                "Ks": round(best_ks_geral, 4), "Kl": round(kl_calc, 4), "Expo": round(best_expo_geral, 4),
                "NSE_Cal": round(nse_c, 4), "NSE_Log_Cal": round(nsel_c, 4), "NSE_Sqrt_Cal": round(nses_c, 4), "PBIAS_Cal": round(pbias_c, 2), "RMSE_Cal": round(rmse_c, 2),
                "NSE_Val": round(nse_v, 4), "NSE_Log_Val": round(nsel_v, 4), "NSE_Sqrt_Val": round(nses_v, 4), "PBIAS_Val": round(pbias_v, 2), "RMSE_Val": round(rmse_v, 2)
            })
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
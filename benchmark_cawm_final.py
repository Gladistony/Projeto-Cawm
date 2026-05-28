import time
import numpy as np
import pandas as pd
from typing import Tuple

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Aviso: CuPy não encontrado. Teste de GPU será ignorado.")

from db_init import initialize_db
from db_models import Bacia, PrecipitationDaily, EvaporationMonthly, FlowDaily, CalibrationPeriod

# ==============================================================================
# 1. FUNÇÃO DE EXTRAÇÃO DE DADOS
# ==============================================================================
def extrair_dados_bacia(nome_bacia: str):
    print(f"[{time.strftime('%H:%M:%S')}] Extraindo dados reais e alinhando datas para: {nome_bacia}...")
    
    resultado_db = initialize_db()
    if isinstance(resultado_db, tuple):
        for item in resultado_db:
            if hasattr(item, 'query'):
                session = item
                break
            elif callable(item): 
                session = item()
                if hasattr(session, 'query'):
                    break
    else:
        session = resultado_db
        
    bacia = session.query(Bacia).filter(Bacia.nome == nome_bacia).first()
    if not bacia:
        raise ValueError(f"Bacia {nome_bacia} não encontrada.")
        
    area_km2 = bacia.area_km2
    SUBmax = bacia.submax if bacia.submax is not None else 1000.0
    a_param = 1.0 
    
    periodo = session.query(CalibrationPeriod).filter_by(bacia_id=bacia.id).first()
    if not periodo:
        raise ValueError(f"Sem período de calibração para {nome_bacia}.")
        
    calib_start = pd.Timestamp(periodo.calib_start)
    calib_end = pd.Timestamp(periodo.calib_end)
    
    precip = session.query(PrecipitationDaily).filter_by(bacia_id=bacia.id).order_by(PrecipitationDaily.data).all()
    vazao = session.query(FlowDaily).filter_by(bacia_id=bacia.id).order_by(FlowDaily.data).all()
    evap_mensal = session.query(EvaporationMonthly).filter_by(bacia_id=bacia.id).all()
    session.close()
    
    df_chuva = pd.DataFrame({"data": [p.data for p in precip], "chuva": [float(p.valor) for p in precip]})
    df_vazao = pd.DataFrame({"data": [q.data for q in vazao], "vazao": [float(q.valor) for q in vazao]})
    df = pd.merge(df_vazao, df_chuva, on="data", how="inner").sort_values("data")
    
    mapa_evap = {int(e.mes): float(e.valor) for e in evap_mensal}
    df["mes"] = pd.to_datetime(df["data"], errors="coerce").dt.month
    df["evap"] = df["mes"].map(mapa_evap).fillna(5.0)
    
    df["data_dt"] = pd.to_datetime(df["data"])
    mask_calib = ((df["data_dt"] >= calib_start) & (df["data_dt"] <= calib_end)).to_numpy()
    
    P = df["chuva"].to_numpy(dtype=np.float32)
    E = df["evap"].to_numpy(dtype=np.float32)
    Q_obs = df["vazao"].to_numpy(dtype=np.float32)
    
    print(f"[{time.strftime('%H:%M:%S')}] Total de dias: {len(P)} | Dias Calibração (Máscara): {mask_calib.sum()}")
    return P, E, Q_obs, mask_calib, area_km2, SUBmax, a_param

# ==============================================================================
# MOTOR DO CAWM VETORIZADO (RÉPLICA DO MÉTODO ANTIGO - Kl Analítico)
# ==============================================================================
def simular_cawm_vetorizado_replica(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, ks_array, expo_array):
    dias = len(P)
    particulas = len(ks_array)
    b = 1.666666667
    T_sec = 86400.0
    
    if area < 6000:
        k_calha = 0.3745 * (area**-0.489) + 0.0146
    elif area <= 60000:
        k_calha = 34.343 * (area**-0.853)
    else:
        k_calha = 0.0028
        
    F_conversao = (area * 1000000.0 / T_sec) / 1000.0
    ret_corrig = xp.zeros(particulas, dtype=xp.float32)
    reserv_solo_corrig = xp.zeros(particulas, dtype=xp.float32)
    S3 = xp.zeros(particulas, dtype=xp.float32)

    C_hist = xp.zeros((dias, particulas), dtype=xp.float32)
    C_expo_hist = xp.zeros((dias, particulas), dtype=xp.float32)

    for d in range(dias):
        P_d = P[d]
        E_d = E[d]

        evap_inicial = xp.where(ret_corrig + P_d >= E_d, E_d, ret_corrig + P_d)
        ret_corrig = xp.maximum(ret_corrig + P_d - evap_inicial, 0.0)
        evap_n_atendida = E_d - evap_inicial

        Pn = P_d - evap_inicial
        Pn_pos = xp.maximum(Pn, 0.0)
        hiperb = xp.tanh(Pn_pos / SUBmax)

        reserv_solo = xp.minimum(reserv_solo_corrig, SUBmax)
        Sub = reserv_solo / SUBmax
        
        termo1 = SUBmax * (1.0 - Sub**2) * hiperb
        termo2 = 1.0 + Sub * hiperb
        Ps = termo1 / (termo2 + 1e-9)
        escoamento = P_d - evap_inicial - Ps

        E_comp = (1.0 - xp.exp(-a_param * (reserv_solo / SUBmax))) * evap_n_atendida
        RE = xp.minimum(xp.minimum(evap_n_atendida, reserv_solo), E_comp)
        Solo = xp.maximum(reserv_solo - RE, 0.0)
        rec_rio = ks_array * Solo

        S1 = xp.maximum(S3 + escoamento + rec_rio, 0.0)
        C = xp.minimum(k_calha * (S1 ** b), S1)
        S2 = S1 - C
        S3 = S2  
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
    
    kl_array = (soma_C_masked - vol_obs_mm) / (soma_C_expo_masked + 1e-9)
    kl_array = xp.maximum(kl_array, 0.0) 

    Q_calc_hist = xp.zeros((dias, particulas), dtype=xp.float32)
    for d in range(dias):
        perdas = xp.minimum(kl_array * C_expo_hist[d, :], C_hist[d, :])
        Q_calc_hist[d, :] = (C_hist[d, :] - perdas) * F_conversao

    Q_calc_masked = Q_calc_hist[mask_xp, :]
    media_obs = xp.mean(Q_obs_masked)
    denominador_nash = xp.sum((Q_obs_masked - media_obs)**2) + 1e-9
    numerador_nash = xp.sum((xp.expand_dims(Q_obs_masked, 1) - Q_calc_masked)**2, axis=0)
    nash_array = 1.0 - (numerador_nash / denominador_nash)

    return nash_array, kl_array

# ==============================================================================
# 2. MÉTODO ESCALAR (LEGADO)
# ==============================================================================
def pso_escalar(P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2):
    X = np.random.rand(particulas, 2)
    X[:, 0] = X[:, 0] * 1.0
    X[:, 1] = X[:, 1] * 2.5 + 0.5
    
    V = np.zeros((particulas, 2))
    Pbest = np.copy(X)
    Pbest_nash = np.full(particulas, -9999.0)
    
    Gbest = np.zeros(2)
    Gbest_nash = -9999.0
    it_sem_melhora = 0
    historico_convergencia = []
    
    for it in range(iteracoes):
        houve_melhora = False
        nash_array = np.zeros(particulas)
        
        for i in range(particulas):
            ks_unico = np.array([X[i, 0]])
            expo_unico = np.array([X[i, 1]])
            
            n_val, _ = simular_cawm_vetorizado_replica(
                np, P, E, Q_obs, mask_calib, area, SUBmax, a_param, ks_unico, expo_unico
            )
            nash_array[i] = n_val[0]
            
            if nash_array[i] > Pbest_nash[i]:
                Pbest_nash[i] = nash_array[i]
                Pbest[i] = X[i]
                if nash_array[i] > Gbest_nash:
                    Gbest_nash = nash_array[i]
                    Gbest = np.copy(X[i])
                    houve_melhora = True

        it_sem_melhora = 0 if houve_melhora else it_sem_melhora + 1
        print(f"      Iteração [{it+1:02d}/{iteracoes}] | Gbest NSE: {Gbest_nash:8.5f}", end="\r")
        historico_convergencia.append(Gbest_nash)
        
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break
            
        for i in range(particulas):
            r1, r2 = np.random.rand(2)
            V[i] = (w * V[i]) + (c1 * r1 * (Pbest[i] - X[i])) + (c2 * r2 * (Gbest - X[i]))
            X[i] = X[i] + V[i]
            X[i, 0] = np.clip(X[i, 0], 0.0, 1.0)
            X[i, 1] = np.clip(X[i, 1], 0.5, 3.0)
            
    return Gbest_nash, historico_convergencia

# ==============================================================================
# 3. MÉTODO VETORIZADO (MATRIZES CPU/GPU)
# ==============================================================================
def pso_vetorizado(xp, P, E, Q_obs, mask_calib, area, SUBmax, a_param, particulas, iteracoes, paciencia, w, c1, c2):
    P_xp = xp.asarray(P)
    E_xp = xp.asarray(E)
    Q_obs_xp = xp.asarray(Q_obs)
    
    X = xp.random.rand(particulas, 2, dtype=xp.float32)
    X[:, 0] = X[:, 0] * 1.0
    X[:, 1] = X[:, 1] * 2.5 + 0.5
    
    V = xp.zeros((particulas, 2), dtype=xp.float32)
    Pbest = xp.copy(X)
    Pbest_nash = xp.full(particulas, -9999.0, dtype=xp.float32)
    
    Gbest = xp.zeros(2, dtype=xp.float32)
    Gbest_nash = -9999.0
    it_sem_melhora = 0
    historico_convergencia = []
    
    for it in range(iteracoes):
        nash_array, _ = simular_cawm_vetorizado_replica(
            xp, P_xp, E_xp, Q_obs_xp, mask_calib, area, SUBmax, a_param, X[:, 0], X[:, 1]
        )
        
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
            
        print(f"      Iteração [{it+1:02d}/{iteracoes}] | Gbest NSE: {Gbest_nash:8.5f}", end="\r")
        historico_convergencia.append(Gbest_nash)
        
        if it_sem_melhora >= paciencia or it == iteracoes - 1:
            print()
            break
            
        R1 = xp.random.rand(particulas, 2, dtype=xp.float32)
        R2 = xp.random.rand(particulas, 2, dtype=xp.float32)
        
        V = (w * V) + (c1 * R1 * (Pbest - X)) + (c2 * R2 * (Gbest - X))
        X = X + V
        X[:, 0] = xp.clip(X[:, 0], 0.0, 1.0)
        X[:, 1] = xp.clip(X[:, 1], 0.5, 3.0)
        
    return Gbest_nash, historico_convergencia

# ==============================================================================
# MOTOR ORQUESTRADOR
# ==============================================================================
def executar_benchmark_final():
    nome_bacia = "Chorozinho"
    iteracoes_max = 20
    paciencia = 10
    rodadas = 10
    
    hiperparametros = [
        {"w": 0.4, "c1": 1.0, "c2": 2.0}, {"w": 0.5, "c1": 1.5, "c2": 1.5},
        {"w": 0.6, "c1": 2.0, "c2": 1.0}, {"w": 0.7, "c1": 1.5, "c2": 2.0},
        {"w": 0.8, "c1": 2.0, "c2": 2.0}, {"w": 0.9, "c1": 1.0, "c2": 1.0},
        {"w": 0.9, "c1": 0.5, "c2": 2.5}, {"w": 0.7, "c1": 2.5, "c2": 0.5},
        {"w": 0.5, "c1": 2.0, "c2": 2.0}, {"w": 0.8, "c1": 1.5, "c2": 1.5},
        {"w": 0.6, "c1": 1.0, "c2": 1.0}, {"w": 0.4, "c1": 2.0, "c2": 1.5}
    ]
    
    P, E, Q_obs, mask_calib, area, SUBmax, a_param = extrair_dados_bacia(nome_bacia)
    
    # Define qual backend será usado para a busca rápida (GPU se existir, senão CPU)
    backend_busca = cp if HAS_GPU else np
    nome_busca = "Matricial GPU" if HAS_GPU else "Matricial CPU"

    print("\n" + "="*85)
    print(f" FASE 1: GRID SEARCH EXPRESSO VIA {nome_busca.upper()}")
    print(f" Procurando a melhor combinação entre {len(hiperparametros)} opções...")
    print("="*85)

    melhor_combo = None
    maior_nash_medio = -9999.0
    
    for idx, params in enumerate(hiperparametros, 1):
        w, c1, c2 = params["w"], params["c1"], params["c2"]
        print(f"  [{idx:02d}/12] Testando w={w}, c1={c1}, c2={c2}...", end=" ")
        
        nashes_combo = []
        for _ in range(rodadas):
            nash, _ = pso_vetorizado(backend_busca, P, E, Q_obs, mask_calib, area, SUBmax, a_param, 2000, iteracoes_max, paciencia, w, c1, c2)
            nashes_combo.append(nash)
            
        media_combo = np.mean(nashes_combo)
        print(f"-> NSE Médio: {media_combo:.5f}")
        
        if media_combo > maior_nash_medio:
            maior_nash_medio = media_combo
            melhor_combo = params

    w_opt, c1_opt, c2_opt = melhor_combo["w"], melhor_combo["c1"], melhor_combo["c2"]

    print("\n" + "="*85)
    print(f" FASE 2: BENCHMARK COMPLETO NA MELHOR COMBINAÇÃO")
    print(f" Hiperparâmetros Vencedores: w={w_opt}, c1={c1_opt}, c2={c2_opt}")
    print("="*85)

    configuracoes = [
        ("Escalar (Legado)", pso_escalar, None, 50),
        ("Matricial CPU", pso_vetorizado, np, 2000),
        ("Matricial GPU", pso_vetorizado, cp if HAS_GPU else None, 2000)
    ]
    
    resultados_resumo = []
    dados_convergencia = [] 
    
    for metodo_nome, funcao_run, xp_backend, num_particulas in configuracoes:
        if xp_backend is None and metodo_nome == "Matricial GPU":
            continue
            
        print(f"\n▶ Executando: {metodo_nome} (Partículas: {num_particulas})")
        tempos, nashes = [], []
        
        for rodada in range(1, rodadas + 1):
            inicio = time.perf_counter()
            
            if xp_backend is None:
                nash, historico = funcao_run(P, E, Q_obs, mask_calib, area, SUBmax, a_param, num_particulas, iteracoes_max, paciencia, w_opt, c1_opt, c2_opt)
            else:
                nash, historico = funcao_run(xp_backend, P, E, Q_obs, mask_calib, area, SUBmax, a_param, num_particulas, iteracoes_max, paciencia, w_opt, c1_opt, c2_opt)
                if xp_backend == cp:
                    cp.cuda.Stream.null.synchronize()
                    
            fim = time.perf_counter()
            tempos.append(fim - inicio)
            nashes.append(nash)
            
            for it_num, valor_nash in enumerate(historico, start=1):
                dados_convergencia.append({
                    "Método": metodo_nome,
                    "Rodada": rodada,
                    "Iteração": it_num,
                    "Gbest_NSE": valor_nash
                })
                
        tempo_medio = np.mean(tempos)
        tempo_projetado = tempo_medio * len(hiperparametros)
        
        resultados_resumo.append({
            "Método": metodo_nome,
            "Partículas": num_particulas,
            "Tempo/Rodada (s)": tempo_medio,
            "Estimativa Grid Completo (s)": tempo_projetado,
            "NSE Médio": np.mean(nashes),
            "Melhor NSE": np.max(nashes)
        })

    df_resumo = pd.DataFrame(resultados_resumo)
    print("\n" + "="*85)
    print(" RESUMO FINAL DO BENCHMARK (Projeção para 12 Combinações x 10 Rodadas)")
    print("="*85)
    print(df_resumo.to_string(index=False))
    df_resumo.to_csv("benchmark_cawm_replica_resumo.csv", index=False)
    
    df_convergencia = pd.DataFrame(dados_convergencia)
    df_convergencia.to_csv("benchmark_cawm_replica_convergencia.csv", index=False)
    
    print("\nArquivos gerados com sucesso!")

if __name__ == "__main__":
    executar_benchmark_final()
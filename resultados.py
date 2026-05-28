import time
import numpy as np
import pandas as pd
from typing import Tuple

# Tenta importar CuPy para a GPU
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Aviso: CuPy não encontrado. Teste de GPU será ignorado.")

from db_init import initialize_db
from db_models import Bacia, PrecipitationDaily, EvaporationMonthly, FlowDaily

# ==============================================================================
# 1. FUNÇÃO DE EXTRAÇÃO DE DADOS (DB)
# ==============================================================================
def extrair_dados_bacia(nome_bacia: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    print(f"[{time.strftime('%H:%M:%S')}] Extraindo dados do banco para a bacia: {nome_bacia}...")
    
    # === CORREÇÃO DA SESSÃO (Tupla vs Sessão Direta) ===
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
    # ===================================================
    
    bacia = session.query(Bacia).filter(Bacia.nome == nome_bacia).first()
    if not bacia:
        raise ValueError(f"Bacia {nome_bacia} não encontrada no banco.")
        
    area_km2 = bacia.area_km2
    
    precip = session.query(PrecipitationDaily).filter_by(bacia_id=bacia.id).order_by(PrecipitationDaily.data).all()
    vazao = session.query(FlowDaily).filter_by(bacia_id=bacia.id).order_by(FlowDaily.data).all()
    evap_mensal = session.query(EvaporationMonthly).filter_by(bacia_id=bacia.id).all()
    session.close()
    
    P = np.array([p.valor for p in precip], dtype=np.float32)
    # Cria um dicionário para busca rápida (Mês -> Valor da Evaporação)
    mapa_evap = {e.mes: e.valor for e in evap_mensal}
    
    # Monta o array E (Evaporação Diária) olhando para o mês de cada dia de precipitação
    E_lista = []
    for p in precip:
        mes_atual = p.data.month # Extrai o mês da data daquele dia
        # Pega a evaporação do mês correspondente (se não achar, usa 5.0 como segurança)
        E_lista.append(mapa_evap.get(mes_atual, 5.0)) 
        
    E = np.array(E_lista, dtype=np.float32)
    Q_obs = np.array([q.valor if q.valor is not None else 0.0 for q in vazao], dtype=np.float32)
    
    min_len = min(len(P), len(Q_obs))
    return P[:min_len], E[:min_len], Q_obs[:min_len], area_km2

# ==============================================================================
# 2. MÉTODO 1: ESCALAR (LEGADO)
# ==============================================================================
def pso_escalar(P, E, Q_obs, area, particulas, iteracoes, paciencia, w, c1, c2):
    dias = len(P)
    vol_obs = np.sum(Q_obs)
    media_obs = np.mean(Q_obs)
    
    X = np.random.rand(particulas, 3)
    V = np.zeros((particulas, 3))
    Pbest = np.copy(X)
    Pbest_nash = np.full(particulas, -9999.0)
    
    Gbest = np.zeros(3)
    Gbest_nash = -9999.0
    Gbest_pbias = 0.0
    Gbest_rmse = 0.0
    
    it_sem_melhora = 0
    
    for it in range(iteracoes):
        houve_melhora_na_iteracao = False
        
        for i in range(particulas):
            Q_calc_raw = np.zeros(dias)
            S_solo = 0.0
            ks, a, expo = X[i]
            
            for d in range(dias):
                evap_direta = min(P[d], E[d])
                chuva_liq = P[d] - evap_direta
                S_solo += chuva_liq * ks
                escoamento = max(0, S_solo - (E[d] * a))
                S_solo -= escoamento
                Q_calc_raw[d] = escoamento * (1.0 / (expo + 0.1))
                
            vol_calc_raw = np.sum(Q_calc_raw)
            kl = vol_obs / (vol_calc_raw + 1e-9)
            Q_calc = Q_calc_raw * kl
            
            numerador = np.sum((Q_obs - Q_calc)**2)
            denominador = np.sum((Q_obs - media_obs)**2)
            nash = 1 - (numerador / (denominador + 1e-9))
            
            if nash > Pbest_nash[i]:
                Pbest_nash[i] = nash
                Pbest[i] = X[i]
                if nash > Gbest_nash:
                    Gbest_nash = nash
                    Gbest = np.copy(X[i])
                    Gbest_pbias = np.sum(Q_calc - Q_obs) / (vol_obs + 1e-9) * 100
                    Gbest_rmse = np.sqrt(np.mean((Q_obs - Q_calc)**2))
                    houve_melhora_na_iteracao = True
                    
        # Sistema de Paciência (Early Stopping)
        if houve_melhora_na_iteracao:
            it_sem_melhora = 0
        else:
            it_sem_melhora += 1
            
        # === NOVO: PRINT DE ACOMPANHAMENTO DA ITERAÇÃO ===
        print(f"      Iteração [{it+1:02d}/{iteracoes}] | Gbest NSE: {Gbest_nash:8.4f} | Paciência: {it_sem_melhora}/{paciencia}", end="\r")
        if it == iteracoes - 1 or it_sem_melhora >= paciencia:
            print() # Quebra a linha para o próximo teste não sobrepor
        # =================================================
            
        if it_sem_melhora >= paciencia:
            break # Interrompe o PSO prematuramente
                    
        for i in range(particulas):
            r1, r2 = np.random.rand(2)
            V[i] = (w * V[i]) + (c1 * r1 * (Pbest[i] - X[i])) + (c2 * r2 * (Gbest - X[i]))
            X[i] = X[i] + V[i]
            
    return Gbest_nash, Gbest_pbias, Gbest_rmse

# ==============================================================================
# 3. MÉTODO 2 e 3: VETORIZADO (MATRIZES SIMT)
# ==============================================================================
def pso_vetorizado(xp, P, E, Q_obs, area, particulas, iteracoes, paciencia, w, c1, c2):
    P_xp = xp.asarray(P)
    E_xp = xp.asarray(E)
    Q_obs_xp = xp.asarray(Q_obs)
    dias = len(P)
    
    vol_obs = xp.sum(Q_obs_xp)
    media_obs = xp.mean(Q_obs_xp)
    
    X = xp.random.rand(particulas, 3).astype(xp.float32)
    V = xp.zeros((particulas, 3), dtype=xp.float32)
    Pbest = xp.copy(X)
    Pbest_nash = xp.full(particulas, -9999.0, dtype=xp.float32)
    
    Gbest = xp.zeros(3, dtype=xp.float32)
    Gbest_nash = -9999.0
    Gbest_pbias = 0.0
    Gbest_rmse = 0.0
    
    E_matriz = xp.broadcast_to(E_xp, (particulas, dias))
    
    it_sem_melhora = 0
    
    for it in range(iteracoes):
        ks = X[:, 0:1]
        a = X[:, 1:2]
        expo = X[:, 2:3]
        
        evap_direta = xp.minimum(P_xp, E_xp)
        chuva_liq = P_xp - evap_direta
        chuva_liq_matriz = xp.broadcast_to(chuva_liq, (particulas, dias))
        
        S_solo_virtual = chuva_liq_matriz * ks
        escoamento = xp.maximum(0, S_solo_virtual - (E_matriz * a))
        Q_calc_raw = escoamento * (1.0 / (expo + 0.1))
        
        vol_calc_raw = xp.sum(Q_calc_raw, axis=1)
        kl = vol_obs / (vol_calc_raw + 1e-9)
        Q_calc = Q_calc_raw * kl[:, None]
        
        numerador = xp.sum((Q_obs_xp - Q_calc)**2, axis=1)
        denominador = xp.sum((Q_obs_xp - media_obs)**2)
        nash_array = 1 - (numerador / (denominador + 1e-9))
        
        melhorias = nash_array > Pbest_nash
        Pbest_nash = xp.where(melhorias, nash_array, Pbest_nash)
        Pbest = xp.where(melhorias[:, None], X, Pbest)
        
        idx_max = xp.argmax(Pbest_nash)
        max_nash = float(Pbest_nash[idx_max])
        
        # Sistema de Paciência (Early Stopping)
        if max_nash > Gbest_nash:
            Gbest_nash = max_nash
            Gbest = xp.copy(Pbest[idx_max])
            
            melhor_q_calc = Q_calc[idx_max]
            Gbest_pbias = float(xp.sum(melhor_q_calc - Q_obs_xp) / (vol_obs + 1e-9) * 100)
            Gbest_rmse = float(xp.sqrt(xp.mean((Q_obs_xp - melhor_q_calc)**2)))
            it_sem_melhora = 0
        else:
            it_sem_melhora += 1
            
        # === NOVO: PRINT DE ACOMPANHAMENTO DA ITERAÇÃO ===
        print(f"      Iteração [{it+1:02d}/{iteracoes}] | Gbest NSE: {Gbest_nash:8.4f} | Paciência: {it_sem_melhora}/{paciencia}", end="\r")
        if it == iteracoes - 1 or it_sem_melhora >= paciencia:
            print() # Quebra a linha para o próximo teste não sobrepor
        # =================================================
            
        if it_sem_melhora >= paciencia:
            break # Interrompe o PSO prematuramente
            
        R1 = xp.random.rand(particulas, 3).astype(xp.float32)
        R2 = xp.random.rand(particulas, 3).astype(xp.float32)
        
        V = (w * V) + (c1 * R1 * (Pbest - X)) + (c2 * R2 * (Gbest - X))
        X = X + V
        
    return Gbest_nash, Gbest_pbias, Gbest_rmse

# ==============================================================================
# 4. MOTOR DE BENCHMARK
# ==============================================================================
def executar_benchmark():
    nome_bacia = "Chorozinho"
    iteracoes_max = 20
    paciencia = 5 # Para se o Nash global não melhorar por 5 iterações seguidas
    rodadas_por_combo = 10
    
    hiperparametros = [
        {"w": 0.4, "c1": 1.0, "c2": 2.0},
        {"w": 0.5, "c1": 1.5, "c2": 1.5},
        {"w": 0.6, "c1": 2.0, "c2": 1.0},
        {"w": 0.7, "c1": 1.5, "c2": 2.0},
        {"w": 0.8, "c1": 2.0, "c2": 2.0},
        {"w": 0.9, "c1": 1.0, "c2": 1.0},
        {"w": 0.9, "c1": 0.5, "c2": 2.5},
        {"w": 0.7, "c1": 2.5, "c2": 0.5},
        {"w": 0.5, "c1": 2.0, "c2": 2.0},
        {"w": 0.8, "c1": 1.5, "c2": 1.5},
        {"w": 0.6, "c1": 1.0, "c2": 1.0},
        {"w": 0.4, "c1": 2.0, "c2": 1.5}
    ]
    
    try:
        P, E, Q_obs, area = extrair_dados_bacia(nome_bacia)
    except Exception as e:
        print(f"Erro na extração de dados: {e}")
        return

    resultados = []
    print("\n" + "="*85)
    print(f" INICIANDO BENCHMARK: {nome_bacia.upper()} ")
    print(f" Máx Iterações: {iteracoes_max} | Paciência: {paciencia} | Rodadas/Combo: {rodadas_por_combo}")
    print("="*85)

    for i, params in enumerate(hiperparametros, 1):
        w, c1, c2 = params["w"], params["c1"], params["c2"]
        print(f"\n[{i}/12] Testando Combo: w={w}, c1={c1}, c2={c2}")
        
        # AQUI DEFINIMOS AS CONDIÇÕES DIFERENTES PARA CADA MÉTODO
        configuracoes = [
            ("Escalar (Legado)", pso_escalar, None, 50),          # 50 Partículas
            ("Matricial CPU", pso_vetorizado, np, 2000),          # 2000 Partículas
            ("Matricial GPU", pso_vetorizado, cp if HAS_GPU else None, 2000) # 2000 Partículas
        ]
        
        for metodo_nome, funcao_run, xp_backend, num_particulas in configuracoes:
            if xp_backend is None and metodo_nome == "Matricial GPU":
                continue 
                
            tempos, nashes, pbiases, rmses = [], [], [], []
            
            for _ in range(rodadas_por_combo):
                inicio = time.perf_counter()
                
                if xp_backend is None:
                    nash, pbias, rmse = funcao_run(P, E, Q_obs, area, num_particulas, iteracoes_max, paciencia, w, c1, c2)
                else:
                    nash, pbias, rmse = funcao_run(xp_backend, P, E, Q_obs, area, num_particulas, iteracoes_max, paciencia, w, c1, c2)
                    if xp_backend == cp:
                        cp.cuda.Stream.null.synchronize()
                        
                fim = time.perf_counter()
                
                tempos.append(fim - inicio)
                nashes.append(nash)
                pbiases.append(pbias)
                rmses.append(rmse)
                
            resultados.append({
                "Combo": f"{w}-{c1}-{c2}",
                "Método": metodo_nome,
                "Partículas": num_particulas,
                "Tempo(s)": np.mean(tempos),
                "NSE": np.mean(nashes),
                "PBIAS(%)": np.mean(pbiases),
                "RMSE": np.mean(rmses)
            })
            
            print(f"  -> {metodo_nome:17s} (P={num_particulas:<4}) | Tempo: {np.mean(tempos):.4f}s | NSE: {np.mean(nashes):.4f}")

    df_resultados = pd.DataFrame(resultados)
    print("\n" + "="*85)
    print(" RESUMO FINAL DO BENCHMARK")
    print("="*85)
    print(df_resultados.to_string(index=False))
    df_resultados.to_csv("benchmark_cawm_assimetrico.csv", index=False)
    print("\nResultados completos salvos em 'benchmark_cawm_assimetrico.csv'")

if __name__ == "__main__":
    executar_benchmark()
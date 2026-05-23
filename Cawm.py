from cawm_config import get_array_backend, get_base_paths
from funcoes import (
    mapear_e_verificar_bacias,
    extrair_parametros_shapes,
    carregar_series_temporais,
    validar_bacias_antes_pso,
    _mascara_intervalo,
    calibrar_ks_kl_pso,
    calibrar_parametros_pso,
)
import time
import numpy as np

combinacoes_teste = [
        (0.8, 1.0, 2.0), # Original
        (0.7, 1.5, 1.5), # Equilibrado
        (0.9, 2.0, 1.0), # Alta inércia e foco na própria memória
        (0.5, 1.0, 2.5), # Baixa inércia e alto foco no melhor global
        (0.6, 2.0, 2.0)  # Conservador
    ]
N_PARTICULAS = 6500
MAX_ITER = 10
LIMITES_PADRAO = {
    "ks": (0.0, 1.0),
    "kl": (0.0, 2.0),
}
LIMITES_RECALIBRACAO = {
    "ks": LIMITES_PADRAO["ks"],
    "kl": LIMITES_PADRAO["kl"],
    "a": (0.0, 2.0),
    "expo_perdas": (0.5, 3.0),
}


def executar_calibracao_completa(nome_bacia, dados, xp):
    """Executa calibração em 2 fases: primeira com 2 parametros, depois com 4.
    Retorna o melhor resultado de todas as combinações."""
    
    print(f"\n🧪 {nome_bacia}: Iniciando calibração em 2 fases (2 métodos x 2 fases)")
    
    todos_resultados = {}
    
    # ====== FASE 1: Calibração de Ks e Kl ======
    print(f"\n📌 FASE 1: Calibrando Ks e Kl...")
    for metodo in ("estado_inicial", "spinup"):
        print(f"  🧩 Método: {metodo}")
        mascara_calibracao = _mascara_intervalo(len(dados["chuva"]), dados["intervalos"], usar_linha=0)
        
        inicio_metodo = time.perf_counter()
        resultado = calibrar_ks_kl_pso(
            nome_bacia=nome_bacia,
            parametros=dados["parametros"],
            chuva=dados["chuva"],
            evaporacao=dados["evaporacao"],
            vazao_obs=dados["vazao_obs"],
            xp=xp,
            metodo=metodo,
            n_particulas=N_PARTICULAS,
            max_iter=MAX_ITER,
            coeficientes=combinacoes_teste[0],
            limites=LIMITES_PADRAO,
            mascara_calibracao=mascara_calibracao,
        )
        tempo_total = time.perf_counter() - inicio_metodo
        resultado["tempo_total_segundos"] = tempo_total
        resultado["fase"] = 1
        
        todos_resultados[f"fase1_{metodo}"] = resultado
        print(
            f"    ✅ NSE = {resultado['nse']:.5f} | "
            f"Ks = {resultado['ks']:.6f} | Kl = {resultado['kl']:.6f} | "
            f"tempo = {tempo_total:.2f}s"
        )
    
    # Pega os valores do melhor de fase 1 para inicializar fase 2
    melhor_fase1 = max(todos_resultados.values(), key=lambda x: x["nse"])
    valor_inicial_a = dados["parametros"].get("a", 0.5)
    valor_inicial_expo = dados["parametros"].get("expo_perdas", 1.5)
    
    # ====== FASE 2: Calibração de Ks, Kl, a e expo_perdas ======
    print(f"\n📌 FASE 2: Recalibrando com 4 parâmetros (Ks, Kl, a, expo_perdas)...")
    for metodo in ("estado_inicial", "spinup"):
        print(f"  🧩 Método: {metodo}")
        mascara_calibracao = _mascara_intervalo(len(dados["chuva"]), dados["intervalos"], usar_linha=0)
        
        inicio_metodo = time.perf_counter()
        resultado = calibrar_parametros_pso(
            nome_bacia=nome_bacia,
            parametros=dados["parametros"],
            chuva=dados["chuva"],
            evaporacao=dados["evaporacao"],
            vazao_obs=dados["vazao_obs"],
            xp=xp,
            metodo=metodo,
            nomes_parametros=("ks", "kl", "a", "expo_perdas"),
            limites=LIMITES_RECALIBRACAO,
            n_particulas=N_PARTICULAS,
            max_iter=MAX_ITER,
            coeficientes=combinacoes_teste[0],
            mascara_calibracao=mascara_calibracao,
            valores_iniciais={
                "ks": melhor_fase1["ks"],
                "kl": melhor_fase1["kl"],
                "a": valor_inicial_a,
                "expo_perdas": valor_inicial_expo,
            },
        )
        tempo_total = time.perf_counter() - inicio_metodo
        resultado["tempo_total_segundos"] = tempo_total
        resultado["fase"] = 2
        
        todos_resultados[f"fase2_{metodo}"] = resultado
        print(
            f"    ✅ NSE = {resultado['nse']:.5f} | "
            f"Ks = {resultado['ks']:.6f} | Kl = {resultado['kl']:.6f} | "
            f"A = {resultado.get('a', np.nan):.6f} | Expo = {resultado.get('expo_perdas', np.nan):.6f} | "
            f"tempo = {tempo_total:.2f}s"
        )
    
    # Seleciona o melhor de todos os 4
    melhor_geral = max(todos_resultados.values(), key=lambda x: x["nse"])
    melhor_geral["todos_resultados"] = todos_resultados
    
    return melhor_geral, todos_resultados

def main():
    xp, usar_gpu = get_array_backend(prefer_gpu=True)
    pastas = get_base_paths()
    dicionario_bacias = mapear_e_verificar_bacias(pastas)
    extrair_parametros_shapes(dicionario_bacias)

    # Validar todas as bacias antes de alocar séries e iniciar PSO
    print("\n🔎 Validando bacias antes do PSO...")
    validas, invalidas = validar_bacias_antes_pso(dicionario_bacias)
    if invalidas:
        print("\n⚠️ Foram encontradas bacias com problemas (serão ignoradas):")
        for nb, info in invalidas.items():
            print(f"  - {nb}: {info.get('motivo')}")
    # Filtra dicionário para manter apenas as válidas
    dicionario_bacias = {k: v for k, v in dicionario_bacias.items() if k in validas}

    print(f"\n🎯 Total de bacias prontas para processamento: {len(dicionario_bacias)}")
    
    print("\n📅 Carregando as séries temporais (CSVs)...")
    bacias_prontas = {} # Novo dicionário só com o que deu 100% certo

    for nome_bacia, dados in dicionario_bacias.items():
        if "parametros" in dados: 
            vazao_obs, chuva, evaporacao, datas, intervalos = carregar_series_temporais(nome_bacia, dados, xp)
            
            if chuva is not None:
                # Armazenamos tudo num "pacote" só para a simulação
                bacias_prontas[nome_bacia] = {
                    "parametros": dados["parametros"],
                    "vazao_obs": vazao_obs,
                    "chuva": chuva,
                    "evaporacao": evaporacao,
                    "datas": datas,
                    "intervalos": intervalos,
                    "pasta_bacia": dados["pasta_bacia"],
                }
                print(f"  ✅ {nome_bacia}: Séries carregadas! ({len(chuva)} dias na memória)")

    print(f"\n🚀 Tudo pronto! {len(bacias_prontas)} bacias armadas para a simulação CAWMe.")

    if not bacias_prontas:
        print("⚠️ Nenhuma bacia válida para calibração.")
        return

    print("\n🧪 Iniciando calibração em 2 fases para cada bacia...")
    resultados_finais = {}

    for nome_bacia, dados in bacias_prontas.items():
        melhor_geral, todos_resultados = executar_calibracao_completa(
            nome_bacia=nome_bacia,
            dados=dados,
            xp=xp,
        )
        resultados_finais[nome_bacia] = {
            "melhor_geral": melhor_geral,
            "todos_resultados": todos_resultados,
        }

    print("\n📊 Resumo final da calibração:")
    for nome_bacia, resultado_bacia in resultados_finais.items():
        melhor_geral = resultado_bacia["melhor_geral"]
        todos_resultados = resultado_bacia["todos_resultados"]
        
        print(f"\n  • {nome_bacia}:")
        print(f"    🏆 Melhor resultado geral:")
        print(
            f"      Fase {melhor_geral['fase']} / {melhor_geral['metodo']}: NSE = {melhor_geral['nse']:.5f} | "
            f"Ks = {melhor_geral['ks']:.6f} | Kl = {melhor_geral['kl']:.6f}"
        )
        if melhor_geral['fase'] == 2:
            print(
                f"      A = {melhor_geral.get('a', np.nan):.6f} | "
                f"Expo = {melhor_geral.get('expo_perdas', np.nan):.6f}"
            )
        
        print(f"    📋 Detalhes de todas as calibrações:")
        for chave, resultado in sorted(todos_resultados.items()):
            fase = resultado["fase"]
            metodo = resultado["metodo"]
            print(
                f"      Fase {fase} / {metodo}: NSE = {resultado['nse']:.5f} | "
                f"Ks = {resultado['ks']:.6f} | Kl = {resultado['kl']:.6f}"
            )
            if fase == 2:
                print(
                    f"        A = {resultado.get('a', np.nan):.6f} | "
                    f"Expo = {resultado.get('expo_perdas', np.nan):.6f}"
                )


if __name__ == "__main__":
    main()


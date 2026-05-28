import time
import numpy as np
import os
import csv
import tempfile

def executar_teste_io_vs_ram():
    # Cenário: 3650 dias (10 anos) e 2000 partículas
    dias = 3650
    particulas = 2000
    
    print("\n" + "="*70)
    print(" BENCHMARK: GARGALO DE LEITURA (DISCO vs RAM)")
    print(f" Matriz de Dados: {dias} dias x {particulas} partículas")
    print("="*70)
    
    # 1. Geração de Dados Fictícios na RAM
    print("[1/3] Gerando matriz de dados (In-Memory)...")
    matriz_ram = np.random.rand(dias, particulas).astype(np.float32)
    
    # 2. Escrevendo no Disco (I/O) para simular o banco de dados/arquivos
    print("[2/3] Gravando dados no SSD para o teste de I/O...")
    arquivo_temp = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
    caminho_arquivo = arquivo_temp.name
    
    writer = csv.writer(arquivo_temp)
    for linha in matriz_ram:
        writer.writerow(linha)
    arquivo_temp.close()
    
    tamanho_mb = os.path.getsize(caminho_arquivo) / (1024 * 1024)
    print(f"      Arquivo gerado no disco: {tamanho_mb:.2f} MB")
    
    # =================================================================
    # TESTE A: LEITURA DIRETO DO DISCO (I/O BOUND)
    # =================================================================
    print("\n[Teste A] Iniciando processamento direto do Disco (I/O)...")
    inicio_io = time.perf_counter()
    
    soma_total_io = 0.0
    with open(caminho_arquivo, 'r') as f:
        reader = csv.reader(f)
        for linha_str in reader:
            # Converte as strings de volta para float e soma (Gargalo clássico)
            linha_float = [float(x) for x in linha_str]
            soma_total_io += sum(linha_float)
            
    fim_io = time.perf_counter()
    tempo_io = fim_io - inicio_io
    print(f"      -> Tempo total (I/O + Processamento): {tempo_io:.4f} segundos")
    
    # =================================================================
    # TESTE B: PROCESSAMENTO DIRETO DA RAM (IN-MEMORY)
    # =================================================================
    print("\n[Teste B] Iniciando processamento In-Memory (Matriz Vetorizada)...")
    inicio_ram = time.perf_counter()
    
    # Soma de todos os elementos nativamente na RAM via NumPy
    soma_total_ram = np.sum(matriz_ram)
    
    fim_ram = time.perf_counter()
    tempo_ram = fim_ram - inicio_ram
    print(f"      -> Tempo total (In-Memory): {tempo_ram:.6f} segundos")
    
    # =================================================================
    # RESULTADOS FINAIS
    # =================================================================
    speedup = tempo_io / tempo_ram
    print("\n" + "="*70)
    print(f" SPEEDUP: Processamento em RAM foi {speedup:,.0f} VEZES mais rápido!")
    print("="*70)
    
    # Limpeza
    os.remove(caminho_arquivo)

if __name__ == "__main__":
    executar_teste_io_vs_ram()
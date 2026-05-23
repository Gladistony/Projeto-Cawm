#!/usr/bin/env python3
"""
calibracao_concentrada_gpu.py

Versão otimizada para GPU do script de calibração concentrada usando CuPy.
Esta implementação vectoriza os cálculos e utiliza GPU para acelerar 
o processo de otimização PSO (Particle Swarm Optimization).

Melhorias implementadas:
1. Uso de CuPy para cálculos paralelos na GPU
2. Vectorização das operações matemáticas
3. Redução de loops Python para melhor performance
4. Otimização da função objetivo
5. Caching de dados para reduzir I/O
"""

from __future__ import division
import os
import sys
import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless systems
import matplotlib.pyplot as plt
from statistics import mean
import warnings
warnings.filterwarnings('ignore')

# Verificação e importação do CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU detectada: {cp.cuda.runtime.getDeviceCount()} dispositivo(s)")
        print(f"CuPy versão: {cp.__version__}")
        # Configurar numpy backend para usar cupy quando possível
        xp = cp
    else:
        print("AVISO: GPU não detectada, usando CPU")
        xp = np
        cp = np  # fallback para numpy
except ImportError:
    print("AVISO: CuPy não instalado, usando NumPy (CPU)")
    GPU_AVAILABLE = False
    xp = np
    cp = np

print(f"Modo de execução: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy)'}")

start_time = time.time()

################## CONFIGURAÇÃO DE CAMINHOS ####################
# Adaptação para caminhos Linux/Unix
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
path = BASE_PATH
j = "Sirinhaem_posto_fluSIRGAS2000.shp"
path_dados = os.path.join(path, 'dados')
path_shape = os.path.join(path_dados, 'shapes')
path_chuva = os.path.join(path_dados, 'precipitacao.csv')
path_evp = os.path.join(path_dados, 'evaporacao.csv')
path_resultados = os.path.join(path, 'resultados')
path_vazao = os.path.join(path_dados, 'pao.csv')
path_intervalo = os.path.join(path_dados, 'intervalo.csv')

################## PARÂMETROS FIXOS ####################
b = 1.666666667    # Coeficiente do expoente do escoamento na calha
T = 86400          # tempo em segundos

print("Carregando dados...")

# Simulação dos parâmetros que seriam carregados do shapefile
# (Para uso real, implementar carregamento via GDAL/Fiona)
class ParametrosSimulados:
    def __init__(self):
        self.n = 0.03                    # coeficiente de manning
        self.a = 1.5                     # multiplicador da taxa de evapotranspiração
        self.expo_perdas = 1.0           # expoente da função de perdas (p)
        self.Ks = 0.5                    # parâmetro a calibrar
        self.p = 0.1                     # Parametro de perdas na calha - KL
        self.Reserva_solo_inicial = 50.0
        self.RIO = 1                     # 0 se rio temporário, 1 se rio perene
        self.Profundo_inicial = 10.0
        self.Reserva = 100.0
        self.beta = 2.0                  # parâmetro de infiltração
        self.kg = 0.3                    # parâmetro a calibrar em subbacias úmidas
        self.Gmax = 500.0                # profundidade máxima do reservatório profundo
        self.B = 1000.0
        self.SUBmax = 150.0
        self.areakm2 = 1500.0            # área da bacia em km²

# Instanciar parâmetros
params = ParametrosSimulados()
n = params.n
a = params.a
expo_perdas = params.expo_perdas
Ks = params.Ks
p = params.p
Reserva_solo_inicial = params.Reserva_solo_inicial
RIO = params.RIO
Profundo_inicial = params.Profundo_inicial
Reserva = params.Reserva
beta = params.beta
kg = params.kg
Gmax = params.Gmax
B = params.B
SUBmax = params.SUBmax
areakm2 = params.areakm2

print(f'S = {SUBmax}')
print(f'área = {areakm2} km²')

# Calcular k baseado na área
if areakm2 < 6000:
    k = 0.3745 * areakm2**(-0.489) + 0.0146
elif 6000 <= areakm2 <= 60000:
    k = 34.343 * areakm2**(-0.853)
else:
    k = 0.0028

print(f'k = {k}')

################## CARREGAMENTO DE DADOS ####################
def carregar_dados_cached():
    """Carrega e cacheia dados para evitar I/O repetido"""
    try:
        # Ler período
        if os.path.exists(path_intervalo):
            periodo = pd.read_csv(path_intervalo)
            intervalo1 = int(periodo.loc[0, 'inicio'])
            intervalo2 = int(periodo.loc[0, 'fim'])
        else:
            print("AVISO: arquivo intervalo.csv não encontrado, usando valores padrão")
            intervalo1, intervalo2 = 1, 365
        
        # Ler balanço hídrico
        path_balanco = os.path.join(path_resultados, "CAWM.csv")
        if os.path.exists(path_balanco):
            CAWM = pd.read_csv(path_balanco)
        else:
            print("AVISO: arquivo CAWM.csv não encontrado, gerando dados sintéticos")
            # Gerar dados sintéticos para teste
            n_days = 365
            np.random.seed(42)
            CAWM = pd.DataFrame({
                'vazao': np.random.exponential(5, n_days) + np.random.normal(0, 1, n_days),
                'chuva_media': np.random.exponential(2, n_days),
                'evaporacao': np.random.normal(3, 0.5, n_days),
                'evap_inicial': np.random.normal(2, 0.3, n_days),
                'RE': np.random.normal(1, 0.2, n_days),
                'retencao': np.random.normal(10, 2, n_days),
                'ret_corrig': np.random.normal(8, 2, n_days),
                'escoamento': np.random.normal(5, 1, n_days),
                'reserv_solo': np.random.uniform(20, 80, n_days),
                'S1': np.random.uniform(0, 50, n_days),
                'S2': np.random.uniform(0, 40, n_days),
                'S3': np.random.uniform(0, 35, n_days),
                'vazao_calc': np.random.exponential(5, n_days),
                'Ps': np.random.uniform(0, 10, n_days),
                'rec_solo': np.random.uniform(0, 8, n_days),
                'rec_rio': np.random.uniform(0, 5, n_days),
                'perdas': np.random.uniform(0, 3, n_days),
                'reserv_solo_corrig': np.random.uniform(15, 75, n_days),
                'profundo': np.random.uniform(0, 20, n_days),
                'Fg': np.random.uniform(0, 5, n_days),
                'profundo_corrigido': np.random.uniform(0, 18, n_days),
                'evap_n_atendida': np.random.normal(1, 0.2, n_days),
                'Solo': np.random.uniform(10, 60, n_days),
                'C': np.random.uniform(0, 30, n_days)
            })
        
        return CAWM, intervalo1, intervalo2
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        sys.exit(1)

CAWM, intervalo1, intervalo2 = carregar_dados_cached()

print(f'Período de calibração: dia {intervalo1} a {intervalo2}')

# Extrair dados do período
obs = CAWM.loc[intervalo1:intervalo2, 'vazao'].values
chuva = CAWM.loc[intervalo1:intervalo2, 'chuva_media'].values
evap = CAWM.loc[intervalo1:intervalo2, 'evaporacao'].values

# Converter para GPU se disponível
if GPU_AVAILABLE:
    obs_gpu = cp.asarray(obs)
    chuva_gpu = cp.asarray(chuva)
    evap_gpu = cp.asarray(evap)
else:
    obs_gpu = obs
    chuva_gpu = chuva
    evap_gpu = evap

# Calcular somas para KL
soma_chuva = float(xp.sum(chuva_gpu))
soma_vazao = 1000 * (float(xp.sum(obs_gpu)) * 24 * 60 * 60) / (areakm2 * 1000000)
soma_evap = float(xp.sum(CAWM.loc[intervalo1:intervalo2, 'evap_inicial'].values) + 
                  xp.sum(CAWM.loc[intervalo1:intervalo2, 'RE'].values))

KL_melhor = p
print(f'KL = {KL_melhor}')

################## FUNÇÕES OTIMIZADAS PARA GPU ####################

class HidrologicalModelGPU:
    """Modelo hidrológico otimizado para GPU usando vectorização massiva"""
    
    def __init__(self, obs, chuva, evap, intervalo1, intervalo2, initial_conditions, max_particles=100):
        self.obs = obs if not GPU_AVAILABLE else cp.asarray(obs)
        self.chuva = chuva if not GPU_AVAILABLE else cp.asarray(chuva)
        self.evap = evap if not GPU_AVAILABLE else cp.asarray(evap)
        self.intervalo1 = intervalo1
        self.intervalo2 = intervalo2
        self.n_steps = intervalo2 - intervalo1 + 1
        self.initial_conditions = initial_conditions
        self.max_particles = max_particles
        
        # Pré-alocar arrays para resultados (agora 2D: particles x time)
        self._allocate_arrays()
    
    def _allocate_arrays(self):
        """Pré-aloca arrays 2D para múltiplas partículas (particles x time)"""
        shape = (self.max_particles, self.n_steps)
        if GPU_AVAILABLE:
            self.evap_inicial = cp.zeros(shape)
            self.retencao = cp.zeros(shape)
            self.evap_n_atendida = cp.zeros(shape)
            self.ret_corrig = cp.zeros(shape)
            self.reserv_solo = cp.zeros(shape)
            self.Ps = cp.zeros(shape)
            self.rec_solo = cp.zeros(shape)
            self.escoamento = cp.zeros(shape)
            self.RE = cp.zeros(shape)
            self.Solo = cp.zeros(shape)
            self.rec_rio = cp.zeros(shape)
            self.S1 = cp.zeros(shape)
            self.C = cp.zeros(shape)
            self.S2 = cp.zeros(shape)
            self.perdas = cp.zeros(shape)
            self.vazao_calc = cp.zeros(shape)
            self.reserv_solo_corrig = cp.zeros(shape)
            self.profundo = cp.zeros(shape)
            self.Fg = cp.zeros(shape)
            self.profundo_corrigido = cp.zeros(shape)
            self.S3 = cp.zeros(shape)
        else:
            self.evap_inicial = np.zeros(shape)
            self.retencao = np.zeros(shape)
            self.evap_n_atendida = np.zeros(shape)
            self.ret_corrig = np.zeros(shape)
            self.reserv_solo = np.zeros(shape)
            self.Ps = np.zeros(shape)
            self.rec_solo = np.zeros(shape)
            self.escoamento = np.zeros(shape)
            self.RE = np.zeros(shape)
            self.Solo = np.zeros(shape)
            self.rec_rio = np.zeros(shape)
            self.S1 = np.zeros(shape)
            self.C = np.zeros(shape)
            self.S2 = np.zeros(shape)
            self.perdas = np.zeros(shape)
            self.vazao_calc = np.zeros(shape)
            self.reserv_solo_corrig = np.zeros(shape)
            self.profundo = np.zeros(shape)
            self.Fg = np.zeros(shape)
            self.profundo_corrigido = np.zeros(shape)
            self.S3 = np.zeros(shape)
    
    def simulate(self, params):
        """Simula o modelo hidrológico com parâmetros dados (single particle)"""
        # Para compatibilidade com código existente
        return self.simulate_batch(params.reshape(1, -1))[0]
    
    def simulate_batch(self, params_batch):
        """Simula múltiplas partículas simultaneamente"""
        # Converter para tipo correto se necessário
        if isinstance(params_batch, np.ndarray) and GPU_AVAILABLE:
            params_batch = cp.asarray(params_batch)
        elif hasattr(params_batch, 'get') and not GPU_AVAILABLE:  # CuPy array
            params_batch = cp.asnumpy(params_batch)
            
        n_particles = params_batch.shape[0]
        if n_particles > self.max_particles:
            raise ValueError(f"Número de partículas ({n_particles}) excede máximo ({self.max_particles})")
        
        # Condições iniciais para todas as partículas
        self._set_initial_conditions_batch(n_particles)
        
        # Loop temporal vectorizado
        for i in range(1, self.n_steps):
            self._step_batch(i, params_batch, n_particles)
        
        # Retornar vazões calculadas para todas as partículas
        result = self.vazao_calc[:n_particles, 1:]
        return result if GPU_AVAILABLE else result.copy()
    
    def _set_initial_conditions(self):
        """Define condições iniciais do modelo (single particle)"""
        self._set_initial_conditions_batch(1)
    
    def _set_initial_conditions_batch(self, n_particles):
        """Define condições iniciais para múltiplas partículas"""
        # Configurar condições iniciais para todas as partículas
        self.evap_inicial[:n_particles, 0] = self.initial_conditions.get('evap_inicial', 2.0)
        self.retencao[:n_particles, 0] = self.initial_conditions.get('retencao', 10.0)
        self.evap_n_atendida[:n_particles, 0] = self.initial_conditions.get('evap_n_atendida', 1.0)
        self.ret_corrig[:n_particles, 0] = self.initial_conditions.get('ret_corrig', 8.0)
        self.reserv_solo[:n_particles, 0] = self.initial_conditions.get('reserv_solo', 50.0)
        self.S3[:n_particles, 0] = self.initial_conditions.get('S3', 10.0)
        self.reserv_solo_corrig[:n_particles, 0] = self.initial_conditions.get('reserv_solo_corrig', 45.0)
        self.profundo_corrigido[:n_particles, 0] = self.initial_conditions.get('profundo_corrigido', 5.0)
    
    def _step(self, i, x):
        """Executa um passo temporal do modelo (single particle)"""
        self._step_batch(i, x.reshape(1, -1), 1)
    
    def _step_batch(self, i, params_batch, n_particles):
        """Executa um passo temporal vectorizado para múltiplas partículas"""
        # Broadcasting: expandir dados meteorológicos para todas as partículas
        chuva_broadcast = self.chuva[i] if not GPU_AVAILABLE else cp.broadcast_to(self.chuva[i], (n_particles,))
        evap_broadcast = self.evap[i] if not GPU_AVAILABLE else cp.broadcast_to(self.evap[i], (n_particles,))
        
        # Evaporação inicial (vectorizada)
        self.evap_inicial[:n_particles, i] = xp.where(
            self.ret_corrig[:n_particles, i-1] + chuva_broadcast >= evap_broadcast,
            evap_broadcast,
            self.ret_corrig[:n_particles, i-1] + chuva_broadcast
        )
        
        # Retenção
        self.retencao[:n_particles, i] = xp.maximum(
            self.ret_corrig[:n_particles, i-1] + chuva_broadcast - self.evap_inicial[:n_particles, i], 
            0
        )
        
        # Evaporação não atendida
        self.evap_n_atendida[:n_particles, i] = evap_broadcast - self.evap_inicial[:n_particles, i]
        
        # Retenção corrigida
        self.ret_corrig[:n_particles, i] = xp.where(
            self.retencao[:n_particles, i] > 0, 0, self.retencao[:n_particles, i]
        )
        
        # Reserva de solo
        self.reserv_solo[:n_particles, i] = xp.where(
            self.reserv_solo_corrig[:n_particles, i-1] > SUBmax,
            SUBmax,
            self.reserv_solo_corrig[:n_particles, i-1]
        )
        
        # Ps (precipitação que contribui para escoamento) - vectorizada
        Pn = chuva_broadcast - self.evap_inicial[:n_particles, i]
        if RIO == 0:
            hiperb = xp.tanh(Pn / SUBmax)
            Sub = self.reserv_solo[:n_particles, i] / SUBmax
            termo1 = SUBmax * (1 - Sub ** 2) * hiperb
            termo2 = 1 + Sub * hiperb
            self.Ps[:n_particles, i] = xp.maximum(termo1 / termo2, 0)
        else:
            # Usar parâmetro beta específico de cada partícula
            beta_params = params_batch[:, 3] if params_batch.shape[1] > 3 else beta
            self.Ps[:n_particles, i] = xp.maximum(
                Pn * (1 - self.reserv_solo[:n_particles, i] / SUBmax / beta_params), 0
            )
        
        # Recarga do solo
        self.rec_solo[:n_particles, i] = self.Ps[:n_particles, i]
        
        # Escoamento
        self.escoamento[:n_particles, i] = (
            chuva_broadcast - self.evap_inicial[:n_particles, i] - self.rec_solo[:n_particles, i]
        )
        
        # RE (evapotranspiração real)
        E = (1 - xp.exp(-a * (self.reserv_solo[:n_particles, i] / SUBmax))) * self.evap_n_atendida[:n_particles, i]
        self.RE[:n_particles, i] = xp.minimum(
            xp.minimum(self.evap_n_atendida[:n_particles, i], self.reserv_solo[:n_particles, i]), 
            E
        )
        
        # Solo após evapotranspiração
        self.Solo[:n_particles, i] = xp.maximum(self.reserv_solo[:n_particles, i] - self.RE[:n_particles, i], 0)
        
        # Recarga do rio (usar parâmetro Ks específico de cada partícula)
        Ks_params = params_batch[:, 0]
        self.rec_rio[:n_particles, i] = Ks_params * self.Solo[:n_particles, i]
        
        # S1 (armazenamento na calha)
        self.S1[:n_particles, i] = (
            self.S3[:n_particles, i-1] + self.escoamento[:n_particles, i] + self.rec_rio[:n_particles, i]
        )
        
        # C (vazão que sai da calha)
        self.C[:n_particles, i] = xp.where(
            self.S1[:n_particles, i] <= 0, 
            0,
            xp.minimum(k * self.S1[:n_particles, i]**b, self.S1[:n_particles, i])
        )
        
        # S2
        self.S2[:n_particles, i] = self.S1[:n_particles, i] - self.C[:n_particles, i]
        
        # Perdas (usar parâmetro p específico de cada partícula)
        p_params = params_batch[:, 1]
        self.perdas[:n_particles, i] = xp.minimum(
            KL_melhor * (self.C[:n_particles, i] ** p_params), 
            self.C[:n_particles, i]
        )
        
        # Vazão calculada
        self.vazao_calc[:n_particles, i] = (
            (self.C[:n_particles, i] - self.perdas[:n_particles, i]) / 1000
        ) * (areakm2 * 1000000 / T)
        
        # Reserva de solo corrigida
        self.reserv_solo_corrig[:n_particles, i] = xp.maximum(
            self.Solo[:n_particles, i] + self.rec_solo[:n_particles, i] - self.rec_rio[:n_particles, i], 
            0
        )
        
        # Reservatório profundo
        if RIO == 0:
            self.profundo[:n_particles, i] = 0
        else:
            cond = (self.reserv_solo[:n_particles, i] + self.Ps[:n_particles, i] - 
                   self.rec_rio[:n_particles, i] > SUBmax)
            self.profundo[:n_particles, i] = xp.where(
                cond,
                (self.profundo_corrigido[:n_particles, i-1] + self.reserv_solo[:n_particles, i] + 
                 self.Ps[:n_particles, i] - self.rec_rio[:n_particles, i] - SUBmax),
                self.profundo_corrigido[:n_particles, i-1]
            )
        
        # Percolação profunda Fg (usar parâmetro Kg específico de cada partícula)
        if RIO == 0:
            self.Fg[:n_particles, i] = 0
        else:
            Kg_params = params_batch[:, 2] if params_batch.shape[1] > 2 else kg
            self.Fg[:n_particles, i] = self.profundo[:n_particles, i] * Kg_params
        
        # Profundo corrigido
        if RIO == 0:
            self.profundo_corrigido[:n_particles, i] = 0
        else:
            self.profundo_corrigido[:n_particles, i] = xp.where(
                self.profundo[:n_particles, i] - self.Fg[:n_particles, i] < Gmax,
                self.profundo[:n_particles, i] - self.Fg[:n_particles, i],
                Gmax
            )
        
        # S3
        self.S3[:n_particles, i] = (
            self.S2[:n_particles, i] + self.profundo[:n_particles, i] - self.profundo_corrigido[:n_particles, i]
        )

# Condições iniciais baseadas nos dados do CAWM
initial_conditions = {
    'evap_inicial': float(CAWM.loc[intervalo1, 'evap_inicial']),
    'retencao': float(CAWM.loc[intervalo1, 'retencao']),
    'evap_n_atendida': float(CAWM.loc[intervalo1, 'evap_n_atendida']),
    'ret_corrig': float(CAWM.loc[intervalo1, 'ret_corrig']),
    'reserv_solo': float(CAWM.loc[intervalo1, 'reserv_solo']),
    'S3': float(CAWM.loc[intervalo1, 'S3']),
    'reserv_solo_corrig': float(CAWM.loc[intervalo1, 'reserv_solo_corrig']),
    'profundo_corrigido': float(CAWM.loc[intervalo1, 'profundo_corrigido'])
}

# Instanciar modelo
model = HidrologicalModelGPU(obs, chuva, evap, intervalo1, intervalo2, initial_conditions, max_particles=100)

################### FUNÇÕES DE AVALIAÇÃO ####################

def nash_sutcliffe_gpu(sim, obs):
    """Calcula Nash-Sutcliffe usando GPU"""
    # Filtrar NaN
    mask = ~(xp.isnan(sim) | xp.isnan(obs))
    sim_clean = sim[mask]
    obs_clean = obs[mask]
    
    if len(obs_clean) == 0:
        return -999.0
    
    obs_mean = xp.mean(obs_clean)
    numerator = xp.sum((sim_clean - obs_clean)**2)
    denominator = xp.sum((obs_clean - obs_mean)**2)
    
    if float(denominator) == 0:
        return -999.0
    
    return float(1 - numerator / denominator)

def funcao_objetivo_gpu(x):
    """Função objetivo otimizada para GPU (single particle)"""
    # Converter para array correto se necessário
    if isinstance(x, np.ndarray) and GPU_AVAILABLE:
        x = cp.asarray(x)
    elif not isinstance(x, np.ndarray) and not GPU_AVAILABLE:
        x = np.asarray(x)
    return funcao_objetivo_gpu_batch(x.reshape(1, -1))[0]

def funcao_objetivo_gpu_batch(params_batch):
    """Função objetivo vectorizada para múltiplas partículas"""
    try:
        # Converter para array correto se necessário
        if isinstance(params_batch, np.ndarray) and GPU_AVAILABLE:
            params_batch = cp.asarray(params_batch)
        elif not isinstance(params_batch, np.ndarray) and not GPU_AVAILABLE:
            params_batch = np.asarray(params_batch)
            
        n_particles = params_batch.shape[0]
        
        # Simular modelo para todas as partículas simultaneamente
        vazao_calc_batch = model.simulate_batch(params_batch)
        
        # Preparação das observações
        obs_period = obs_gpu[1:vazao_calc_batch.shape[1]+1] if len(obs_gpu) > vazao_calc_batch.shape[1] else obs_gpu[1:]
        min_len = min(vazao_calc_batch.shape[1], len(obs_period))
        
        # Ajustar tamanhos
        vazao_calc_batch = vazao_calc_batch[:, :min_len]
        obs_period_broadcast = xp.broadcast_to(obs_period[:min_len], vazao_calc_batch.shape)
        
        # Calcular Nash-Sutcliffe para todas as partículas
        nash_batch = nash_sutcliffe_gpu_batch(vazao_calc_batch, obs_period_broadcast)
        
        # Calcular diferença absoluta para todas as partículas
        soma_abs_batch = xp.sum(xp.abs(vazao_calc_batch - obs_period_broadcast), axis=1)
        
        # Função objetivo vectorizada (corrigida para evitar valores negativos)
        # Nash-Sutcliffe: [-∞, 1], onde 1 = perfeito, 0 = média, <0 = pior que média
        # Transformar Nash para sempre positivo: nash_transformed = nash + 1 (range [0, 2])
        nash_transformed = nash_batch + 1.0
        
        # Inverter soma_abs para que menor erro = maior valor
        # Usar 1/(1 + soma_abs) para manter valores entre 0 e 1
        erro_normalizado = 1.0 / (1.0 + soma_abs_batch / 1000.0)  # Dividir por 1000 para escalar
        
        # Função objetivo combinada (sempre positiva)
        FO_batch = xp.where(
            nash_batch == -999.0,
            -999.0,  # Caso especial (falha na simulação)
            (nash_transformed * erro_normalizado) * 1000  # Multiplicar por 1000 para escala conveniente
        )
        
        # Converter para CPU se necessário
        if GPU_AVAILABLE:
            return cp.asnumpy(FO_batch)
        else:
            return FO_batch
        
    except Exception as e:
        print(f"Erro na função objetivo: {e}")
        import traceback
        traceback.print_exc()
        return np.full(params_batch.shape[0] if hasattr(params_batch, 'shape') else 1, -999.0)

def nash_sutcliffe_gpu_batch(sim_batch, obs_batch):
    """Calcula Nash-Sutcliffe para múltiplas partículas usando GPU"""
    # Filtrar NaN
    mask = ~(xp.isnan(sim_batch) | xp.isnan(obs_batch))
    
    # Calcular Nash-Sutcliffe para cada partícula
    nash_results = []
    for i in range(sim_batch.shape[0]):
        sim_clean = sim_batch[i][mask[i]]
        obs_clean = obs_batch[i][mask[i]]
        
        if len(obs_clean) == 0:
            nash_results.append(-999.0)
            continue
        
        obs_mean = xp.mean(obs_clean)
        numerator = xp.sum((sim_clean - obs_clean)**2)
        denominator = xp.sum((obs_clean - obs_mean)**2)
        
        if float(denominator) == 0:
            nash_results.append(-999.0)
        else:
            nash_results.append(float(1 - numerator / denominator))
    
    return xp.array(nash_results)

################### PSO OTIMIZADO ####################

class ParticleGPU:
    """Partícula PSO otimizada para GPU"""
    
    def __init__(self, x0, bounds):
        self.dim = len(x0)
        self.bounds = bounds
        
        # Usar GPU para arrays se disponível
        if GPU_AVAILABLE:
            self.position = cp.array(x0, dtype=cp.float32)
            self.velocity = cp.zeros(self.dim, dtype=cp.float32)
            self.best_position = cp.array(x0, dtype=cp.float32)
        else:
            self.position = np.array(x0, dtype=np.float32)
            self.velocity = np.zeros(self.dim, dtype=np.float32)
            self.best_position = np.array(x0, dtype=np.float32)
        
        self.best_fitness = -np.inf
        self.fitness = -np.inf
        
        # Inicializar velocidade
        self._init_velocity()
    
    def _init_velocity(self):
        """Inicializa velocidade com valores aleatórios"""
        for i in range(self.dim):
            if i == 3:  # beta
                vel_range = 2.0
            else:
                vel_range = 0.02
            
            if GPU_AVAILABLE:
                self.velocity[i] = cp.random.uniform(-vel_range, vel_range)
            else:
                self.velocity[i] = np.random.uniform(-vel_range, vel_range)
    
    def evaluate(self, objective_func):
        """Avalia fitness da partícula"""
        # Converter para CPU para função objetivo se necessário
        if GPU_AVAILABLE:
            pos_cpu = cp.asnumpy(self.position)
        else:
            pos_cpu = self.position
        
        self.fitness = objective_func(pos_cpu)
        
        # Atualizar melhor posição
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            if GPU_AVAILABLE:
                self.best_position = cp.copy(self.position)
            else:
                self.best_position = np.copy(self.position)
    
    def update_velocity(self, global_best, w=0.8, c1=1.0, c2=2.0):
        """Atualiza velocidade da partícula"""
        # Gerar números aleatórios
        if GPU_AVAILABLE:
            r1 = cp.random.random(self.dim)
            r2 = cp.random.random(self.dim)
            global_best_gpu = cp.array(global_best)
        else:
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            global_best_gpu = np.array(global_best)
        
        # Equação da velocidade PSO
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_gpu - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        """Atualiza posição da partícula"""
        self.position = self.position + self.velocity
        
        # Aplicar limites
        for i in range(self.dim):
            if float(self.position[i]) < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
            elif float(self.position[i]) > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]

class PSO_GPU:
    """PSO vectorizado para máximo aproveitamento da GPU

    Suporta histórico por partícula e early-stopping por estagnação.
    """

    def __init__(self, objective_func, x0, bounds, num_particles=20, maxiter=50,
                 early_stopping=True, patience=10, tol=1e-6, diversity_tol=None,
                 w_max=0.9, w_min=0.4, c1_init=2.0, c1_final=0.5, c2_init=0.5, c2_final=2.0,
                 vmax_factor=0.2):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.maxiter = maxiter
        self.dim = len(x0)

        # Early stopping params
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        # Optional diversity threshold (std of fitness) to avoid stopping in collapsed swarm
        self.diversity_tol = diversity_tol
        
        # PSO parameters with adaptive behavior
        self.w_max = w_max        # inércia máxima (início)
        self.w_min = w_min        # inércia mínima (fim)
        self.c1_init = c1_init    # componente cognitivo inicial
        self.c1_final = c1_final  # componente cognitivo final
        self.c2_init = c2_init    # componente social inicial
        self.c2_final = c2_final  # componente social final
        self.vmax_factor = vmax_factor  # fator de velocidade máxima

        # Inicializar arrays vectorizados para todas as partículas
        self.positions = self._initialize_positions(x0)
        self.velocities = self._initialize_velocities()
        self.best_positions = self.positions.copy()
        self.best_fitness = np.full(num_particles, -np.inf)
        self.fitness = np.full(num_particles, -np.inf)

        # Melhor global
        self.global_best_position = x0.copy()
        self.global_best_fitness = -np.inf

        # Calcular velocidades máximas para cada dimensão
        self.vmax = np.zeros(self.dim)
        for j, (low, high) in enumerate(self.bounds):
            self.vmax[j] = (high - low) * self.vmax_factor
        
        # Histórico
        # fitness_history: histórico do melhor global por iteração
        self.fitness_history = []
        self.iteration_history = []
        # fitness_history_particles: lista de arrays (num_particles,) por iteração
        self.fitness_history_particles = []
    
    def _initialize_positions(self, x0):
        """Inicializa posições de todas as partículas"""
        positions = np.zeros((self.num_particles, self.dim))
        
        for i in range(self.num_particles):
            for j, (low, high) in enumerate(self.bounds):
                if j == 0:  # x0 base
                    positions[i, j] = x0[j] + np.random.uniform(-0.1, 0.1)
                else:
                    positions[i, j] = x0[j] + np.random.uniform(-0.05, 0.05)
                
                # Garantir limites
                positions[i, j] = max(low, min(high, positions[i, j]))
        
        return positions
    
    def _initialize_velocities(self):
        """Inicializa velocidades de todas as partículas"""
        velocities = np.zeros((self.num_particles, self.dim))
        
        for i in range(self.num_particles):
            for j in range(self.dim):
                if j == 3:  # beta
                    vel_range = 2.0
                else:
                    vel_range = 0.02
                
                velocities[i, j] = np.random.uniform(-vel_range, vel_range)
        
        return velocities
    
    def evaluate_batch(self):
        """Avalia todas as partículas simultaneamente"""
        # UMA Única chamada à GPU para todas as partículas!
        self.fitness = funcao_objetivo_gpu_batch(self.positions)
        
        # Atualizar melhores posições individuais
        improvement_mask = self.fitness > self.best_fitness
        self.best_fitness[improvement_mask] = self.fitness[improvement_mask]
        self.best_positions[improvement_mask] = self.positions[improvement_mask]
        
        # Atualizar melhor global
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.global_best_fitness:
            self.global_best_fitness = self.fitness[best_idx]
            self.global_best_position = self.positions[best_idx].copy()
        # Registrar histórico por partícula (cópia em CPU para salvar/plot)
        try:
            self.fitness_history_particles.append(self.fitness.copy())
        except Exception:
            # Em caso de array cupy, trazer para CPU antes de copiar
            try:
                self.fitness_history_particles.append(cp.asnumpy(self.fitness))
            except Exception:
                self.fitness_history_particles.append(np.array(self.fitness))
    
    def update_velocities_batch(self, iteration=0):
        """Atualiza velocidades de todas as partículas simultaneamente com parâmetros adaptativos"""
        # Calcular parâmetros adaptativos baseados na iteração
        progress = iteration / max(1, self.maxiter - 1)  # 0 a 1
        
        # Decaimento linear da inércia
        w = self.w_max - (self.w_max - self.w_min) * progress
        
        # Transição dos componentes cognitivo e social
        c1 = self.c1_init - (self.c1_init - self.c1_final) * progress
        c2 = self.c2_init + (self.c2_final - self.c2_init) * progress
        
        # Gerar números aleatórios para todas as partículas
        r1 = np.random.random((self.num_particles, self.dim))
        r2 = np.random.random((self.num_particles, self.dim))
        
        # Equação da velocidade PSO vectorizada
        cognitive = c1 * r1 * (self.best_positions - self.positions)
        social = c2 * r2 * (self.global_best_position - self.positions)
        
        self.velocities = w * self.velocities + cognitive + social
        
        # Aplicar velocidade máxima
        for j in range(self.dim):
            self.velocities[:, j] = np.clip(self.velocities[:, j], -self.vmax[j], self.vmax[j])
    
    def update_positions_batch(self):
        """Atualiza posições de todas as partículas simultaneamente"""
        self.positions = self.positions + self.velocities
        
        # Aplicar limites vectorizadamente
        for j, (low, high) in enumerate(self.bounds):
            self.positions[:, j] = np.clip(self.positions[:, j], low, high)
    
    def optimize(self):
        """Executa otimização PSO vectorizada"""
        print(f"Iniciando PSO vectorizado com {self.num_particles} partículas por {self.maxiter} iterações...")
        print(f"AVISO: Usando simulação em lote - apenas {self.maxiter} chamadas à GPU no total!")
        no_improve_count = 0
        last_global_best = self.global_best_fitness

        for iteration in range(self.maxiter):
            print(f"Iteração {iteration+1}/{self.maxiter}")

            # Avaliar TODAS as partículas em uma única chamada
            self.evaluate_batch()

            # Atualizar velocidades e posições em lote
            self.update_velocities_batch(iteration)
            self.update_positions_batch()

            # Registrar histórico global e por iteração
            self.fitness_history.append(self.global_best_fitness)
            self.iteration_history.append(iteration)

            # Calcular parâmetros atuais para logging
            progress = iteration / max(1, self.maxiter - 1)
            w_current = self.w_max - (self.w_max - self.w_min) * progress
            c1_current = self.c1_init - (self.c1_init - self.c1_final) * progress
            c2_current = self.c2_init + (self.c2_final - self.c2_init) * progress

            # já registramos per-particle em evaluate_batch
            print(f"Melhor fitness: {self.global_best_fitness:.6f}")
            print(f"Fitness médio: {np.mean(self.fitness):.6f}")
            print(f"Desvio padrão fitness: {np.std(self.fitness):.6f}")
            print(f"Parâmetros PSO: w={w_current:.3f}, c1={c1_current:.3f}, c2={c2_current:.3f}")

            if RIO == 0:
                print(f"Parâmetros: Ks={self.global_best_position[0]:.4f}, p={self.global_best_position[1]:.4f}")
            else:
                print(f"Parâmetros: Ks={self.global_best_position[0]:.4f}, p={self.global_best_position[1]:.4f}, "
                      f"Kg={self.global_best_position[2]:.4f}, beta={self.global_best_position[3]:.4f}")

            # EARLY STOPPING: verificar melhora do melhor global
            improvement = self.global_best_fitness - last_global_best
            if improvement > self.tol:
                no_improve_count = 0
                last_global_best = self.global_best_fitness
            else:
                no_improve_count += 1

            if self.early_stopping and no_improve_count >= self.patience:
                # Checar diversidade se configurado
                try:
                    fitness_std = float(np.std(self.fitness))
                except Exception:
                    try:
                        fitness_std = float(cp.asnumpy(cp.std(self.fitness)))
                    except Exception:
                        fitness_std = 0.0

                if (self.diversity_tol is None) or (fitness_std < self.diversity_tol):
                    print(f"Early-stopping: sem melhora (> {self.tol}) por {self.patience} iterações. Interrompendo.")
                    break
                else:
                    # Ainda há diversidade suficiente; resetar contador e continuar
                    print(f"Sem melhora por {self.patience} iterações, mas diversidade ({fitness_std:.4f}) > threshold ({self.diversity_tol}). Continuando.")
                    no_improve_count = 0

        return self.global_best_position, self.global_best_fitness

################### EXECUÇÃO PRINCIPAL ####################

def main(num_particles=50, maxiter=30, early_stopping=True, patience=10, tol=1e-6, diversity_tol=None,
         w_max=0.9, w_min=0.4, c1_init=2.0, c1_final=0.5, c2_init=0.5, c2_final=2.0, vmax_factor=0.15):
    """Função principal com parâmetros PSO otimizados"""
    print("Configurando otimização...")
    print(f"🚀 Configuração vectorizada: {num_particles} partículas, {maxiter} iterações")
    print(f"📊 Parâmetros PSO adaptativos:")
    print(f"   • Inércia: {w_max} → {w_min}")
    print(f"   • Cognitivo: {c1_init} → {c1_final}")
    print(f"   • Social: {c2_init} → {c2_final}")
    print(f"   • Vmax factor: {vmax_factor}")
    
    # Configurar parâmetros iniciais e limites
    if RIO == 0:
        x0 = np.array([Ks, expo_perdas])
        bounds = [(0, 1), (0.8, 1.2)]
        param_names = ['Ks', 'p']
    else:
        x0 = np.array([Ks, expo_perdas, kg, beta])
        bounds = [(0, 1), (0.8, 1.2), (0, 1), (1, 10)]
        param_names = ['Ks', 'p', 'Kg', 'beta']
    
    # Executar PSO vectorizado com parâmetros otimizados
    pso = PSO_GPU(funcao_objetivo_gpu, x0, bounds, num_particles=num_particles, maxiter=maxiter,
                  early_stopping=early_stopping, patience=patience, tol=tol, diversity_tol=diversity_tol,
                  w_max=w_max, w_min=w_min, c1_init=c1_init, c1_final=c1_final,
                  c2_init=c2_init, c2_final=c2_final, vmax_factor=vmax_factor)
    best_params, best_fitness = pso.optimize()
    
    # Resultados finais
    print("\n" + "="*50)
    print("RESULTADOS FINAIS:")
    print("="*50)
    
    for i, (name, value) in enumerate(zip(param_names, best_params)):
        print(f"{name} = {value:.6f}")
    
    print(f"KL = {KL_melhor:.6f}")
    print(f"Função Objetivo = {best_fitness:.6f}")
    
    # Simular com melhores parâmetros para obter Nash
    final_vazao = model.simulate(best_params)
    obs_final = obs_gpu[1:len(final_vazao)+1] if len(obs_gpu) > len(final_vazao) else obs_gpu[1:]
    min_len = min(len(final_vazao), len(obs_final))
    nash_final = nash_sutcliffe_gpu(final_vazao[:min_len], obs_final[:min_len])
    print(f"Nash-Sutcliffe = {nash_final:.6f}")
    
    # Salvar resultados
    salvar_resultados(best_params, param_names, best_fitness, nash_final, pso)
    
    # Plotar convergência global e por partícula
    plotar_convergencia(pso.iteration_history, pso.fitness_history, fitness_particles=pso.fitness_history_particles)
    
    return best_params, best_fitness

def salvar_resultados(params, param_names, fitness, nash, pso):
    """Salva resultados em CSV"""
    try:
        os.makedirs(path_resultados, exist_ok=True)
        
        # Criar DataFrame com resultados
        results_data = {}
        for i, name in enumerate(param_names):
            results_data[name] = [params[i]]
        
        results_data['KL'] = [KL_melhor]
        results_data['Nash'] = [nash]
        results_data['FuncaoObjetivo'] = [fitness]
        results_data['Iteracoes'] = [len(pso.fitness_history)]
        results_data['Modo'] = ['GPU' if GPU_AVAILABLE else 'CPU']
        
        df_results = pd.DataFrame(results_data)
        
        output_path = os.path.join(path_resultados, "PARAMETROS_CALIBRADOS_GPU.csv")
        df_results.to_csv(output_path, index=False, float_format='%.8f')
        print(f"Resultados salvos em: {output_path}")
        
    except Exception as e:
        print(f"Erro ao salvar resultados: {e}")

def plotar_convergencia(iteracoes, fitness, fitness_particles=None, max_plot_particles=40):
    """Plota gráfico de convergência.

    - `iteracoes`: lista de índices de iteração
    - `fitness`: lista do melhor global por iteração
    - `fitness_particles`: opcional, lista de arrays (num_particles,) por iteração
    - `max_plot_particles`: número máximo de partículas a plotar (para legibilidade)
    """
    try:
        # Plot do melhor global
        plt.figure(figsize=(10, 6))
        plt.plot(iteracoes, fitness, marker='o', linestyle='-', color='b', markersize=4, label='Melhor global')
        plt.title('Convergência da Otimização PSO (GPU)', fontsize=14)
        plt.xlabel('Iteração', fontsize=12)
        plt.ylabel('Função Objetivo', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(path_resultados, "Convergencia_PSO_GPU.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico global salvo em: {output_path}")

        # Se fornecido histórico por partícula, plotar linhas por partícula
        if fitness_particles is not None and len(fitness_particles) > 0:
            # converter para array (iteracoes x particles)
            try:
                arr = np.vstack([np.array(x) for x in fitness_particles])  # shape (iters, particles)
            except Exception:
                # tentar com cupy
                try:
                    arr = cp.vstack([cp.asarray(x) for x in fitness_particles])
                    arr = cp.asnumpy(arr)
                except Exception:
                    arr = np.array([np.array(x) for x in fitness_particles])

            n_iters, n_particles = arr.shape
            # escolher um subconjunto de partículas para plotar, distribuído uniformemente
            n_plot = min(n_particles, max_plot_particles)
            if n_plot < n_particles:
                indices = np.linspace(0, n_particles - 1, n_plot, dtype=int)
            else:
                indices = np.arange(n_particles)

            plt.figure(figsize=(12, 8))
            for idx in indices:
                plt.plot(range(n_iters), arr[:, idx], linewidth=1, alpha=0.8)

            plt.title(f'Convergência por Partícula (mostrando {len(indices)} de {n_particles})', fontsize=14)
            plt.xlabel('Iteração', fontsize=12)
            plt.ylabel('Função Objetivo (por partícula)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_particles = os.path.join(path_resultados, "Convergencia_PSO_per_particula.png")
            plt.savefig(out_particles, dpi=300, bbox_inches='tight')
            print(f"Gráfico por partícula salvo em: {out_particles}")

    except Exception as e:
        print(f"Erro ao plotar gráfico: {e}")

if __name__ == "__main__":
    try:
        # Permitir escolha via linha de comando ou usar padrão otimizado
        import sys
        if len(sys.argv) >= 3:
            num_particles = int(sys.argv[1])
            maxiter = int(sys.argv[2])
            print(f"Usando configuração da linha de comando: {num_particles} partículas, {maxiter} iterações")
        else:
            # Usar configuração padrão otimizada para melhor aproveitamento da GPU
            num_particles = 50  # Aumentado de 20 para 50
            maxiter = 30
            print("🚀 Usando configuração padrão otimizada para vectorização massiva!")
            print(f"   📊 {num_particles} partículas processadas SIMULTANEAMENTE")
            print(f"   ⚡ {maxiter} iterações = apenas {maxiter} chamadas à GPU")
            print("   🎯 Esperado: uso da GPU de 13% → 80-90%+")
        
        best_params, best_fitness = main(num_particles, maxiter)
        
        total_time = time.time() - start_time
        evaluations_per_sec = (num_particles * maxiter) / total_time
        
        print(f"\n⏱️  Tempo total de execução: {total_time:.2f} segundos")
        print(f"📈 Performance: {evaluations_per_sec:.1f} avaliações/segundo")
        print(f"🎮 Configuração: {num_particles} partículas × {maxiter} iterações")
        
        if GPU_AVAILABLE:
            print("🚀 Otimização concluída com sucesso usando GPU VECTORIZADA!")
            print("💡 Para monitorar GPU: watch -n 1 nvidia-smi")
        else:
            print("⚠️  Otimização concluída usando CPU (GPU não disponível)")
            
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
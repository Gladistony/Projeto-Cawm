from __future__ import division

import pandas as pd
import numpy as np
import math
from statistics import mean
import random
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

start_time = time.time()

frame = pd.DataFrame()

path = 'C:\CAWM\CONCENTRADO'
j = "Sirinhaem_posto_fluSIRGAS2000.shp"
path_dados = path + "\\" + 'dados'
path_shape = path_dados + "\\" + 'shapes'
path_chuva = path_dados + "\\" + 'precipitacao.csv'
path_evp = path_dados + "\\" + 'evaporacao.csv'
path_resultados = path + "\\" + 'resultados'
path_vazao = path_dados + "\\" + 'pao.csv'
path_intervalo = path_dados + "\\" + 'intervalo.csv'


################## PARÂMETROS FIXOS ####################
b = 1.666666667    #Coeficiente do expoente do escoamento na calha. Parâmetro não calibrável
T = 86400 #tempo em segundos

# abrindo a camada vetorial com o calculo S

path_layer = path_shape + "\\" + j
layer = QgsVectorLayer(path_layer, j)
QgsProject.instance().addMapLayer(layer)

for feature in layer.getFeatures():
    n = feature.attributes()[0] # coeficiente de manning
    a = feature.attributes()[1] #multiplicador da taxa de evapotranspiração a extrair do solo
    expo_perdas = feature.attributes()[2] #expoente da função de perdas (p) - parâmetro a calibrar
    Ks = feature.attributes()[3]    # parâmetro a calibrar
    p = feature.attributes()[4]    #Parametro de perdas na calha - KL - CALCULADO

    Reserva_solo_inicial = feature.attributes()[5]
    RIO = feature.attributes()[6] # 0 se rio temporário, 1 se rio perene
    Profundo_inicial = feature.attributes()[7] # profundo corrigido inicial
    Reserva = feature.attributes()[8]  # reserva na calha
    beta = feature.attributes()[9] # parâmetro de infiltração a calibrar em subbacias úmidas
    kg = feature.attributes()[10] # parâmetro a calibrar em subbacias úmidas
    Gmax = feature.attributes()[11] #profundidade máxima do reservatório profundo
    B = feature.attributes()[12] #profundidade máxima do reservatório profundo
    SUBmax = feature.attributes()[13]
    
    print('S', SUBmax)

# com dados métricos
d = QgsDistanceArea()
d.setEllipsoid("WGS84")
for feature in layer.getFeatures():
    geom = feature.geometry()
    area = d.measureArea(geom)
    areakm2 = d.convertAreaMeasurement(area, QgsUnitTypes.AreaSquareKilometers)
        
print('área = ', areakm2)

if areakm2 < 6000:
    k = 0.3745*areakm2**(-0.489)+0.0146
if (areakm2 >= 6000 and areakm2 <= 60000):
    k = 34.343 * areakm2 ** (-0.853)
if areakm2 > 60000:
    k = 0.0028
print('k = ', k) # parametro para o cálculo do escoamento Fr que sai da calha

#Ler o arquivo em que se encontra o período
periodo = pd.read_csv(path_intervalo)

#Ler o arquivo em que se encontra o balanço hídrico sem calibração
path_balanco = path_resultados + "\\" + "CAWM.csv"
CAWM = pd.read_csv(path_balanco)

intervalo1 = int(periodo.loc[0, 'inicio'])
intervalo2 = int(periodo.loc[0, 'fim'])
print('dia inicial: ',intervalo1)
print('dia final: ',intervalo2)

obs = CAWM.loc[intervalo1:intervalo2, 'vazao']
chuva = CAWM.loc[intervalo1:intervalo2, 'chuva_media']
evap = CAWM.loc[intervalo1:intervalo2, 'evaporacao']

soma_chuva = sum(chuva)
soma_vazao =1000*(sum(obs)*24*60*60)/(areakm2*1000000)

soma_evap = sum(CAWM.loc[intervalo1:intervalo2, 'evap_inicial']) + sum(CAWM.loc[intervalo1:intervalo2, 'RE']) 

# Calcula o valor do Parametro de perdas na calha - KL
#KL_melhor = (soma_chuva - soma_vazao - soma_evap) / (abs(soma_chuva - soma_evap))
KL_melhor = p
print('KL = ', KL_melhor)



frame = pd.concat([obs, chuva, evap], axis=1)

frame['evap_inicial'] = ""
frame.loc[intervalo1,'evap_inicial'] = CAWM.loc[intervalo1,'evap_inicial']
frame['retencao'] = ""
frame.loc[intervalo1,'retencao'] = CAWM.loc[intervalo1,'retencao']
frame['evap_n_atendida'] = ""
frame.loc[intervalo1,'evap_n_atendida'] = CAWM.loc[intervalo1,'evap_n_atendida']
frame['ret_corrig'] = ""
frame.loc[intervalo1,'ret_corrig'] = CAWM.loc[intervalo1,'ret_corrig']
frame['escoamento'] = ""
frame.loc[intervalo1,'escoamento'] = CAWM.loc[intervalo1,'escoamento']
frame['reserv_solo'] = ""
frame.loc[intervalo1,'reserv_solo'] = CAWM.loc[intervalo1,'reserv_solo']
frame['S1'] = ""
frame.loc[intervalo1,'S1'] = CAWM.loc[intervalo1,'S1']
frame['RE'] = ""
frame.loc[intervalo1,'RE'] = CAWM.loc[intervalo1,'RE']
frame['Solo'] = ""
frame.loc[intervalo1,'Solo'] = CAWM.loc[intervalo1,'Solo']
frame['C'] = ""
frame.loc[intervalo1,'C'] = CAWM.loc[intervalo1,'C']
frame['S2'] = ""
frame.loc[intervalo1,'S2'] = CAWM.loc[intervalo1,'S2']
frame['vazao_calc'] = ""
frame.loc[intervalo1,'vazao_calc'] = CAWM.loc[intervalo1,'vazao_calc']
frame['Ps'] = ""
frame.loc[intervalo1,'Ps'] = CAWM.loc[intervalo1,'Ps']
frame['rec_solo'] = ""
frame.loc[intervalo1,'rec_solo'] = CAWM.loc[intervalo1,'rec_solo']
frame['rec_rio'] = ""
frame.loc[intervalo1,'rec_rio'] = CAWM.loc[intervalo1,'rec_rio']
frame['perdas'] = ""
frame.loc[intervalo1,'perdas'] = CAWM.loc[intervalo1,'perdas']
frame['reserv_solo_corrig'] = ""
frame.loc[intervalo1,'reserv_solo_corrig'] = CAWM.loc[intervalo1,'reserv_solo_corrig']
frame['profundo'] = ""
frame.loc[intervalo1,'profundo'] = CAWM.loc[intervalo1,'profundo']
frame['Fg'] = ""
frame.loc[intervalo1,'Fg'] = CAWM.loc[intervalo1,'Fg']
frame['profundo_corrigido'] = ""
frame.loc[intervalo1,'profundo_corrigido'] = CAWM.loc[intervalo1,'profundo_corrigido']
frame['S3'] = ""
frame.loc[intervalo1,'S3'] = CAWM.loc[intervalo1,'S3']


################### FUNÇÃO OBJETIVO ####################

def funcao(x):

    global KL_aux
    global nash_aux

    KL_aux = []
    nash_aux = []

    if RIO == 0:
        print('ks = ', x[0], 'p = ', x[1])  # valores das variáveis
    else:
        print('ks = ', x[0], 'Kg = ', x[2], 'beta = ', x[3], 'p = ', x[1])  # valores das variáveis




    CAWM = pd.read_csv(path_balanco)
    obs = CAWM.loc[intervalo1:intervalo2, 'vazao']
    
    def evap_inicial(frame):
        if (frame.loc[i-1,'ret_corrig']+frame.loc[i,'chuva_media']>= frame.loc[i,'evaporacao']):
            return frame.loc[i, 'evaporacao']
        else:
            return frame.loc[i-1,'ret_corrig']+frame.loc[i,'chuva_media']

    def ret_corrig():
        if(frame.loc[i,'retencao']> 0):
            return 0
        else:
            return frame.loc[i,'retencao']

    def reserv_solo():

        if frame.loc[i-1,'reserv_solo_corrig'] > SUBmax:
            return SUBmax
        else:
            return frame.loc[i-1,'reserv_solo_corrig']


    def Ps(x):
        Pn = frame.loc[i, 'chuva_media'] - frame.loc[i, 'evap_inicial']
        if RIO == 0:
            hiperb = np.tanh(Pn / SUBmax)
            Sub = frame.loc[i, 'reserv_solo'] / SUBmax
            termo1 = SUBmax * (1 - Sub ** 2) * hiperb
            termo2 = 1 + Sub * hiperb
            Ps = termo1 / termo2
            return max(Ps, 0)
        else:

            Ps = Pn * (1 - frame.loc[i, 'reserv_solo'] / SUBmax / x[3])

            return max(Ps, 0)


    def RE():
        E = (1 - math.exp(-a * (frame.loc[i, 'reserv_solo'] / SUBmax))) * frame.loc[i, 'evap_n_atendida']
        return min(frame.loc[i, 'evap_n_atendida'], frame.loc[i, 'reserv_solo'], E)

    def solo(frame):
        return max(frame.loc[i, 'reserv_solo']-frame.loc[i, 'RE'], 0)

    def rec_rio(x):
        return x[0] * frame.loc[i, 'Solo']

    def C():
        if (frame.loc[i,'S1']<=0):
            return 0
        else:
            return min(k*frame.loc[i,'S1']**b,frame.loc[i,'S1'])

    def perdas():

        return min(KL_melhor * (frame.loc[i, 'C'] ** x[1]), frame.loc[i, 'C'])


    def reserv_solo_corrig():
        return max(frame.loc[i,'Solo']+frame.loc[i,'rec_solo']-frame.loc[i,'rec_rio'],0)

# função que calcula o reservatório subterraneo
    def profundo():
        if RIO == 0:
            return 0
        else:
            if frame.loc[i, 'reserv_solo'] + frame.loc[i, 'Ps'] - frame.loc[i, 'rec_rio'] > SUBmax:
                return (frame.loc[i - 1, 'profundo_corrigido'] + frame.loc[i, 'reserv_solo'] + frame.loc[i, 'Ps'] -
                        frame.loc[i, 'rec_rio'] - SUBmax)
            else:
                return (frame.loc[i - 1, 'profundo_corrigido'])

# função que calcula a percolação profunda Fg
    def percolacao_profunda(x):

        if RIO == 0:
            return 0
        else:
            Fg = frame.loc[i, 'profundo'] * x[2]
            return (Fg)

# função que corrige a profundidade
    def profundo_corrigido():
        if RIO == 0:
            return 0
        else:
            if frame.loc[i, 'profundo'] - frame.loc[i, 'Fg'] < Gmax:
                return frame.loc[i, 'profundo'] - frame.loc[i, 'Fg']
            else:
                return (Gmax)

    def filter_nan(sim, obs):
        if sim.isnull().sum() >= 1 or obs.isnull().sum() >= 1:
            obs.isnull()
            obs.fillna(0, inplace=True)
            sim.isnull()
            sim.fillna(0, inplace=True)
            return sim, obs
        else:
            pass

    def NS(sim, obs):
        filter_nan(sim, obs)
        print(sum((sim - obs)**2)/sum((obs - o_med)**2))
        
        return 1 - sum((sim - obs)**2)/sum((obs - o_med)**2)
        
    

    for i in range(intervalo1+1, intervalo2+1):
        frame.loc[i, 'evap_inicial'] = evap_inicial(frame)
        frame.loc[i, 'retencao'] = max(frame.loc[i - 1, 'ret_corrig'] + frame.loc[i, 'chuva_media'] - frame.loc[i, 'evap_inicial'], 0)
        frame.loc[i, 'evap_n_atendida'] = frame.loc[i, 'evaporacao'] - frame.loc[i, 'evap_inicial']
        frame.loc[i, 'ret_corrig'] = ret_corrig()
        frame.loc[i, 'reserv_solo'] = reserv_solo()
        frame.loc[i, 'Ps'] = Ps(x)
        frame.loc[i, 'rec_solo'] = frame.loc[i, 'Ps']
        frame.loc[i, 'escoamento'] = frame.loc[i, 'chuva_media'] - frame.loc[i, 'evap_inicial'] - frame.loc[i, 'rec_solo']
        frame.loc[i, 'RE'] = RE()
        frame.loc[i, 'Solo'] = solo(frame)
        frame.loc[i, 'rec_rio'] = rec_rio(x)
        frame.loc[i, 'S1'] = frame.loc[i - 1, 'S3'] + frame.loc[i, 'escoamento'] + frame.loc[i, 'rec_rio']
        frame.loc[i, 'C'] = C()
        frame.loc[i, 'S2'] = frame.loc[i, 'S1'] - frame.loc[i, 'C']
        frame.loc[i, 'perdas'] = perdas()
        frame.loc[i, 'vazao_calc'] = ((frame.loc[i, 'C'] - frame.loc[i, 'perdas']) / 1000) * (areakm2 * 1000000 / T)
        frame.loc[i, 'reserv_solo_corrig'] = reserv_solo_corrig()
        frame.loc[i, 'profundo'] = profundo()
        frame.loc[i, 'Fg'] = percolacao_profunda(x)
        frame.loc[i, 'profundo_corrigido'] = profundo_corrigido()
        frame.loc[i, 'S3'] = frame.loc[i, 'S2'] + frame.loc[i, 'profundo'] - frame.loc[i, 'profundo_corrigido']



    chuva_frame = frame.loc[1:, 'chuva_media']
    soma_chuva_frame = sum(chuva_frame)

    vazao_frame = frame.loc[1:, 'vazao']
    soma_vazao_frame = 1000*(sum(vazao_frame)*24*60*60)/(areakm2*1000000)

    soma_evap_frame = sum(frame.loc[1: , 'evap_inicial']) + sum(frame.loc[1: , 'RE'])

    KL = (soma_chuva_frame - soma_vazao_frame - soma_evap_frame)/(soma_vazao_frame + soma_chuva_frame - soma_vazao_frame - soma_evap_frame)
    KL_aux.append(KL)

    print('KL auxiliar = ', KL_aux)


    s = frame.loc[intervalo1+1:intervalo2+1,'vazao_calc']
    o = obs.loc[intervalo1+1:intervalo2+1]
    vazao_obs = CAWM.loc[1:,'vazao']
    vazao_obs_corrig = o.dropna()
    o_med = mean(vazao_obs_corrig)
  
    Nash = NS(s, o)

    nash_aux.append(Nash)
    print("Nash: ", Nash)

    simulado = frame.loc[intervalo1 + 1:intervalo2 + 1, 'vazao_calc']
    observado = obs.loc[intervalo1 + 1:intervalo2 + 1]
    s_med = mean(simulado)
    o_media = mean(observado.dropna())
    observado1 = observado.replace(np.nan, 0, regex=True)
    #Rsqr = Rsquared()

    sim = frame.loc[intervalo1 + 1:intervalo2 + 1, 'vazao_calc']
    obs = obs.loc[intervalo1 + 1:intervalo2 + 1]
    obs = obs.replace(np.nan, 0, regex=True)
    
    filter_nan(s, o)

    soma_abs = sum(s - o)
    soma_abs = abs(soma_abs)
    print('soma_abs', soma_abs)


    FO = (Nash/soma_abs)*1000000
    print("FO: ", FO)

    return FO

#### MAIN ####
# a classe Particle armazera informações sobre a partícula
class Particle:
    def __init__(self, x0):  # definindo as partículas (x0 é a posição inicial das partículas)
        self.position_i = []  # posição atual da partícula
        self.velocity_i = []  # velocidade atual da partícula
        self.pos_best_i = []  # melhor posição da partícula individual (mínimo ou máximo local)
        self.err_best_i = -1  # variação da função (melhor resultado da função)
        self.err_i = -1  # resultado individual da função

        # inicializar as velocidades e posições das partículas
        for i in range(0, num_dimensions):
            if i == 3:
                self.velocity_i.append(
                    random.uniform(-1, 1))
            else:
                
                self.velocity_i.append(
                    random.uniform(-0.01, 0.01))  # inicializa a velocidade com números aleatórios que vai de -1 a 1
            self.position_i.append(x0[i])

    # calcula a função e atualiza os mínimos ou máximos locais
    def evaluate(self, costFuncion):
        self.err_i = costFuncion(self.position_i)  # recebe o resultado da função com a partícula i


        # verifica se a posição atual é a melhor individual
        if self.err_i > self.err_best_i or self.err_best_i == -1:
            self.err_best_i = self.err_i  # atualizando o melhor resultado
            self.pos_best_i = self.position_i  # atualizando a melho partícula/posição

    # atualizar a velocidade das partículas EQUAÇÃO DA VELOCIDADE
    def update_velocity(self, pos_best_g):
        w = 0.8  # fator de inércia (quanto a velocidade anterior pesa)
        c1 = 1  # constante cognitiva
        c2 = 2  # constante social

        for i in range(0, num_dimensions):
            if i == 3:
                r1 = random.uniform(0, 1)  # variável aleatória
                r2 = random.uniform(0, 1)
            else:
                
                r1 = random.uniform(0, 0.01)  # variável aleatória
                r2 = random.uniform(0, 0.01)

            # equação da velocidade
            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social
            print('velocidade', self.velocity_i[i])

    # ajustar as bordas
    # atualizar a posição das partículas com base em novas atualizações de velocidade
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]  # EQUAÇÃO DA POSIÇÃO
            print('posição', self.position_i[i])

            # ajustando a máxima posição se necessária
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # ajustando a mínima posição se necessário
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]
            
            

#KL_melhor = p
nash_melhor = 0

class PSO():
    def __init__(self, costFuncion, x0, bounds, num_particles, maxiter):
        global num_dimensions
        global KL_melhor
        global nash_melhor

        num_dimensions = len(x0)  # número de partículas
        err_best_g = -1  # resultado da função com o melhor mínimo global
        pos_best_g = []  # melhor mínimo global
        melhor = []
        iteracao = []
        melhor_KL = []


        # estabelecer o enxame
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))  # adiciono as partículas

        # começando as iterações
        i = 0




        while i < maxiter:

            print("iteração: ", i)
            iteracao.append(i)

            # percorra as partículas do enxame e avalie
            for j in range(0, num_particles):
                swarm[j].evaluate(costFuncion)
                print('pos_best_g', pos_best_g)


                # determine se a partícula atual é a melhor globalmente
                if swarm[j].err_i > err_best_g or err_best_g == -1:
                    pos_best_g2 = swarm[j].position_i
                    vetor2=[]
                    for k in range(num_dimensions):
                        veto = float(pos_best_g2[k])
                        vetor2.append(veto)
                    pos_best_g = vetor2
                        
                    
                    print('pos_best_g', pos_best_g)
                    err_best_g = float(swarm[j].err_i)


                    KL_melhor = KL_aux[0]
                    nash_melhor = nash_aux[0]
                    melhor_KL.append(KL_melhor)




            # percorra as partículas do enxame e atualize as velocidades e posições
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)

            melhor.append(err_best_g)
            print("melhor", melhor)
            i += 1



        # imprmir o resultado final
        print('FINAL:')
        
        
        num = len(melhor_KL)
        if num == 1:
            KL_melhor = melhor_KL[0]
        else: 
            KL_melhor = melhor_KL[num-2]


        if RIO == 0:
            print('Ks = ', pos_best_g[0],'p = ', pos_best_g[1], 'KL = ', KL_melhor)  # valores das variáveis


            frame['Ks'] = ""
            frame.loc[1, 'Ks'] = float(pos_best_g[0])
            frame['p'] = ""
            frame.loc[1, 'p'] = float(pos_best_g[1])
            frame['KL'] = ""
            frame.loc[1, 'KL'] = float(KL_melhor)
            frame['Nash'] = ""
            frame.loc[1, 'Nash'] = float(nash_melhor)
            
            path_out = path_resultados + "\\" + "PARAMETROS_CALIBRADOS.csv"
            frame.to_csv(path_out, float_format='%.20f')

        else:
            print('Ks = ', pos_best_g[0], 'Kg = ', pos_best_g[2], 'p = ', pos_best_g[1], 'KL = ', KL_melhor, 'beta = ', pos_best_g[3])  #valores das variáveis



            frame['Ks'] = ""
            frame.loc[1, 'Ks'] = float(pos_best_g[0])
            frame['Kg'] = ""
            frame.loc[1, 'Kg'] = float(pos_best_g[2])
            frame['p'] = ""
            frame.loc[1, 'p'] = float(pos_best_g[1])
            frame['KL'] = ""
            frame.loc[1, 'KL'] = float(KL_melhor)
            frame['Beta'] = ""
            frame.loc[1, 'Beta'] = float(pos_best_g[3])
            frame['Nash'] = ""
            frame.loc[1, 'Nash'] = float(nash_melhor)
            
            path_out = path_resultados + "\\" + "PARAMETROS_CALIBRADOS.csv"
            frame.to_csv(path_out, float_format='%.20f')

        print('FO = ', err_best_g)  # o melhor resultado da função
        print('Nash = ', nash_melhor)

        # gráfico FO x iteração



        plt.plot(iteracao, melhor, marker='o', linestyle='-', color='b')
        plt.title('Sirinhaem')
        plt.xlabel('Iteração')
        plt.ylabel('Função Objetivo')
        plt.grid(True)
        print("iteração", iteracao)
        print("melhor", melhor)
        
        path_save = path_resultados + "\\" + "Grafico_convergencia.png"



        plt.savefig(path_save, bbox_inches='tight')
        plt.show()



#estimativas iniciais das variáveis de decisão


if RIO == 0:
    x = np.zeros(2)
    x[0] = Ks
    x[1] = expo_perdas

    initial = [x[0], x[1]]

    bounds = [(0, 1), (0.8, 1.2)]

else:
    x = np.zeros(4)
    x[0] = Ks
    x[1] = expo_perdas
    x[2] = kg
    x[3] = beta
    

    initial = [x[0], x[1], x[2], x[3]]

    bounds = [(0, 1), (0.8, 1.2), (0, 1), (1, 10)]


PSO(funcao, initial, bounds, num_particles=5, maxiter=10)

print("Tempo para rodar todo o programa: %.5f seconds" % (time.time() - start_time))





import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

print("Bem-vindo ao CAWM concentrado")

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
T = 86400   #tempo em segundos

################## FUNÇÕES DO CAWM #####################

# função preenche os vazios da série de vazão simulada e observada
def filter_nan(obs):
    if  obs.isnull().sum() >= 1:
        obs.isnull()
        obs.fillna(0, inplace=True)

        return obs
    else:
        pass



#função que calcula a evaporação inicial ou direta
#Se P>=Evaporação potencial, retorne a evaporação = evaporação direta
#Senão retorne chuva média = evaporação direta

def evap_inicial(frame):
    if (float(frame.loc[i-1,'ret_corrig'])+float(frame.loc[i,'chuva_media'])>= float(frame.loc[i,'evaporacao'])):
        return float(frame.loc[i, 'evaporacao'])
    else:
        return float(frame.loc[i-1,'ret_corrig'])+float(frame.loc[i,'chuva_media'])

# garante que a retenção não seja um valor negativo
def ret_corrig():
    if(frame.loc[i,'retencao']> 0):
        return 0
    else:
        return frame.loc[i,'retencao']

#corresponde a célula L do excel
def reserv_solo():
    if i == 1:
        return Reserva_solo_inicial  #reserva do solo inicial
    else:
        if frame.loc[i-1,'reserv_solo_corrig'] > SUBmax:
            return SUBmax
        else:
            return frame.loc[i-1,'reserv_solo_corrig']

# corresponde a célula U do Excel
# ou seja, é igual ao St atualizado (descontado com a evaporação) + recarga do solo (Ps) - escoamento subsuperficial (Fs)
def reserv_solo_corrig():
    return max(frame.loc[i,'Solo']+frame.loc[i,'rec_solo']-frame.loc[i,'rec_rio'],0)

#função que calcula a evapotranspiração complementar (Es)
def RE():
    E = (1 - math.exp(-a*(frame.loc[i,'reserv_solo']/SUBmax)))*frame.loc[i,'evap_n_atendida']
    return min(frame.loc[i,'evap_n_atendida'],frame.loc[i,'reserv_solo'],E)

#função que calcula o escoamento na calha Fr
def C():
    if (frame.loc[i,'S1']<=0):  #verifica se tem água no reservatório
        return 0
    else:
        #calculo do Fr
        return min(k*frame.loc[i,'S1']**b,frame.loc[i,'S1'])

#função que calcula a recarga do solo Ps
def Ps():
    #precipitação efetiva
    Pn = float(frame.loc[i,'chuva_media'])-float(frame.loc[i,'evap_inicial'])

    if RIO == 0:
        hiperb = np.tanh(Pn / SUBmax)
        Sub = frame.loc[i, 'reserv_solo'] / SUBmax
        termo1 = SUBmax * (1 - Sub ** 2) * hiperb
        termo2 = 1 + Sub * hiperb
        Ps = termo1 / termo2
        return max(Ps, 0)
    else:

        Ps = Pn * (1 - frame.loc[i, 'reserv_solo']/SUBmax/beta)

        return max(Ps, 0)




#função que calcula o fluxo subsuperficial fs
def rec_rio():
    return Ks * frame.loc[i, 'Solo']


def zerolistmaker(n):
    listofzeros = [float(0)] * n
    return listofzeros

#corresponde a célula O do excel
#reserva do solo atualizada com o desconto da evaporação complementar Es
def solo(frame):
    return max(frame.loc[i, 'reserv_solo']-frame.loc[i, 'RE'], 0)


def perdas():
    return min(p * (frame.loc[i, 'C'] ** expo_perdas), frame.loc[i, 'C'])
    #return min(p * (frame.loc[i, 'S2'] ** 1.5), frame.loc[i, 'S2'])


#função que calcula o reservatório subterraneo
def profundo ():
    if frame.loc[i, 'reserv_solo'] + frame.loc[i, 'Ps'] - frame.loc[i, 'rec_rio'] > SUBmax:
        return (frame.loc[i - 1, 'profundo_corrigido'] + frame.loc[i, 'reserv_solo'] + frame.loc[i, 'Ps'] -
                    frame.loc[i, 'rec_rio'] - SUBmax)
    else:
        return (frame.loc[i - 1, 'profundo_corrigido'])


#função que calcula a percolação profunda Fg
def percolacao_profunda ():
    Fg = frame.loc[i, 'profundo']*kg
    return (Fg)

#função que corrige a profundidade
def profundo_corrigido ():
    if frame.loc[i, 'profundo'] - frame.loc[i, 'Fg'] < Gmax:
        return frame.loc[i, 'profundo'] - frame.loc[i, 'Fg']
    else:
        return (Gmax)

def NS(sim, obs):
    filter_nan(sim)
    filter_nan(obs)
        
    return 1 - sum((sim - obs)**2)/sum((obs - o_med)**2)

def NSsqrtQ (sim, obs):
    filter_nan(sim)
    filter_nan(obs)
    
    return 1 - sum((sim**(0.5) - obs**0.5)**2)/sum((obs**(0.5) - o_med**(0.5))**2)
 
def NSlog(sim, obs):
    
    sim = np.nan_to_num(sim)
    obs = np.nan_to_num(obs)
    

    
    if (obs <=0).any().any() or (sim <=0).any().any():
        sim_log = sim
        obs_log = obs
    else:
            

        sim_log = np.log10(sim)
        obs_log = np.log10(obs)
        print(sim_log, obs_log)


        
    return 1 - sum((sim_log - obs_log)**2)/sum((obs_log - np.log10(o_med))**2) 




def RMSE(sim, obs):
    filter_nan(sim)
    filter_nan(obs)
    n = len(sim)
    return  (1/n)*(sum((sim - obs)**2))**0.5
   
def Pbias(sim, obs):
    filter_nan(sim)
    filter_nan(obs)
    
    return (sum(sim - obs)*100)/(sum(obs))
    
def RSR(sim, obs):
    filter_nan(sim)
    filter_nan(obs)
    
    return ((sum((sim - obs)**2))**0.5)/((sum((obs - o_med)**2))**0.5)

############ cálculo do balanço hídrico ###############

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


# com dados métricos
d = QgsDistanceArea()
d.setEllipsoid("WGS84")
for feature in layer.getFeatures():
    geom = feature.geometry()
    area = d.measureArea(geom)
    areakm2 = d.convertAreaMeasurement(area, QgsUnitTypes.AreaSquareKilometers)
        
print('área = ', areakm2)

#criar novos atributos
layer.startEditing()
layer.addAttribute(QgsField('AREA_KM', QVariant.Double))
layer.commitChanges()
    
layer.startEditing()
for feature in layer.getFeatures():
    id = feature.id()
    attr_value = {14:areakm2}
    layer.changeAttributeValues(id,attr_value)

layer.commitChanges()
    
if areakm2 < 6000:
    k = 0.3745*areakm2**(-0.489)+0.0146
if (areakm2 >= 6000 and areakm2 <= 60000):
    k = 34.343 * areakm2 ** (-0.853)
if areakm2 > 60000:
    k = 0.0028
        
print('k = ', k) # parametro para o cálculo do escoamento Fr que sai da calha

frame = pd.DataFrame()



#cria um DataFrame vazio

#lendo o arquivo de vazao
vazao = pd.read_csv(path_vazao,float_precision='high')


#lendo o arquivo de chuva
chuva = pd.read_csv(path_chuva, usecols=['chuva_media'],float_precision='high')


#concatena as colunas de vazao e chuva em um mesmo dataframe
frame = pd.concat([vazao, chuva],axis=1)
frame['mes'] = pd.DatetimeIndex(frame['data']).month
print("Arquivos de chuva e vazão lidos com sucesso.")

#preenhendo os vazios das séries observadas
filter_nan(frame['vazao'])
filter_nan(frame['chuva_media'])


#cria um dicionario com o arquivo de evaporação
with open(path_evp, mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {int(rows[0]):float(rows[1]) for rows in reader}
print("Arquivo de evaporação lido com sucesso.")


#adiciona uma nova coluna no dataframe com os valores de evaporação
frame['evaporacao'] = frame['mes'].map(mydict)

n = len(frame.columns) #numero de colunas do dataframe

frame.loc[-1] = zerolistmaker(n)
frame.index = frame.index+1
frame = frame.sort_index()


frame['evap_inicial'] = ""
frame.loc[0,'evap_inicial'] = float(0)
frame['retencao'] = ""
frame.loc[0,'retencao'] = float(0)
frame['evap_n_atendida'] = ""
frame.loc[0,'evap_n_atendida'] = float(0)
frame['ret_corrig'] = ""
frame.loc[0,'ret_corrig'] = float(0)
frame['escoamento'] = ""
frame.loc[0,'escoamento'] = float(0)
frame['reserv_solo'] = ""
frame.loc[0,'reserv_solo'] = float(0)
frame['S1'] = ""
frame.loc[0,'S1'] = float(0)
frame['RE'] = ""
frame.loc[0,'RE'] = float(0)
frame['Solo'] = ""
frame.loc[0,'Solo'] = float(0)
frame['C'] = ""
frame.loc[0,'C'] = float(0)
frame['S2'] = ""
frame.loc[0,'S2'] = float(0)
frame['vazao_calc'] = ""
frame.loc[0,'vazao_calc'] = float(0)
frame['Ps'] = ""
frame.loc[0,'Ps'] = float(0)
frame['rec_solo'] = ""
frame.loc[0,'rec_solo'] = float(0)
frame['rec_rio'] = ""
frame.loc[0,'rec_rio'] = float(0)
frame['perdas'] = ""
frame.loc[0,'perdas'] = float(0)
frame['reserv_solo_corrig'] = ""
frame.loc[0,'reserv_solo_corrig'] = float(0)
frame['profundo'] = ""
frame.loc[0,'profundo'] = float(0)
frame['Fg'] = ""
frame.loc[0,'Fg'] = float(0)
frame['profundo_corrigido'] = ""
frame.loc[0,'profundo_corrigido'] = float(Profundo_inicial)
frame['S3'] = ""
frame.loc[0,'S3'] = float(Reserva)
frame['volume_precipitado'] = ""
frame['lamina_Q_observada'] = ""
frame['evaporacao_potencial'] = ""
frame['evaporacao_real_maxima'] = ""
frame['evaporacao_real'] = ""
frame['solo_inicio'] = ""
frame['solo_fim'] = ""
frame['armazenamento_sup_inicio'] = ""
frame['armazenamento_sup_fim'] = ""
frame['perdas'] = ""
frame['escoado'] = ""
frame['balanco'] = ""


for i in range(1,len(frame)):
    print(i)
    frame.loc[i,'evap_inicial'] = evap_inicial(frame)
    frame.loc[i,'retencao'] = max(float(frame.loc[i-1,'ret_corrig'])+float(frame.loc[i,'chuva_media'])-float(frame.loc[i,'evap_inicial']),0)
    frame.loc[i,'evap_n_atendida'] = frame.loc[i,'evaporacao']-frame.loc[i,'evap_inicial']
    frame.loc[i,'ret_corrig'] = ret_corrig()
    frame.loc[i, 'reserv_solo'] = reserv_solo()
    frame.loc[i, 'Ps'] = Ps()
    frame.loc[i, 'rec_solo'] = frame.loc[i, 'Ps']
    frame.loc[i,'escoamento'] = float(frame.loc[i,'chuva_media'])-float(frame.loc[i,'evap_inicial'])-float(frame.loc[i,'rec_solo'])
    frame.loc[i, 'RE'] = RE()
    frame.loc[i, 'Solo'] = solo(frame)
    frame.loc[i, 'rec_rio'] = rec_rio()
    frame.loc[i, 'S1'] = frame.loc[i - 1, 'S3'] + frame.loc[i, 'escoamento'] + frame.loc[i, 'rec_rio']
    frame.loc[i,'C'] = C()
    frame.loc[i,'S2'] = frame.loc[i,'S1']-frame.loc[i,'C']
    frame.loc[i,'perdas'] = perdas()
    frame.loc[i, 'vazao_calc'] = ((frame.loc[i, 'C'] - frame.loc[i, 'perdas']) / 1000) * (areakm2 * 1000000 / T)
    frame.loc[i,'reserv_solo_corrig'] = reserv_solo_corrig()
    frame.loc[i, 'profundo'] = profundo()
    frame.loc[i, 'Fg'] = percolacao_profunda()
    frame.loc[i, 'profundo_corrigido'] = profundo_corrigido()
    frame.loc[i, 'S3'] = frame.loc[i,'S2'] + frame.loc[i, 'profundo'] - frame.loc[i, 'profundo_corrigido']


# Balanço Hidrico
soma_chuva = (sum((frame.loc[1: i,'chuva_media'])))
frame.loc[1,'volume_precipitado'] = float(soma_chuva)
print("soma chuva", soma_chuva)

vazao_observada = (sum((frame.loc[1: i,'vazao'])))
lamina = 1000*((vazao_observada)*24*60*60)/(areakm2*1000000)
frame.loc[1,'lamina_Q_observada'] = float(lamina)
print('lamina', lamina)

evaporacao_potencial = sum(frame.loc[1: i,'evaporacao'])
frame.loc[1,'evaporacao_potencial'] = float(evaporacao_potencial)

evaporacao_real_max = soma_chuva - lamina
frame.loc[1,'evaporacao_real_maxima'] = float(evaporacao_real_max)

evaporacao_inicial = float(sum(frame.loc[1: i,'evap_inicial']))
RE_ = float(sum(frame.loc[1: i,'RE']))
evaporacao_real = evaporacao_inicial + RE_
frame.loc[1,'evaporacao_real'] = float(evaporacao_real)

solo_inicio = Reserva_solo_inicial + float(frame.loc[1, 'profundo_corrigido'])
frame.loc[1,'solo_inicio'] = float(solo_inicio)

solo_fim = float(frame.loc[i, 'reserv_solo_corrig']) + float(frame.loc[i, 'profundo_corrigido'])
frame.loc[1,'solo_fim'] = float(solo_fim)

armazenamento_sup_inicio = float(frame.loc[1, 'S1'])
frame.loc[1,'armazenamento_sup_inicio'] = float(armazenamento_sup_inicio)

armazenamento_sup_fim = float(frame.loc[i, 'S1'])
frame.loc[1,'armazenamento_sup_fim'] = float(armazenamento_sup_fim)

perdas_ = float(sum(frame.loc[1: i, 'perdas']))
C_ = float(sum(frame.loc[1: i, 'C']))
escoado = C_ - perdas_
frame.loc[1,'escoado'] = float(escoado)

balanco = soma_chuva + solo_inicio + armazenamento_sup_inicio - evaporacao_real - escoado - solo_fim - armazenamento_sup_fim - perdas_
frame.loc[1,'balanco'] = float(balanco)

output_file = path_resultados + '\\' + 'CAWM.csv'

frame.to_csv(output_file, float_format='%.3f')

est = pd.read_csv(path_vazao)
maior_vazao = max(est['vazao'])


print('Arquivo do balanço hídrico criado.')
print('Preparando gráficos do período total')

chuva2 = -1*frame['chuva_media']

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(frame.index,frame['vazao'],label='Qobs m³/s',color='red')
lns2 = ax.plot(frame.index,frame['vazao_calc'],label='Qcalc m³/s',color='blue')
ax2 = ax.twinx()
lns3 = ax2.plot(frame.index,chuva2,label='chuva',color='green')

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc = 0)


#Estabelece os intervalos dos eixos
ax2.set_yticks(np.arange(-1000,0,100))
ax.set_yticks(np.arange(0,maior_vazao,200))
ax.set_xticks(np.arange(0,len(frame),300))

#Insere nomes nos eixos
ax.set_xlabel('Dias')
ax.set_ylabel('Vazão')
ax2.set_ylabel('Chuva')

#Insere um título no gráfico
plt.title('Simulação histórica do modelo chuva-vazão diário')
plt.rcParams['figure.figsize'] = (100,70)

output_file2 = path_resultados + '\\' + 'Simulacao_historica.png'

plt.savefig(output_file2, bbox_inches='tight')

plt.show() #Mostrando gráfico

#gráficos julianos
#periodo de calibração

#Ler o arquivo em que se encontra o período de calibração
periodo = pd.read_csv(path_intervalo)
intervalo1 = int(periodo.loc[0, 'inicio'])
intervalo2 = int(periodo.loc[0, 'fim'])

frame_s = pd.DataFrame()



CAWM_RESULTADOS = pd.read_csv(output_file,float_precision='high')

frame_s = pd.concat([CAWM_RESULTADOS],axis=1)


s = frame_s.loc[intervalo1+1:intervalo2+1,'vazao_calc']
o = frame_s.loc[intervalo1+1:intervalo2+1, 'vazao']


vazao_obs_corrig = o.dropna()
o_med = vazao_obs_corrig.mean()
  
Nash_cali = NS(s, o)
NashsqrtQ_cali = NSsqrtQ(s,o)
Nashlog_cali = NSlog(s,o)
RMSE_cali = RMSE(s,o)
Pbias_cali = Pbias(s,o)
RSR_cali = RSR(s,o)

inicio = frame.loc[intervalo1,'data']
data_i = datetime.strptime(inicio, "%m/%d/%Y")

fim = frame.loc[intervalo2,'data']
data_f = datetime.strptime(fim, "%m/%d/%Y")
print("período de calibração:", data_i," a ",data_f)


media_vazao_observada_calib =[]
media_vazao_observada_calib2 =[]

#médias no ano
dias_ano = 365
for i in range(intervalo1, intervalo1 + dias_ano):
    dia = frame.loc[i, "data"]
    dia_datetime = datetime.strptime(dia, "%m/%d/%Y")
    mes = dia_datetime.month
    day = dia_datetime.day
    vazao_obs_cali = float((frame.loc[i, "vazao"]))
    vazao_calculada_cali = float((frame.loc[i, "vazao_calc"]))
    media = []
    media2 = []
    media.append(vazao_obs_cali)
    media2.append(vazao_calculada_cali)

    k = 1

    for j in range(i+1, intervalo2):  #periodo de calibração
        dia_cali = frame.loc[j, "data"]

        dia_cali_datetime = datetime.strptime(dia_cali, "%m/%d/%Y")
        mes2 = dia_cali_datetime.month
        day2 = dia_cali_datetime.day


        if mes == mes2 and day == day2:

            vazao_obs_cali2 = float(frame.loc[j, "vazao"])
            vazao_calculada_cali2 = float(frame.loc[j, "vazao_calc"])

            media.append(vazao_obs_cali2)

            media2.append(vazao_calculada_cali2)
            k+=1
    mediavazao = sum(media)/k


    mediavazao2 = sum(media2)/k
    media_vazao_observada_calib.append(mediavazao)
    media_vazao_observada_calib2.append(mediavazao2)



fig2 = plt.figure()
ax4 = fig2.add_subplot(111)

s1 = ax4.plot(media_vazao_observada_calib,label='Qobs m³/s',color='red')
s2 = ax4.plot(media_vazao_observada_calib2,label='Qcalc m³/s',color='blue')

lns = s1+s2
labs = [l.get_label() for l in lns]
ax4.legend(lns,labs,loc = 0)

#Insere nomes nos eixos
ax4.set_xlabel('Dias do ano')
ax4.set_ylabel('Vazão')


#Insere um título no gráfico
plt.title('Vazão média no período de calibração por dia do ano')
plt.rcParams['figure.figsize'] = (75,50)

output_file3 = path_resultados + '\\' + 'Vazao_media_calibracao.png'

plt.savefig(output_file3, bbox_inches='tight')

plt.show() #Mostrando gráfico

#período de validação
intervalo3 = intervalo2 + 1
intervalo4 = len(frame)-1
print('intervalo4', intervalo4)

s = frame_s.loc[intervalo3+1:intervalo4+1,'vazao_calc']
o = frame_s.loc[intervalo3+1:intervalo4+1, 'vazao']

vazao_obs_corrig = o.dropna()
o_med = vazao_obs_corrig.mean()
  
Nash_vali = NS(s, o)
NashsqrtQ_vali = NSsqrtQ(s,o)
Nashlog_vali = NSlog(s,o)
RMSE_vali = RMSE(s,o)
Pbias_vali = Pbias(s,o)
RSR_vali = RSR(s,o)

s = frame_s['vazao_calc']
o = frame_s['vazao']

vazao_obs_corrig = o.dropna()
o_med = vazao_obs_corrig.mean()
  
Nash = NS(s, o)
NashsqrtQ = NSsqrtQ(s,o)
Nashlog = NSlog(s,o)
RMSE = RMSE(s,o)
Pbias = Pbias(s,o)
RSR = RSR(s,o)

inicio_vali = frame.loc[intervalo3,'data']
data_i_vali = datetime.strptime(inicio_vali, "%m/%d/%Y")


final_vali = frame.loc[len(frame)-2, 'data']
data_f_vali = datetime.strptime(final_vali, "%m/%d/%Y")

print("período de validação: ", data_i_vali, "a", data_f_vali)

media_vazao_observada_vali =[]
media_vazao_observada_vali2 =[]
for i in range(intervalo3, intervalo3 + dias_ano):
    dia = frame.loc[i, "data"]
    dia_datetime = datetime.strptime(dia, "%m/%d/%Y")
    mes = dia_datetime.month
    day = dia_datetime.day
    vazao_obs_vali = float((frame.loc[i, "vazao"]))
    vazao_calculada_vali = float((frame.loc[i, "vazao_calc"]))
    media = []
    media2 = []
    media.append(vazao_obs_vali)
    media2.append(vazao_calculada_vali)

    k = 1

    for j in range(i + 1, intervalo4):  # periodo de validação
        dia_vali = frame.loc[j, "data"]
        dia_vali_datetime = datetime.strptime(dia_vali, "%m/%d/%Y")
        mes2 = dia_vali_datetime.month
        day2 = dia_vali_datetime.day

        if mes == mes2 and day == day2:
            vazao_obs_vali2 = float(frame.loc[j, "vazao"])
            vazao_calculada_vali2 = float(frame.loc[j, "vazao_calc"])
            media.append(vazao_obs_vali2)
            media2.append(vazao_calculada_vali2)
            k += 1
    mediavazao = sum(media) / k
    mediavazao2 = sum(media2) / k
    media_vazao_observada_vali.append(mediavazao)
    media_vazao_observada_vali2.append(mediavazao2)



fig2 = plt.figure()
ax4 = fig2.add_subplot(111)

s1 = ax4.plot(media_vazao_observada_vali,label='Qobs m³/s',color='red')
s2 = ax4.plot(media_vazao_observada_vali2,label='Qcalc m³/s',color='blue')

lns = s1+s2
labs = [l.get_label() for l in lns]
ax4.legend(lns,labs,loc = 0)

#Insere nomes nos eixos
ax4.set_xlabel('Dias do ano')
ax4.set_ylabel('Vazão')


#Insere um título no gráfico
plt.title('Vazão média no período de validação por dia do ano')
plt.rcParams['figure.figsize'] = (75,50)

output_file4 = path_resultados + '\\' + 'Vazao_media_validacao.png'

plt.savefig(output_file4, bbox_inches='tight')

plt.show() #Mostrando gráfico

#período total
dia_inicial = frame.loc[1,'data']
data_i_total = datetime.strptime(dia_inicial, "%m/%d/%Y")

dia_final = frame.loc[len(frame)-1, 'data']
data_f_total = datetime.strptime(dia_final, "%m/%d/%Y")

media_vazao_observada =[]
media_vazao_calculada =[]

for i in range(1, dias_ano):
    dia = frame.loc[i, "data"]
    dia_datetime = datetime.strptime(dia, "%m/%d/%Y")
    mes = dia_datetime.month
    day = dia_datetime.day
    vazao_obs = float((frame.loc[i, "vazao"]))
    vazao_calculada= float((frame.loc[i, "vazao_calc"]))
    media = []
    media2 = []
    media.append(vazao_obs)
    media2.append(vazao_calculada)

    k = 1
    for j in range(i + 1, len(frame) - 1):  # periodo total
        dia = frame.loc[j, "data"]
        dia_datetime = datetime.strptime(dia, "%m/%d/%Y")
        mes2 = dia_datetime.month
        day2 = dia_datetime.day

        if mes == mes2 and day == day2:
            vazao_obs = float(frame.loc[j, "vazao"])
            vazao_calculada = float(frame.loc[j, "vazao_calc"])
            media.append(vazao_obs)
            media2.append(vazao_calculada)
            k += 1
    mediavazao = sum(media) / k
    mediavazao2 = sum(media2) / k
    media_vazao_observada.append(mediavazao)
    media_vazao_calculada.append(mediavazao2)


fig2 = plt.figure()
ax4 = fig2.add_subplot(111)

s1 = ax4.plot(media_vazao_observada,label='Qobs m³/s',color='red')
s2 = ax4.plot(media_vazao_calculada,label='Qcalc m³/s',color='blue')

lns = s1+s2
labs = [l.get_label() for l in lns]
ax4.legend(lns,labs,loc = 0)

#Insere nomes nos eixos
ax4.set_xlabel('Dias do ano')
ax4.set_ylabel('Vazão')


#Insere um título no gráfico
plt.title('Vazão média no período total por dia do ano')
plt.rcParams['figure.figsize'] = (75,50)

output_file5 = path_resultados + '\\' + 'Vazao_media_periodo_total.png'

plt.savefig(output_file5, bbox_inches='tight')

plt.show() #Mostrando gráfico


#índices estatísticos
print('ÍNDICES ESTATÍSTICOS DO PERÍODO DE CALIBRAÇÃO:')
print('Nash de calibração:', Nash_cali)
print('Nash sqrtQ de calibração:', NashsqrtQ_cali)
print('Nashlog de calibração:', Nashlog_cali)
print('RMSE de calibração:', RMSE_cali)
print('RSE de calibração:', RSR_cali)
print('Pbias de calibração:', Pbias_cali)


print('ÍNDICES ESTATÍSTICOS DO PERÍODO DE VALIDAÇÃO:')
print('Nash de validação:', Nash_vali)
print('Nash sqrtQ de validação:', NashsqrtQ_vali)
print('Nashlog de validação:', Nashlog_vali)
print('RMSE de validação:', RMSE_vali)
print('RSE de validação:', RSR_vali)
print('Pbias de validação:', Pbias_vali)

print('ÍNDICES ESTATÍSTICOS DO PERÍODO GLOBAL:')
print('Nash global:', Nash)
print('Nash sqrtQ global:', NashsqrtQ)
print('Nashlog global:', Nashlog)
print('RMSE global:', RMSE)
print('RSE global:', RSR)
print('Pbias global:', Pbias)


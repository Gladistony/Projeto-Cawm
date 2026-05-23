from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df3 = pd.DataFrame()

# função preenche os vazios da série de vazão simulada e observada
def filter_nan(sim, obs):
    if sim.isnull().sum() >= 1 or obs.isnull().sum() >= 1:
        obs.isnull()
        obs.fillna(0, inplace=True)
        sim.isnull()
        sim.fillna(0, inplace=True)
        return sim, obs
    else:
        pass

# função para o cálculo do Nash
def NS(sim, obs):
    filter_nan(sim, obs)
    return 1 - sum((sim - obs) ** 2) / sum((obs - o_med) ** 2)

#função para o cálculo do R ao quadrado
def Rsquared():

    f_termo = sum((simulado - s_med) * (observado1 - o_media))
    s_termo = (sum((simulado - s_med) ** 2) * sum((observado1 - o_media) ** 2)) ** (1 / 2)
    rquad = (f_termo / s_termo) ** 2
    return rquad


s = pd.read_csv(r'resultados\CAWM.csv',usecols=['vazao_calc'])
o = pd.read_csv(r'resultados\CAWM.csv',usecols=['vazao'])
df = pd.concat([o, s], axis=1)


vazao_obs = df['vazao']

vazao_obs_corrig = vazao_obs.replace(np.nan, 0, regex=True)
o_med = mean(vazao_obs_corrig)

inter = pd.read_csv(r'dados\intervalo.csv')
qtde_i = len(inter)
print(qtde_i)


for i in range(qtde_i):
    x = 0
    print(x)
    while x < qtde_i:
        sim = df.loc[inter.loc[x, 'inicio']:inter.loc[x, 'fim'], 'vazao_calc']
        obs = df.loc[inter.loc[x, 'inicio']:inter.loc[x, 'fim'], 'vazao']
        NS(sim,obs)
        print(NS)
        df3.loc[x+1,'Nash'] = NS(sim,obs)
        x = x + 1

    s = pd.read_csv(r'resultados\CAWM.csv',
                usecols=['vazao_calc'])
    o = pd.read_csv(r'resultados\CAWM.csv',
                usecols=['vazao'])
    df = pd.concat([o, s], axis=1)


    x = 0
    while x <qtde_i:
        simulado = df.loc[inter.loc[x,'inicio']:inter.loc[x,'fim'],'vazao_calc']
        observado = df.loc[inter.loc[x,'inicio']:inter.loc[x,'fim'],'vazao']
        s_med = mean(simulado)
        o_media = mean(observado.dropna())
        observado1 = observado.replace(np.nan, 0, regex=True)
        df3.loc[x+1,'R^2'] = Rsquared()
        x = x + 1

    s = pd.read_csv(r'resultados\CAWM.csv',
                usecols=['vazao_calc'])
    o = pd.read_csv(r'resultados\CAWM.csv',
                usecols=['vazao'])

    df2 = pd.concat([o, s], axis=1)
    df2['vazao'] = df2['vazao'].replace(np.nan,0,regex=True)
    df2['soma_abs'] = abs(df2['vazao_calc']-df2['vazao'])

    x = 0
    while x <qtde_i:
        soma_abs1 = sum(df2.loc[inter.loc[x, 'inicio']:inter.loc[x, 'fim'], 'soma_abs'])
        df3.loc[x+1,'Soma_abs'] = soma_abs1
        x = x + 1

df3.to_csv(r'resultados/Resultados_estatisticos.csv', float_format='%.4f')

s = pd.read_csv(r'resultados\CAWM.csv',usecols=['vazao_calc'])
o = pd.read_csv(r'resultados\CAWM.csv',usecols=['vazao'])
chuva =pd.read_csv(r'resultados\CAWM.csv',usecols=['chuva_media'])
df4 = pd.concat([o, s,chuva], axis=1)
pp = PdfPages(r'resultados/Graficos_sem_otimizacao.pdf')
x = 0


while x<qtde_i:
    simulado = df4.loc[inter.loc[x, 'inicio']:inter.loc[x, 'fim'], 'vazao_calc']
    observado = df4.loc[inter.loc[x, 'inicio']:inter.loc[x, 'fim'], 'vazao']
    chuva1 = df4.loc[inter.loc[x, 'inicio']:inter.loc[x, 'fim'], 'chuva_media']
    df5 = pd.concat([simulado,observado,chuva1], axis=1)
    chuva2 = -1 * df5['chuva_media']


    fig = plt.figure()
    ax = fig.add_subplot(111)
    lns1 = ax.plot(df5.index, df5['vazao'], label='Qobs m³/s', color='red')
    lns2 = ax.plot(df5.index, df5['vazao_calc'], label='Qcalc m³/s', color='blue')
    ax2 = ax.twinx()
    lns3 = ax2.plot(df5.index, chuva2, label='chuva', color='green')
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    # Estabelece os intervalos dos eixos
    ax2.set_yticks(np.arange(-1000, 0, 100))
    ax.set_yticks(np.arange(0, 900, 100))
    ax.set_xticks(np.arange(inter.loc[x,'inicio'],inter.loc[x,'fim'],50))

    # Insere nomes nos eixos
    ax.set_xlabel('Dias corridos')
    ax.set_ylabel('Vazao')
    ax2.set_ylabel('Chuva')

    plt.title('CAWM: intervalo ' + str(inter.loc[x, 'inicio']) +' a '+ str(inter.loc[x, 'fim']))
    pp.savefig(bbox_inches='tight')
    x = x+1

pp.close()
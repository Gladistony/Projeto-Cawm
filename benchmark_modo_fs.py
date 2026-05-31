from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

from benchmark_cawm_final import (
    BASE_DIR,
    criar_pasta_resultado,
    extrair_dados_por_id,
    limpar_memoria_gpu,
    normalizar_nome_arquivo,
    pso_vetorizado,
    simular_forward_determinista,
    calcular_metricas,
)
from db_init import initialize_db
from db_models import CalibrationPeriod, Bacia

MODO_FS = True
W_FIXO = 0.5
C1_FIXO = 1.5
C2_FIXO = 1.5
PARTICULAS = 2000
ITERACOES = 100
PACIENCIA = 10
IDS_ALVO = [25, 27, 29, 31, 33]


def salvar_csv_local(df: pd.DataFrame, nome_arquivo: str, pasta_resultado: Path) -> None:
    df.to_csv(pasta_resultado / nome_arquivo, index=False)


def salvar_grafico_convergencia(df: pd.DataFrame, bacia: str, station: str, method: str, calib_id: int, pasta_resultado: Path) -> None:
    plt.figure(figsize=(10, 6))
    serie = df.sort_values('Iteracao')
    plt.plot(serie['Iteracao'], serie['FO'], marker='o', linewidth=2, color='navy')
    plt.title(f'Convergencia - {bacia} / {station} ({method})')
    plt.xlabel('Iteracao')
    plt.ylabel('FO')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    nome = f"Grafico_convergencia_{normalizar_nome_arquivo(bacia)}_{calib_id}.png"
    plt.savefig(pasta_resultado / nome, dpi=150, bbox_inches='tight')
    plt.close()


def salvar_grafico_tempos(df: pd.DataFrame, pasta_resultado: Path) -> None:
    if df.empty:
        return

    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Rotulo'] = df_plot['Bacia'] + ' | ' + df_plot['Station'].fillna('')
    ax = df_plot.plot(kind='bar', x='Rotulo', y='Tempo_s', legend=False, color='slateblue', figsize=(12, 6))
    ax.set_title('Tempo de execucao por bacia')
    ax.set_xlabel('Bacia / Estacao')
    ax.set_ylabel('Tempo (s)')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(pasta_resultado / 'Grafico_tempos_modo_fs.png', dpi=150, bbox_inches='tight')
    plt.close()


def salvar_pdf_hidrograma_bacia_fs(
    df_series: pd.DataFrame,
    nome_bacia: str,
    station: str,
    method: str,
    calib_id: int,
    intervalo_calib: tuple[int, int],
    intervalo_valid: tuple[int, int],
    pasta_resultado: Path,
) -> None:
    def plotar_intervalo(ax, ax2, inicio: int, fim: int, titulo: str) -> None:
        sub = df_series.loc[(df_series.index >= inicio) & (df_series.index <= fim), ['vazao', 'vazao_calc', 'chuva_media']].copy()
        chuva2 = -1 * sub['chuva_media']

        lns1 = ax.plot(sub.index, sub['vazao'], label='Qobs m³/s', color='red')
        lns2 = ax.plot(sub.index, sub['vazao_calc'], label='Qcalc m³/s', color='blue')
        lns3 = ax2.plot(sub.index, chuva2, label='chuva', color='green')
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax2.set_yticks(np.arange(-1000, 0, 100))
        limite_superior = max(900, float(sub['vazao'].max() or 0) + 100)
        ax.set_yticks(np.arange(0, limite_superior, 100))
        passo = max(1, (fim - inicio) // 8 or 1)
        ax.set_xticks(np.arange(inicio, fim + 1, passo))
        ax.set_xlabel('Dias corridos')
        ax.set_ylabel('Vazao')
        ax2.set_ylabel('Chuva')
        ax.set_title(titulo)

    nome = f'Graficos_hidrograma_{normalizar_nome_arquivo(nome_bacia)}_{calib_id}.pdf'
    caminho_pdf = pasta_resultado / nome
    with PdfPages(caminho_pdf) as pp:
        for etiqueta, intervalo in (('calibracao', intervalo_calib), ('validacao', intervalo_valid)):
            inicio, fim = intervalo
            fig = plt.figure(figsize=(11, 5))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            plotar_intervalo(ax, ax2, inicio, fim, f'CAWM: {nome_bacia} - {etiqueta} ({inicio} a {fim}) | {station} | {method}')
            pp.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def _lista_periodos() -> list[CalibrationPeriod]:
    engine, Session = initialize_db()
    session = Session()
    try:
        periodos = session.query(CalibrationPeriod).filter(CalibrationPeriod.id.in_(IDS_ALVO)).all()
        periodos_por_id = {int(periodo.id): periodo for periodo in periodos}
        return [periodos_por_id[calib_id] for calib_id in IDS_ALVO if calib_id in periodos_por_id]
    finally:
        session.close()
        engine.dispose()


def _warmup(primeiro_calib_id: int) -> None:
    P, E, Q_obs, mask_calib, _, area, SUBmax, a_param, _, _ = extrair_dados_por_id(primeiro_calib_id)
    indices_calib = np.where(mask_calib)[0]
    if len(indices_calib) == 0:
        return

    tamanho_warmup = min(60, len(indices_calib))
    limite = indices_calib[0] + tamanho_warmup
    P = P[:limite]
    E = E[:limite]
    Q_obs = Q_obs[:limite]
    mask_calib = mask_calib[:limite]
    pso_vetorizado(
        np,
        P,
        E,
        Q_obs,
        mask_calib,
        area,
        SUBmax,
        a_param,
        12,
        3,
        1,
        W_FIXO,
        C1_FIXO,
        C2_FIXO,
        fo='fo1',
    )


def _metricas_seguras(y_obs: np.ndarray, y_calc: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if len(y_obs) == 0 or len(y_calc) == 0:
        nan = float('nan')
        return nan, nan, nan, nan, nan, nan
    return calcular_metricas(y_obs, y_calc)


def executar_modo_fs() -> None:
    periodos = _lista_periodos()
    if not periodos:
        print('Nenhum periodo de calibracao encontrado no banco.')
        return

    pasta_resultado = criar_pasta_resultado()
    print('\n' + '=' * 90)
    print(' MODO_FS - CPU VETORIAL COM PARAMETROS FIXOS')
    print('=' * 90)
    print(f' Saida em: {pasta_resultado}')
    print(f' Parametros fixos: w={W_FIXO}, c1={C1_FIXO}, c2={C2_FIXO}')

    primeiro_id = int(periodos[0].id)
    print(f'\n[Warmup] Rodando aquecimento com o periodo {primeiro_id}...')
    _warmup(primeiro_id)

    resultados = []
    convergencias = []

    for periodo in periodos:
        calib_id = int(periodo.id)
        try:
            P, E, Q_obs, mask_calib, mask_valid, area, SUBmax, a_param, nome_bacia, metodo = extrair_dados_por_id(calib_id)
        except Exception as exc:
            print(f'  [ERRO] ID {calib_id}: {exc}')
            continue

        station = str(getattr(periodo, 'station', '') or '')
        print(f'\n▶ {nome_bacia} | {station} | metodo={metodo} | calib={int(mask_calib.sum())} dias')

        t_inicio = time.perf_counter()
        nash_final, nse_final, historico, ks_f, expo_f, iteracao_conv = pso_vetorizado(
            np,
            P,
            E,
            Q_obs,
            mask_calib,
            area,
            SUBmax,
            a_param,
            PARTICULAS,
            ITERACOES,
            PACIENCIA,
            W_FIXO,
            C1_FIXO,
            C2_FIXO,
            fo='fo1',
        )
        tempo_exec = time.perf_counter() - t_inicio

        Q_calc_total, kl_calc = simular_forward_determinista(P, E, Q_obs, mask_calib, area, SUBmax, a_param, ks_f, expo_f)

        Q_obs_calib, Q_calc_calib = Q_obs[mask_calib], Q_calc_total[mask_calib]
        Q_obs_valid, Q_calc_valid = Q_obs[mask_valid], Q_calc_total[mask_valid]
        nse_c, nsel_c, nslog_leg_c, nses_c, pbias_c, rmse_c = _metricas_seguras(Q_obs_calib, Q_calc_calib)
        nse_v, nsel_v, nslog_leg_v, nses_v, pbias_v, rmse_v = _metricas_seguras(Q_obs_valid, Q_calc_valid)
        nse_t, nsel_t, nslog_leg_t, nses_t, pbias_t, rmse_t = _metricas_seguras(Q_obs, Q_calc_total)

        convergencia = pd.DataFrame(
            {
                'ID': calib_id,
                'Bacia': nome_bacia,
                'Station': station,
                'Metodo': metodo,
                'Iteracao': list(range(1, len(historico) + 1)),
                'FO': historico,
            }
        )

        serie = pd.DataFrame(
            {
                'vazao': Q_obs,
                'vazao_calc': Q_calc_total,
                'chuva_media': P,
            }
        )
        idx_calib = np.where(mask_calib)[0]
        idx_valid = np.where(mask_valid)[0]

        if len(idx_calib) > 0 and len(idx_valid) > 0:
            salvar_pdf_hidrograma_bacia_fs(
                serie,
                nome_bacia,
                station,
                str(metodo),
                calib_id,
                (int(idx_calib[0]), int(idx_calib[-1])),
                (int(idx_valid[0]), int(idx_valid[-1])),
                pasta_resultado,
            )

        salvar_grafico_convergencia(convergencia, nome_bacia, station, str(metodo), calib_id, pasta_resultado)
        salvar_csv_local(serie, f'CAWM_{normalizar_nome_arquivo(nome_bacia)}_{calib_id}.csv', pasta_resultado)

        resultados.append(
            {
                'ID': calib_id,
                'Bacia': nome_bacia,
                'Station': station,
                'Metodo': metodo,
                'Tempo_s': round(tempo_exec, 2),
                'Ks': round(ks_f, 4),
                'Expo': round(expo_f, 4),
                'Kl': round(kl_calc, 4),
                'NSE_Cal': round(nse_c, 4),
                'NSE_Log_Cal': round(nsel_c, 4),
                'NSE_Log_Legado_Cal': round(nslog_leg_c, 4),
                'NSE_Sqrt_Cal': round(nses_c, 4),
                'PBIAS_Cal': round(pbias_c, 2),
                'RMSE_Cal': round(rmse_c, 2),
                'NSE_Val': round(nse_v, 4),
                'NSE_Log_Val': round(nsel_v, 4),
                'NSE_Log_Legado_Val': round(nslog_leg_v, 4),
                'NSE_Sqrt_Val': round(nses_v, 4),
                'PBIAS_Val': round(pbias_v, 2),
                'RMSE_Val': round(rmse_v, 2),
                'NSE_Total': round(nse_t, 4),
                'NSE_Log_Total': round(nsel_t, 4),
                'NSE_Log_Legado_Total': round(nslog_leg_t, 4),
                'NSE_Sqrt_Total': round(nses_t, 4),
                'PBIAS_Total': round(pbias_t, 2),
                'RMSE_Total': round(rmse_t, 2),
                'Iter_Conv': int(iteracao_conv),
            }
        )

        convergencias.extend(convergencia.to_dict(orient='records'))

        print(
            f"    Tempo={tempo_exec:.2f}s | Ks={ks_f:.4f} | Expo={expo_f:.4f} | Kl={kl_calc:.4f} | "
            f"NSE Cal={nse_c:.4f} | NSE Val={nse_v:.4f}"
        )

    df_resultados = pd.DataFrame(resultados)
    df_convergencias = pd.DataFrame(convergencias)

    salvar_csv_local(df_resultados, 'modo_fs_resultados.csv', pasta_resultado)
    salvar_csv_local(df_convergencias, 'modo_fs_convergencia.csv', pasta_resultado)
    salvar_grafico_tempos(df_resultados, pasta_resultado)

    print('\n' + '=' * 120)
    print(' RESUMO FINAL - MODO_FS')
    print('=' * 120)
    if not df_resultados.empty:
        print(df_resultados.to_string(index=False))
    print(f'\n[SUCESSO] Arquivos gerados em {pasta_resultado}')


if __name__ == '__main__':
    executar_modo_fs()
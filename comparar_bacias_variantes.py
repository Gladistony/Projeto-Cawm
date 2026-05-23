"""
Compara bacias com e sem versão reduzida dentro de Dados/.

O foco é:
- localizar pares normal/reduzido quando existirem
- usar a mesma lógica do CAWM para extrair parâmetros
- apontar diferenças de parâmetros e do intervalo.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from funcoes import extrair_parametros_shapes


SUFIXO_REDUZIDO = " REDUZIDO"


@dataclass
class VarianteBacia:
    regiao: str
    nome_base: str
    nome_pasta: str
    caminho_bacia: Path
    caminho_shape: Path | None
    variante: str


def _buscar_shapefile(caminho_bacia: Path) -> Path | None:
    shapes_dir = caminho_bacia / "shapes"
    if not shapes_dir.exists():
        return None
    shapefiles = sorted(shapes_dir.glob("*.shp"))
    return shapefiles[0] if shapefiles else None


def _classificar_bacia(nome_pasta: str) -> tuple[str, str]:
    if nome_pasta.upper().endswith(SUFIXO_REDUZIDO):
        return nome_pasta[: -len(SUFIXO_REDUZIDO)], "reduzido"
    return nome_pasta, "normal"


def localizar_variantes(pasta_dados: str = "Dados") -> dict[str, list[VarianteBacia]]:
    raiz = Path(pasta_dados)
    agrupadas: dict[str, list[VarianteBacia]] = defaultdict(list)

    for regiao_dir in sorted(raiz.iterdir() if raiz.exists() else []):
        if not regiao_dir.is_dir():
            continue

        for bacia_dir in sorted(regiao_dir.iterdir()):
            if not bacia_dir.is_dir():
                continue

            nome_base, variante = _classificar_bacia(bacia_dir.name)
            agrupadas[nome_base].append(
                VarianteBacia(
                    regiao=regiao_dir.name,
                    nome_base=nome_base,
                    nome_pasta=bacia_dir.name,
                    caminho_bacia=bacia_dir,
                    caminho_shape=_buscar_shapefile(bacia_dir),
                    variante=variante,
                )
            )

    return agrupadas


def _resumir_intervalo(caminho_intervalo: Path) -> dict[str, Any]:
    if not caminho_intervalo.exists():
        return {"existe": False}

    with caminho_intervalo.open("r", encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        linhas = list(leitor)

    if not linhas:
        return {"existe": True, "vazio": True}

    primeira = linhas[0]
    ultima = linhas[-1]
    inicio = primeira.get("inicio")
    fim = primeira.get("fim")

    return {
        "existe": True,
        "vazio": False,
        "linhas": len(linhas),
        "inicio": inicio,
        "fim": fim,
        "primeira_linha": primeira,
        "ultima_linha": ultima,
    }


def _comparar_dicts(base: dict[str, Any], outro: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    chaves = sorted(set(base) | set(outro))
    diferencas: dict[str, tuple[Any, Any]] = {}
    for chave in chaves:
        if base.get(chave) != outro.get(chave):
            diferencas[chave] = (base.get(chave), outro.get(chave))
    return diferencas


def comparar_variantes(pasta_dados: str = "Dados") -> None:
    print("=" * 80)
    print("COMPARAÇÃO DE BACIAS NORMAL x REDUZIDO")
    print("=" * 80)

    agrupadas = localizar_variantes(pasta_dados)

    pares_completos = []
    variantes_unicas = []
    for nome_base, variantes in sorted(agrupadas.items()):
        if len(variantes) >= 2:
            pares_completos.append((nome_base, variantes))
        else:
            variantes_unicas.append((nome_base, variantes[0]))

    print(f"\nBacias com par normal/reduzido: {len(pares_completos)}")
    print(f"Bacias sem par completo: {len(variantes_unicas)}")

    # preparar payload para a função do CAWM
    payload: dict[str, dict[str, str]] = {}
    for nome_base, variantes in pares_completos:
        for variante in variantes:
            if variante.caminho_shape is None:
                continue
            chave = f"{nome_base}::{variante.variante}"
            payload[chave] = {
                "pasta_bacia": str(variante.caminho_bacia),
                "shape": str(variante.caminho_shape),
            }

    print("\nExtraindo parâmetros com a mesma lógica do CAWM...")
    parametros = extrair_parametros_shapes(payload)

    for nome_base, variantes in pares_completos:
        print("\n" + "-" * 80)
        print(f"Bacia: {nome_base}")

        por_variante = {v.variante: v for v in variantes}
        normal = por_variante.get("normal")
        reduzido = por_variante.get("reduzido")

        if normal is None or reduzido is None:
            print("Par incompleto; não foi possível comparar as duas versões.")
            continue

        chave_normal = f"{nome_base}::normal"
        chave_reduzido = f"{nome_base}::reduzido"
        params_normal = parametros.get(chave_normal)
        params_reduzido = parametros.get(chave_reduzido)

        if not params_normal or not params_reduzido:
            print("Falha na extração de parâmetros para uma das versões.")
            continue

        diffs = _comparar_dicts(params_normal, params_reduzido)
        print(f"Região: {normal.regiao}")
        print(f"Pasta normal:   {normal.caminho_bacia}")
        print(f"Pasta reduzida: {reduzido.caminho_bacia}")

        intervalo_normal = _resumir_intervalo(normal.caminho_bacia / "intervalo.csv")
        intervalo_reduzido = _resumir_intervalo(reduzido.caminho_bacia / "intervalo.csv")

        print("\nIntervalo.csv")
        print(f"  normal:   {json.dumps(intervalo_normal, ensure_ascii=False)}")
        print(f"  reduzido: {json.dumps(intervalo_reduzido, ensure_ascii=False)}")

        print("\nDiferenças nos parâmetros extraídos:")
        if diffs:
            for chave, (valor_normal, valor_reduzido) in diffs.items():
                print(f"  - {chave}: {valor_normal} -> {valor_reduzido}")
        else:
            print("  Nenhuma diferença nos parâmetros extraídos.")

        if intervalo_normal != intervalo_reduzido:
            print("\nDiagnóstico: os parâmetros são equivalentes, mas o intervalo.csv é diferente.")
            print("Isso indica que pode fazer sentido unificar a bacia base e guardar apenas o processamento por intervalo.")
        else:
            print("\nDiagnóstico: até o intervalo.csv parece equivalente.")

    if variantes_unicas:
        print("\n" + "=" * 80)
        print("Bacias sem versão reduzida correspondente:")
        for nome_base, variante in variantes_unicas:
            print(f"- {nome_base} ({variante.variante}) -> {variante.caminho_bacia}")


if __name__ == "__main__":
    comparar_variantes()

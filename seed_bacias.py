"""
Script para popular a tabela de bacias no banco de dados.

Reutiliza a função extrair_parametros_shapes de funcoes.py para
extrair os parâmetros básicos de cada bacia e insere na tabela.
"""
import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

from db_init import initialize_db
from db_models import Base, Bacia
from funcoes import extrair_parametros_shapes


def construir_dicionario_bacias(pasta_raiz="Dados"):
    """
    Constrói o dicionário de bacias a partir da estrutura de diretórios.
    
    Formato esperado: Dados/{REGIAO}/{BACIA}/
    Retorna: {"BACIA": {"pasta_bacia": "...", "shape": "...", "regiao": "..."}, ...}
    """
    dicionario_bacias = {}
    pasta_dados = Path(pasta_raiz)
    
    if not pasta_dados.exists():
        print(f"❌ Pasta {pasta_dados} não encontrada")
        return dicionario_bacias
    
    # Iterar sobre regiões
    for regiao_dir in pasta_dados.iterdir():
        if not regiao_dir.is_dir():
            continue
        
        regiao = regiao_dir.name
        
        # Iterar sobre bacias dentro de cada região
        for bacia_dir in regiao_dir.iterdir():
            if not bacia_dir.is_dir():
                continue
            
            bacia = bacia_dir.name
            
            # Procurar pelo shapefile
            shapes_dir = bacia_dir / "shapes"
            if shapes_dir.exists():
                shapefiles = list(shapes_dir.glob("*.shp"))
                if shapefiles:
                    # Usar o primeiro shapefile encontrado
                    dicionario_bacias[bacia] = {
                        "pasta_bacia": str(bacia_dir),
                        "shape": str(shapefiles[0]),
                        "regiao": regiao
                    }
    
    return dicionario_bacias


def popular_bacias():
    """
    Extrai os parâmetros das bacias e popula a tabela no banco de dados.
    """
    print("=" * 70)
    print("📊 POPULANDO TABELA DE BACIAS")
    print("=" * 70)
    
    # 1. Construir dicionário de bacias
    print("\n1️⃣ Mapeando estrutura de diretórios...")
    dicionario_bacias = construir_dicionario_bacias()
    
    if not dicionario_bacias:
        print("❌ Nenhuma bacia encontrada")
        return
    
    print(f"✅ {len(dicionario_bacias)} bacia(s) encontrada(s): {list(dicionario_bacias.keys())}")
    
    # 2. Extrair parâmetros usando a função do CAWM
    print("\n2️⃣ Extraindo parâmetros das bacias...")
    bacias_com_params = extrair_parametros_shapes(dicionario_bacias)
    
    if not bacias_com_params:
        print("❌ Nenhuma bacia com parâmetros válidos")
        return
    
    print(f"\n✅ {len(bacias_com_params)} bacia(s) com parâmetros extraídos com sucesso")
    
    # 3. Inicializar banco de dados
    print("\n3️⃣ Inicializando banco de dados...")
    engine, SessionLocal = initialize_db()
    session = SessionLocal()
    
    try:
        # 4. Popular tabela de bacias
        print("\n4️⃣ Inserindo bacias no banco de dados...")
        bacias_inseridas = 0
        bacias_atualizadas = 0
        
        for nome_bacia, parametros in bacias_com_params.items():
            # Verificar se bacia já existe
            bacia_existente = session.query(Bacia).filter_by(nome=nome_bacia).first()
            
            if bacia_existente:
                # Atualizar dados existentes
                for chave, valor in parametros.items():
                    if hasattr(bacia_existente, chave):
                        setattr(bacia_existente, chave, valor)
                bacias_atualizadas += 1
                status = "🔄 Atualizada"
            else:
                # Inserir nova bacia
                bacia = Bacia(nome=nome_bacia, **parametros)
                session.add(bacia)
                bacias_inseridas += 1
                status = "✨ Inserida"
            
            # Exibir resumo
            area = parametros.get("area_km2", "?")
            regiao = parametros.get("regiao", "?")
            print(f"   {status}: {nome_bacia} ({regiao}) - Área: {area} km²")
        
        # Fazer commit
        session.commit()
        
        print("\n" + "=" * 70)
        print(f"✅ SUCESSO!")
        print(f"   - Bacias inseridas: {bacias_inseridas}")
        print(f"   - Bacias atualizadas: {bacias_atualizadas}")
        print(f"   - Total: {bacias_inseridas + bacias_atualizadas}")
        print("=" * 70)
        
    except Exception as e:
        session.rollback()
        print(f"\n❌ Erro ao popular bacias: {e}")
        raise
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    popular_bacias()

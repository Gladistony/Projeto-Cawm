"""
Módulo mínimo de inicialização do banco SQLite usando SQLAlchemy.

Fornece funções simples para obter o engine, Session e inicializar
uma conexão (criar o arquivo SQLite vazio se não existir).

Uso básico:
    from db_init import initialize_db
    initialize_db('cawm.db')

Por enquanto não define models — apenas conexão e sessão.
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def get_sqlite_url(db_path: str) -> str:
    # Usa caminho absoluto para evitar surpresas
    db_path = os.path.abspath(db_path)
    return f"sqlite:///{db_path}"


def get_engine(db_path: str = "cawm.db", **engine_kwargs):
    """Cria e retorna um SQLAlchemy Engine apontando para `db_path`.

    engine_kwargs são passados para `create_engine`.
    """
    url = get_sqlite_url(db_path)
    # echo=False por padrão; caller pode sobrescrever
    engine = create_engine(url, echo=engine_kwargs.pop("echo", False), **engine_kwargs)
    return engine


def get_sessionmaker(engine, **session_kwargs):
    """Retorna um `sessionmaker` ligado ao engine fornecido."""
    return sessionmaker(bind=engine, **session_kwargs)


def initialize_db(db_path: str = "cawm.db") -> tuple:
    """Inicializa conexão com o banco SQLite e testa a operação mínima.

    Retorna (engine, Session) para uso posterior.
    Cria o arquivo do banco se não existir (SQLite faz isso automaticamente
    ao conectar via SQLAlchemy). Também executa um comando simples para
    verificar a conectividade.
    """
    engine = get_engine(db_path)
    Session = get_sessionmaker(engine)

    # Teste rápido de conexão
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        # Re-raise com mensagem clara
        raise RuntimeError(f"Falha ao conectar ao banco SQLite '{db_path}': {exc}") from exc

    return engine, Session


if __name__ == "__main__":
    # Executável simples para inicializar o DB
    import argparse

    parser = argparse.ArgumentParser(description="Inicializa conexão SQLite via SQLAlchemy")
    parser.add_argument("--db", default="cawm.db", help="Caminho para o arquivo SQLite")
    args = parser.parse_args()

    eng, Sess = initialize_db(args.db)
    print(f"✅ Conectado ao banco: {os.path.abspath(args.db)}")
    print("  Engine:", eng)
    print("  Use: from db_init import get_engine, get_sessionmaker, initialize_db")

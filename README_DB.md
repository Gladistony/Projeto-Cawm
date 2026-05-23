# README — Inicialização do Banco SQLite (SQLAlchemy)

Objetivo
--------
Fornecer um módulo mínimo que permita conectar-se a um arquivo SQLite usando SQLAlchemy.

Instalação
---------
Instale as dependências listadas em `requirements.txt` (adicionamos `SQLAlchemy`):

```bash
pip install -r requirements.txt
```

Uso rápido
----------
Para inicializar (cria/abre o arquivo SQLite e testa a conexão):

```bash
python db_init.py --db cawm.db
```

Ou em Python:

```python
from db_init import initialize_db
engine, Session = initialize_db('cawm.db')

# Usar Session para operações:
with Session() as s:
    # exemplo: executar um SQL simples
    s.execute('SELECT 1')

```

Próximos passos
---------------
- Definir modelos SQLAlchemy (ORM) e criar tabelas via `Base.metadata.create_all(engine)`
- Implementar migração de dados (CSV → DB)
- Adicionar helpers de CRUD e scripts de manutenção

Por enquanto este repositório fornece apenas a base de conexão.

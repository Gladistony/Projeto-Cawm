"""
Configurações globais para os testes.

Garante que o ambiente de teste é isolado do de produção.
"""
import os

# Sempre ativar modo teste para os testes
os.environ["CAWM_ENV"] = "test"

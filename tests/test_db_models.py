"""
Testes para db_models.py
"""
import sys
import os
import unittest
from pathlib import Path
from datetime import datetime

# Ativar modo teste ANTES de importar db_init
os.environ["CAWM_ENV"] = "test"

# Adiciona o diretório pai ao path para importar módulos do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_init import get_engine, get_sessionmaker
from db_models import Base, Bacia
import tempfile


class TestBaciaModel(unittest.TestCase):
    """Testes para o modelo Bacia."""

    def setUp(self):
        """Configura um banco de dados temporário para cada teste."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.engine = get_engine(self.db_path)
        self.SessionLocal = get_sessionmaker(self.engine)

        # Criar todas as tabelas
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        """Limpa o banco de dados temporário."""
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        import os
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_bacia_creation(self):
        """Testa se uma bacia pode ser criada e inserida no banco."""
        session = self.SessionLocal()
        try:
            bacia = Bacia(
                nome="LIMOEIRO",
                regiao="CAPIBARIBE_PSO",
                area_km2=250.5,
                k=0.1,
                a=1.2,
                expo_perdas=0.5,
                beta=2.0,
                kg=0.8,
                rio=1,
                submax=100.0,
                gmax=50.0,
                reserva_solo_inicial=50.0,
                profundo_inicial=30.0,
                s3_inicial=20.0,
            )
            session.add(bacia)
            session.commit()

            # Recuperar e verificar
            recovered = session.query(Bacia).filter_by(nome="LIMOEIRO").first()
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.area_km2, 250.5)
            self.assertEqual(recovered.k, 0.1)
        finally:
            session.close()

    def test_bacia_to_dict(self):
        """Testa o método to_dict da bacia."""
        # Necessário adicionar à sessão para que defaults sejam aplicados
        session = self.SessionLocal()
        try:
            bacia = Bacia(
                nome="PAUDALHO",
                regiao="CAPIBARIBE_PSO",
                area_km2=300.0,
                k=0.15,
                a=1.5,
            )
            session.add(bacia)
            session.flush()  # Aplicar defaults
            bacia_dict = bacia.to_dict()

            self.assertIn("nome", bacia_dict)
            self.assertIn("area_km2", bacia_dict)
            self.assertEqual(bacia_dict["nome"], "PAUDALHO")
            self.assertEqual(bacia_dict["area_km2"], 300.0)
            self.assertEqual(bacia_dict["s1_inicial"], 0.0)  # valor default
            self.assertEqual(bacia_dict["b"], 1.666666667)  # valor default
        finally:
            session.close()

    def test_bacia_defaults(self):
        """Testa se os valores padrão são aplicados corretamente."""
        session = self.SessionLocal()
        try:
            bacia = Bacia(nome="TESTE", area_km2=100.0)
            session.add(bacia)
            session.commit()

            recovered = session.query(Bacia).filter_by(nome="TESTE").first()
            self.assertEqual(recovered.s1_inicial, 0.0)
            self.assertEqual(recovered.s2_inicial, 0.0)
            self.assertEqual(recovered.b, 1.666666667)
            self.assertEqual(recovered.T, 86400.0)
        finally:
            session.close()

    def test_bacia_unique_constraint(self):
        """Testa se o nome da bacia é único."""
        session = self.SessionLocal()
        try:
            bacia1 = Bacia(nome="DUPLICADA", area_km2=100.0)
            bacia2 = Bacia(nome="DUPLICADA", area_km2=200.0)

            session.add(bacia1)
            session.commit()

            session.add(bacia2)
            # Deve lançar uma exceção de integridade
            with self.assertRaises(Exception):
                session.commit()
        finally:
            session.close()

    def test_bacia_repr(self):
        """Testa a representação em string da bacia."""
        bacia = Bacia(id=1, nome="LIMOEIRO", area_km2=250.5)
        repr_str = repr(bacia)

        self.assertIn("Bacia", repr_str)
        self.assertIn("LIMOEIRO", repr_str)
        self.assertIn("250.5", repr_str)


if __name__ == "__main__":
    unittest.main()

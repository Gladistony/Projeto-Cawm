"""
Teste para seed_bacias.py
"""
import sys
import os
import unittest
from pathlib import Path

# Ativar modo teste ANTES de importar db_init
os.environ["CAWM_ENV"] = "test"

# Adicionar o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_init import initialize_db, get_engine
from db_models import Base, Bacia
from seed_bacias import construir_dicionario_bacias
import tempfile


class TestSeedBacias(unittest.TestCase):
    """Testes para o seed de bacias."""

    def setUp(self):
        """Configura um banco de dados temporário para cada teste."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        self.engine = get_engine(self.db_path)
        self.SessionLocal = initialize_db(self.db_path)[1]

    def tearDown(self):
        """Limpa o banco de dados temporário."""
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        import os
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_seed_bacias_criadas_com_sucesso(self):
        """Testa se as bacias foram inseridas com sucesso no banco populado."""
        session = self.SessionLocal()
        try:
            # Inserir bacias de teste
            bacias_teste = [
                Bacia(
                    nome="TESTE_BACIA_1",
                    regiao="TEST_REGION_1",
                    area_km2=1000.0,
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
                ),
                Bacia(
                    nome="TESTE_BACIA_2",
                    regiao="TEST_REGION_2",
                    area_km2=2000.0,
                    k=0.2,
                    a=1.5,
                ),
            ]

            for bacia in bacias_teste:
                session.add(bacia)
            session.commit()

            # Verificar
            total = session.query(Bacia).count()
            self.assertEqual(total, 2)

            # Verificar campos
            bacia1 = session.query(Bacia).filter_by(nome="TESTE_BACIA_1").first()
            self.assertIsNotNone(bacia1)
            self.assertEqual(bacia1.area_km2, 1000.0)
            self.assertEqual(bacia1.k, 0.1)
            self.assertEqual(bacia1.a, 1.2)
            self.assertEqual(bacia1.expo_perdas, 0.5)
            self.assertEqual(bacia1.beta, 2.0)
            self.assertEqual(bacia1.reserva_solo_inicial, 50.0)

        finally:
            session.close()

    def test_seed_bacias_todos_parametros(self):
        """Testa se todos os parâmetros são salvos corretamente."""
        session = self.SessionLocal()
        try:
            bacia = Bacia(
                nome="BACIA_COMPLETA",
                regiao="REGION_TEST",
                area_km2=3500.5,
                k=0.35,
                a=1.4,
                expo_perdas=0.75,
                beta=1.8,
                kg=0.95,
                rio=1,
                submax=120.5,
                gmax=60.3,
                reserva_solo_inicial=45.2,
                profundo_inicial=28.7,
                s3_inicial=18.3,
            )
            session.add(bacia)
            session.commit()

            # Recuperar e verificar
            recovered = session.query(Bacia).filter_by(nome="BACIA_COMPLETA").first()
            self.assertEqual(recovered.area_km2, 3500.5)
            self.assertEqual(recovered.k, 0.35)
            self.assertEqual(recovered.a, 1.4)
            self.assertEqual(recovered.expo_perdas, 0.75)
            self.assertEqual(recovered.beta, 1.8)
            self.assertEqual(recovered.kg, 0.95)
            self.assertEqual(recovered.rio, 1)
            self.assertEqual(recovered.submax, 120.5)
            self.assertEqual(recovered.gmax, 60.3)
            self.assertEqual(recovered.reserva_solo_inicial, 45.2)
            self.assertEqual(recovered.profundo_inicial, 28.7)
            self.assertEqual(recovered.s3_inicial, 18.3)

        finally:
            session.close()

    def test_construir_dicionario_ignora_reduzidas(self):
        """Testa se o scanner do seed ignora pastas reduzidas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            raiz = Path(temp_dir)

            regiao = raiz / "PAJEU_PSO"
            normal = regiao / "SERRA TALHADA"
            reduzida = regiao / "SERRA TALHADA REDUZIDO"

            for pasta in (normal, reduzida):
                (pasta / "shapes").mkdir(parents=True)
                (pasta / "shapes" / "base.shp").touch()

            resultado = construir_dicionario_bacias(str(raiz))

            self.assertIn("SERRA TALHADA", resultado)
            self.assertNotIn("SERRA TALHADA REDUZIDO", resultado)
            self.assertEqual(resultado["SERRA TALHADA"]["regiao"], "PAJEU_PSO")


if __name__ == "__main__":
    unittest.main()

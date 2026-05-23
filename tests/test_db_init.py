import sys
import tempfile
import unittest
from pathlib import Path

# Adicionar o diretório raiz ao path para importar db_init
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_init import initialize_db


class TestDbInit(unittest.TestCase):
    def test_initialize_db_creates_sqlite_file_and_connects(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cawm_test.db"

            engine, session_factory = initialize_db(str(db_path))

            self.assertTrue(db_path.exists(), "O arquivo SQLite deveria ser criado")

            with engine.connect() as connection:
                result = connection.exec_driver_sql("SELECT 1").scalar_one()

            self.assertEqual(result, 1)

            session = session_factory()
            try:
                value = session.connection().exec_driver_sql("SELECT 1").scalar_one()
                self.assertEqual(value, 1)
            finally:
                session.close()
                engine.dispose()


if __name__ == "__main__":
    unittest.main(verbosity=2)
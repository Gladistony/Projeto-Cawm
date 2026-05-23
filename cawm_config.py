import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Dados"
DEFAULT_BASE_PATHS = [
    str(DEFAULT_DATA_DIR / "CAPIBARIBE_PSO"),
    str(DEFAULT_DATA_DIR / "PAJEU_PSO"),
    str(DEFAULT_DATA_DIR / "Ceara"),
]


def get_base_paths() -> list[str]:
    env_paths = os.getenv("CAWM_BASE_PATHS")
    if env_paths:
        paths = [path.strip() for path in env_paths.split(os.pathsep)]
        return [path for path in paths if path]
    return DEFAULT_BASE_PATHS.copy()


def get_array_backend(prefer_gpu: bool = True):
    if not prefer_gpu:
        return np, False

    try:
        import cupy as cp

        if cp.cuda.is_available():
            print(f"✅ GPU detectada! Usando CuPy. Dispositivos: {cp.cuda.runtime.getDeviceCount()}")
            return cp, True

        print("⚠️ CuPy instalado, mas GPU não disponível. Usando NumPy (CPU).")
    except ImportError:
        print("⚠️ CuPy não encontrado. Usando NumPy (CPU).")

    return np, False
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("MASS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATASET_DIR = Path(os.environ.get("MASS_DATASET_DIR", PROJECT_ROOT / "stock_disagreement" / "dataset"))
RESULT_DIR = Path(os.environ.get("MASS_RESULT_DIR", PROJECT_ROOT / "stock_disagreement" / "res"))
LLM_MODEL_NAME = os.environ.get("MASS_LLM_MODEL_NAME", "deepseek-v4-flash")
LLM_BASE_URL = os.environ.get("MASS_LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("MASS_LLM_API_KEY", "")
LLM_MAX_TOKENS = int(os.environ.get("MASS_LLM_MAX_TOKENS", "4096"))


def data_path(*parts: str) -> Path:
    return DATASET_DIR.joinpath(*parts)


def result_path(*parts: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR.joinpath(*parts)


def first_existing_data_path(*names: str) -> Path:
    for name in names:
        path = data_path(name)
        if path.exists() and path.stat().st_size > 8:
            return path
    joined = ", ".join(names)
    raise FileNotFoundError(f"None of these dataset files exist or contain valid data: {joined}")

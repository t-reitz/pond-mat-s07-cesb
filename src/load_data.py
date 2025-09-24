"""
Carregamento dos dados a partir do Excel e tratamento mínimo de datas.

- Usa o caminho e a planilha padrão definidos em config.py
- Converte automaticamente colunas cujo nome começa com "Data_" para datetime
- Remove colunas duplicadas (caso existam nomes repetidos no Excel)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import DEFAULT_DATA_XLSX, DEFAULT_SHEET

# Prefixo que usamos como "heurística" para identificar colunas de datas
DATE_PREFIX = "Data_"

def load_data(path: Path | str = DEFAULT_DATA_XLSX, sheet: str = DEFAULT_SHEET) -> pd.DataFrame:
    """
    Lê o Excel e retorna um DataFrame.
    - path: caminho do arquivo .xlsx
    - sheet: nome da aba a ser lida
    """
    path = Path(path)                     # garante objeto Path
    df = pd.read_excel(path, sheet_name=sheet)  # carrega a planilha

    # Converte para datetime todas as colunas com prefixo "Data_"
    for col in df.columns:
        if col.startswith(DATE_PREFIX):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove colunas duplicadas mantendo a primeira ocorrência
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df

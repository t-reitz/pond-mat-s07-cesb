"""
Funções extra para correlação com p-valor (Pearson) entre o alvo e as features numéricas.
Gera um CSV com colunas: feature, r_pearson, p_value, n (tamanho da amostra usada).
"""

import numpy as np
import pandas as pd
from scipy import stats
from .config import TARGET, OUTPUT_TABLES

def corr_with_target_and_pvalue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula correlação de Pearson e p-valor entre o TARGET e cada coluna numérica de df.
    - Remove pares com NaN antes de calcular
    - Retorna DataFrame ordenado por |r| desc
    """
    # Seleciona apenas numéricas
    num = df.select_dtypes(include=[np.number]).copy()

    # Se não existir o TARGET como numérico, retorna vazio
    if TARGET not in num.columns:
        return pd.DataFrame(columns=["feature", "r_pearson", "p_value", "n"])

    y = num[TARGET]
    out = []
    for col in num.columns:
        if col == TARGET:
            continue
        # Alinha pares válidos (sem NaN)
        mask = ~(y.isna() | num[col].isna())
        yy = y[mask].values
        xx = num[col][mask].values
        if len(xx) < 3:
            continue
        # Correlação de Pearson e p-valor
        r, p = stats.pearsonr(xx, yy)
        out.append({"feature": col, "r_pearson": float(r), "p_value": float(p), "n": int(len(xx))})

    # DataFrame ordenado por |r|
    res = pd.DataFrame(out).sort_values(by="r_pearson", key=lambda s: s.abs(), ascending=False)

    # Salva CSV para usar nos slides/relatórios
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT_TABLES / "corr_with_target_pearson_pvalues.csv", index=False)
    return res

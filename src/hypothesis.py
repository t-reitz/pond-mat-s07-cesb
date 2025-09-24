"""
Testes de hipótese básicos para compor o bloco de inferência:
- Testes de normalidade (Shapiro e D'Agostino)
- ANOVA de uma via com cálculo do tamanho de efeito (eta²)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict

def normality_tests(series: pd.Series) -> Dict[str, float]:
    """
    Aplica testes de normalidade em uma série numérica (com NaNs removidos):
    - Shapiro-Wilk (recomendado para amostras pequenas)
    - D'Agostino-Pearson (testa assimetria e curtose)

    Retorna um dicionário com p-valores. (Quanto maior, menos evidência contra normalidade)
    """
    s = series.dropna().astype(float)
    p_shapiro = stats.shapiro(s).pvalue if len(s) >= 3 else np.nan
    p_dagostino = stats.normaltest(s).pvalue if len(s) >= 8 else np.nan
    return {"p_shapiro": float(p_shapiro), "p_dagostino": float(p_dagostino)}

def anova_oneway(df: pd.DataFrame, value_col: str, group_col: str) -> Dict[str, float]:
    """
    ANOVA de uma via: compara médias do value_col entre níveis de group_col.
    Também retorna o tamanho de efeito eta² (proporção da variância explicada pelo fator).

    Observação: eta² aproximado por:
      eta² = ((k-1)*F) / (((k-1)*F) + (N - k))
      onde k = nº de grupos, N = total de observações utilizadas.
    """
    # Cria uma lista de grupos, cada um com os valores de value_col
    groups = [g.dropna().values for _, g in df.groupby(group_col)[value_col]]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        # Não há grupos suficientes para testar
        return {"F": np.nan, "p_value": np.nan, "eta_sq": np.nan}

    # Estatística F e p-valor da ANOVA
    F, p = stats.f_oneway(*groups)

    # Cálculo de eta² (tamanho de efeito)
    k = len(groups)
    N = sum(len(g) for g in groups)
    eta_sq = ((k-1)*F) / (((k-1)*F) + (N - k))

    return {"F": float(F), "p_value": float(p), "eta_sq": float(eta_sq)}

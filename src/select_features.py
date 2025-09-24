"""
Seleção de features com foco em rapidez e alinhamento ao objetivo de correlação:

Fluxo:
1) Cria uma base de candidatas por grupos (custos/manejo/solo/localização) filtrando por completude >= 70%.
2) Divide em numéricas e categóricas.
3) Ordena numéricas por |correlação| com o alvo.
4) Ordena categóricas por tamanho de efeito aproximado (eta²) via ANOVA.
5) Combina os rankings e pega as TOP_MI primeiras.

Observação:
- A MI completa (mutual_information_ranking) é rodada depois apenas nessas selecionadas,
  no `run_all.py`, para gerar um gráfico/tabela adicional com bom custo-benefício.
"""

import numpy as np
import pandas as pd
from typing import List
from scipy import stats
from .config import (
    MIN_COMPLETENESS, TARGET, COST_COLS, MGMT_COLS, SOIL_COLS, LOCATION_COLS,
    TOP_MI, MAX_CAT_LEVELS
)

def select_by_completeness(df: pd.DataFrame, cols: List[str], threshold: float = MIN_COMPLETENESS) -> List[str]:
    """
    Retorna a lista de colunas de 'cols' cuja fração de não-nulos >= threshold,
    excluindo o TARGET caso estivesse na lista.
    """
    ok = []
    for c in cols:
        if c in df.columns:
            frac = 1 - df[c].isna().mean()
            if frac >= threshold and c != TARGET:
                ok.append(c)
    return ok

def rank_numeric_by_corr(df: pd.DataFrame, num_cols: List[str]) -> List[str]:
    """
    Retorna num_cols ordenadas por |correlação de Pearson| com o TARGET (desc).
    """
    if TARGET not in df.columns or len(num_cols) == 0:
        return []
    num = df[num_cols].select_dtypes(include=[np.number])
    cors = num.corrwith(df[TARGET]).dropna().abs().sort_values(ascending=False)
    return list(cors.index)

def rank_cats_by_anova_eta(df: pd.DataFrame, cat_cols: List[str]) -> List[str]:
    """
    Retorna cat_cols ordenadas pelo tamanho de efeito (eta²) aproximado via ANOVA de uma via.
    Aplicamos filtros:
      - Completude >= MIN_COMPLETENESS
      - Número de níveis entre 2 e MAX_CAT_LEVELS
    """
    ranks = []
    for c in cat_cols:
        if df[c].notna().mean() < MIN_COMPLETENESS:
            continue
        nun = df[c].nunique(dropna=True)
        if nun < 2 or nun > MAX_CAT_LEVELS:
            continue
        sub = df[[c, TARGET]].dropna()
        groups = [g.values for _, g in sub.groupby(c)[TARGET]]
        if len(groups) < 2:
            continue
        # ANOVA simples
        F, p = stats.f_oneway(*groups)
        k = len(groups); N = sum(len(g) for g in groups)
        eta_sq = ((k-1)*F) / (((k-1)*F) + (N - k))
        ranks.append((c, float(eta_sq)))
    ranks = sorted(ranks, key=lambda t: t[1], reverse=True)
    return [c for c,_ in ranks]

def propose_feature_set(df: pd.DataFrame) -> List[str]:
    """
    Cria a lista final de features:
      - Parte das listas por grupo (custos/manejo/solo/localização) filtradas por completude
      - Separa numéricas e categóricas
      - Ranqueia por correlação/eta²
      - Retorna as TOP_MI primeiras do combinado
    """
    # Junta todas as candidatas por grupo, mantendo a ordem sem duplicatas
    base = []
    for grp in [COST_COLS, MGMT_COLS, SOIL_COLS, LOCATION_COLS]:
        base += select_by_completeness(df, grp)
    base = list(dict.fromkeys(base))

    # Fallback: se a base ficou vazia, usa todas as numéricas (exceto TARGET)
    if len(base) == 0:
        base = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET in base:
            base.remove(TARGET)

    # Divide por tipo
    num_cols = df[base].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[base].select_dtypes(include=["object"]).columns.tolist()

    # Ranqueamentos
    num_rank = rank_numeric_by_corr(df, num_cols)
    cat_rank = rank_cats_by_anova_eta(df, cat_cols)

    # Combina priorizando as melhores numéricas, depois categóricas
    combined = num_rank + cat_rank
    final = combined[:TOP_MI]
    return final

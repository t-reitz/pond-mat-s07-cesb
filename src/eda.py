"""
Módulo de EDA (Análise Exploratória de Dados) com foco em saídas
que vão direto para slides/documentação.

Conteúdo:
- Tabelas descritivas (numéricas e frequências categóricas)
- Mapa de calor da correlação
- Top correlações com o alvo (tabela + gráfico de barras)
- Gráficos de dispersão (alvo vs. top features numéricas)
- Boxplots do alvo por categóricas (limitadas em número de níveis)

Observações:
- Não usamos seaborn por exigência (apenas matplotlib).
- Cada figura é salva isoladamente (1 gráfico por figura).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
from .config import (
    OUTPUT_TABLES, OUTPUT_FIGURES, TARGET,
    TOP_CORR_WITH_TARGET, N_SCATTER_TOP, N_BOX_TOP_CAT, MAX_CAT_LEVELS
)

def ensure_dirs():
    """Garante que pastas de saída existam."""
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

def descriptive_tables(df: pd.DataFrame) -> dict[str, Path]:
    """
    Gera e salva tabelas descritivas para:
    - Variáveis numéricas: estatísticas de .describe() + contagem de missing
    - Variáveis categóricas: top 30 categorias por coluna
    - Missingness: fração de ausentes por coluna (ordenada)
    Retorna um dicionário com caminhos dos arquivos salvos.
    """
    ensure_dirs()
    out = {}

    # Estatísticas de colunas numéricas
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        stats = num.describe().T               # estatísticas por coluna (transposto)
        stats["missing"] = num.isna().sum()    # acrescenta nº de ausentes
        stats.to_csv(OUTPUT_TABLES / "desc_numericas.csv", index=True)
        out["desc_numericas"] = OUTPUT_TABLES / "desc_numericas.csv"

    # Frequências das colunas categóricas (top 30 por coluna)
    cat = df.select_dtypes(include=["object"])
    if not cat.empty:
        topcats = {}
        for c in cat.columns:
            vc = cat[c].value_counts(dropna=False).head(30)  # inclui NaN
            topcats[c] = vc
        cat_df = pd.concat(topcats, axis=1)  # concatena em colunas
        cat_df.to_csv(OUTPUT_TABLES / "freq_categoricas_top30.csv")
        out["freq_categoricas_top30"] = OUTPUT_TABLES / "freq_categoricas_top30.csv"

    # Fração de ausentes por coluna
    miss = df.isna().mean().sort_values(ascending=False).rename("frac_missing")
    miss.to_csv(OUTPUT_TABLES / "missingness.csv", header=True)
    out["missingness"] = OUTPUT_TABLES / "missingness.csv"

    return out

def correlation_matrix(df: pd.DataFrame) -> Optional[Path]:
    """
    Calcula e salva:
    - Matriz de correlação entre variáveis numéricas (CSV)
    - Heatmap simples da correlação (PNG)
    Retorna o caminho da figura.
    """
    ensure_dirs()
    # Seleciona apenas colunas numéricas não completamente vazias
    num = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num.shape[1] < 2:   # se tiver menos que 2 colunas, não faz sentido correlacionar
        return None

    corr = num.corr()  # matriz de correlação de Pearson
    corr.to_csv(OUTPUT_TABLES / "correlation_matrix.csv")

    # Heatmap simples (matplotlib)
    fig, ax = plt.subplots(figsize=(12,10))
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(cax)
    ax.set_title("Correlação (variáveis numéricas)")
    fig.tight_layout()
    outpath = OUTPUT_FIGURES / "corr_heatmap.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def top_correlations_with_target(df: pd.DataFrame, k: int = TOP_CORR_WITH_TARGET) -> pd.DataFrame:
    """
    Computa correlação de Pearson de cada variável numérica com o alvo.
    Salva:
    - CSV com as top k correlações (em valor absoluto)
    - Figura em barras horizontais com essas top k.
    Retorna o DataFrame completo (ordenado por |correlação|).
    """
    num = df.select_dtypes(include=[np.number]).copy()
    if TARGET not in num.columns:
        # Se o alvo não for numérico ou não existir, retorna vazio
        return pd.DataFrame(columns=["feature","corr"])

    y = num[TARGET]
    X = num.drop(columns=[TARGET])
    # Série de correlações, ordenada por valor absoluto (desc)
    cors = X.corrwith(y).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    res = pd.DataFrame({"feature": cors.index, "corr": cors.values})

    # Salva top k em CSV
    res.head(k).to_csv(OUTPUT_TABLES / "top_corr_with_target.csv", index=False)

    # Barra horizontal para visual (somente top k)
    fig, ax = plt.subplots(figsize=(8,5))
    top = res.head(k)
    ax.barh(top["feature"][::-1], top["corr"][::-1])  # invertido para maior ficar em cima
    ax.set_title(f"Top {k} correlações com {TARGET}")
    ax.set_xlabel("Correlação de Pearson")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "top_corr_with_target_bar.png", dpi=200)
    plt.close(fig)

    return res

def scatter_plots_y_vs_topk(df: pd.DataFrame, top_corr_df: pd.DataFrame, n: int = N_SCATTER_TOP):
    """
    Gera gráficos de dispersão (alvo vs. variável) para as n variáveis
    com maior |correlação| com o alvo. Inclui linha de tendência linear.
    Salva cada figura separadamente e retorna a lista de caminhos.
    """
    num = df.select_dtypes(include=[np.number]).copy()
    if TARGET not in num.columns or top_corr_df.empty:
        return []

    outs = []
    # Para cada feature selecionada (até n)
    for feat in top_corr_df["feature"].head(n):
        if feat not in num.columns:
            continue

        # Extrai vetores e remove NaN
        x = num[feat].values
        y = num[TARGET].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]; y = y[mask]
        if len(x) < 5:
            continue  # não plota se poucos pontos

        # Ajusta linha de tendência (regressão linear simples)
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)

        # Plota dispersão + linha
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(x, y, alpha=0.4)
        xx = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(xx, poly1d_fn(xx), linestyle="--")
        ax.set_xlabel(feat)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{TARGET} vs {feat}")

        # Salva
        outpath = OUTPUT_FIGURES / f"scatter_{TARGET}_vs_{feat}.png"
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        outs.append(outpath)
    return outs

def boxplots_y_by_cats(df: pd.DataFrame, n_cats: int = N_BOX_TOP_CAT, max_levels: int = MAX_CAT_LEVELS):
    """
    Gera boxplots do alvo por variáveis categóricas "boas" para visualização.
    Critérios:
      - Completude >= 70%
      - Nº de níveis entre 2 e max_levels (para não poluir)
    Retorna lista de caminhos das figuras salvas.
    """
    outs = []
    if TARGET not in df.columns:
        return outs

    # Lista de colunas categóricas
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Pontua e ordena as categóricas por nº de níveis (menor -> melhor) e completude (maior -> melhor)
    scored = []
    for c in cat_cols:
        if df[c].notna().mean() < 0.7:  # descartamos muito faltantes
            continue
        nun = df[c].nunique(dropna=True)
        if nun < 2 or nun > max_levels:
            continue
        scored.append((c, nun, df[c].notna().mean()))
    scored = sorted(scored, key=lambda t: (t[1], -t[2]))

    # Gera figuras para as top selecionadas
    for c, nun, comp in scored[:n_cats]:
        sub = df[[c, TARGET]].dropna()
        groups = []
        labels = []
        for lvl, grp in sub.groupby(c):
            groups.append(grp[TARGET].values)
            labels.append(str(lvl))
        if len(groups) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8,4))
        ax.boxplot(groups, labels=labels, vert=True, showmeans=True)
        ax.set_title(f"{TARGET} por {c} (n níveis={nun})")
        ax.set_ylabel(TARGET)
        fig.tight_layout()
        outpath = OUTPUT_FIGURES / f"box_{TARGET}_by_{c}.png"
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        outs.append(outpath)

    return outs

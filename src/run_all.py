"""
Roteiro principal (orquestrador) para executar a cadeia ponta-a-ponta:

1) Carregar dados
2) EDA rica:
   - tabelas descritivas (num/cat/missing)
   - matriz e heatmap de correlação
   - top correlações com o alvo + gráficos de dispersão
   - boxplots do alvo por categóricas "boas"
3) Seleção de features rápida (correlação + ANOVA) -> conjunto enxuto
4) MI apenas nas selecionadas (tabela + gráfico — custo/benefício)
5) Modelagem (K-Fold) + métricas (R², RMSE, MAE)
6) Figuras: Paridade + Top |coef| (interpretação)
7) Sumário em Markdown para slides/relatório

Como rodar:
  python -m src.run_all --data data/raw/dados.xlsx --sheet GERAL
"""

import argparse
from pathlib import Path
import pandas as pd

from .config import TARGET, DEFAULT_DATA_XLSX, DEFAULT_SHEET, REPORTS
from .load_data import load_data
from .stats_extra import corr_with_target_and_pvalue
from .eda import (
    descriptive_tables, correlation_matrix, top_correlations_with_target,
    scatter_plots_y_vs_topk, boxplots_y_by_cats
)
from .select_features import propose_feature_set
from .info_theory import mutual_information_ranking
from .model import evaluate_models, plot_parity, fit_and_plot_coefs

def main():
    # === Argumentos de linha de comando (opcionais) ===
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA_XLSX), help="Caminho do Excel de dados")
    ap.add_argument("--sheet", type=str, default=DEFAULT_SHEET, help="Nome da aba do Excel")
    args = ap.parse_args()

    # === 1) Carregar dados ===
    df = load_data(args.data, args.sheet)

    # === 2) EDA rica ===
    descriptive_tables(df)                          # tabelas num/cat/missing
    correlation_matrix(df)                          # heatmap + csv de correlação
    top_corr_df = top_correlations_with_target(df)  # top correlações com alvo
    scatter_plots_y_vs_topk(df, top_corr_df)        # dispersões y vs x (top)
    boxplots_y_by_cats(df)                          # boxplots y por categóricas

    # === 3) Seleção rápida de features ===
    features = propose_feature_set(df)              # lista final enxuta

    # === 4) MI apenas nas selecionadas (ótimo p/ ranking + slide) ===
    mi_agg = mutual_information_ranking(df, features)
    (Path("output/tables") / "selected_features_mi_agg.csv").parent.mkdir(parents=True, exist_ok=True)
    mi_agg.head(20).to_csv("output/tables/selected_features_mi_agg.csv", index=False)

    # === 5) Modelagem com K-Fold ===
    metrics = evaluate_models(df, features, TARGET)
    best = metrics.iloc[0]["model"]                  # melhor pelo R² médio

    # === 6) Figuras de modelo (paridade + coeficientes) ===
    fig_par = plot_parity(df, features, TARGET, model_name=best)
    fig_coef = fit_and_plot_coefs(df, features, TARGET, model_name=best)

    corr_with_target_and_pvalue(df)

    # === 7) Sumário em Markdown ===
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = []
    md.append(f"# Sumário — Pipeline Preditivo (Target: {TARGET})\n")
    md.append("## Features selecionadas (ordem por MI nas selecionadas)\n")
    md.append("\n".join([f"- {c}" for c in mi_agg['feature_base'].head(20)]))
    md.append("\n\n## Métricas (CV)\n")
    md.append(metrics.to_markdown(index=False))
    md.append("\n\n## Figuras-chave\n")
    md.append(f"- Paridade: {Path(fig_par).as_posix() if fig_par else 'N/A'}")
    md.append(f"- Top |coef|: {Path(fig_coef).as_posix() if fig_coef else 'N/A'}")
    (REPORTS / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # Prints úteis no terminal (log rápido)
    print("Features:", features)
    print("Melhor modelo:", best)
    print("Resumo:", (REPORTS / "summary.md").as_posix())

if __name__ == "__main__":
    main()

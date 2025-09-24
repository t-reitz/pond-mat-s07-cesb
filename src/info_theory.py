"""
Cálculo de Informação Mútua (MI) entre X e y.

Passos:
- Separa numéricas/categóricas
- Imputa faltantes (mediana para num, mais frequente para cat)
- One-Hot Encoder para categóricas (sparse=False para obter matriz densa)
- Calcula MI nas features expandidas (dummies)
- Agrega a MI de volta para o nome da coluna original (soma das dummies)
- Salva tabelas e um gráfico de barras com as top features por MI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import TARGET, OUTPUT_TABLES, OUTPUT_FIGURES, TOP_MI

def mutual_information_ranking(df: pd.DataFrame, features: list[str], target: str = TARGET) -> pd.DataFrame:
    """
    Calcula MI no espaço de features expandidas (dummies) e agrega por coluna original.

    Parâmetros:
      - df: DataFrame original
      - features: lista de nomes de colunas candidatas de X
      - target: nome da coluna alvo y

    Retorno:
      - DataFrame agregado por "feature_base" (coluna original), ordenado decrescente por MI.
    """
    # Extrai X e y
    X = df[features].copy()
    y = df[target].values

    # Identifica tipos
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Pré-processador:
    # - Categóricas: imputação por mais frequente + OneHot (matriz densa)
    # - Numéricas: imputação por mediana
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), num_cols)
    ])

    # Aplica o pré-processamento
    pipe = Pipeline([("pre", pre)])
    X_enc = pipe.fit_transform(X)  # matriz 2D sem faltantes, com dummies expandidas

    # Calcula MI entre cada coluna de X_enc e y
    mi = mutual_info_regression(X_enc, y, random_state=42)

    # Recupera os nomes expandidos das variáveis categóricas (padrão: col_valor)
    enc = pipe["pre"].named_transformers_["cat"] if len(cat_cols) > 0 else None
    cat_feature_names = enc["ohe"].get_feature_names_out(cat_cols).tolist() if enc is not None and len(cat_cols)>0 else []
    feature_names = cat_feature_names + num_cols

    # Monta DataFrame com MI por "feature expandida"
    res = pd.DataFrame({"feature_expanded": feature_names, "mi": mi})
    res = res.sort_values("mi", ascending=False)
    (OUTPUT_TABLES / "mutual_information_expanded.csv").parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT_TABLES / "mutual_information_expanded.csv", index=False)

    # Função auxiliar para resgatar o nome da coluna original antes do "_"
    def base_name(s: str) -> str:
        # Se possuir "_" e a parte antes do "_" for uma categoria conhecida, assume ser a base
        if "_" in s and s.split("_", 1)[0] in cat_cols:
            return s.split("_", 1)[0]
        return s  # para numéricas, mantém o nome original

    # Coluna com o nome base e agregação (soma das dummies por coluna original)
    res["feature_base"] = [base_name(s) for s in res["feature_expanded"]]
    agg = res.groupby("feature_base", as_index=False)["mi"].sum().sort_values("mi", ascending=False)
    agg.to_csv(OUTPUT_TABLES / "mutual_information_by_feature.csv", index=False)

    # Gráfico: top TOP_MI por MI (coluna base)
    top = agg.head(TOP_MI)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(top["feature_base"][::-1], top["mi"][::-1])
    ax.set_title(f"Top {len(top)} por Informação Mútua (coluna original)")
    ax.set_xlabel("MI (nats)")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURES / "mi_top_features_bar.png", dpi=200)
    plt.close(fig)

    # Salva também o CSV das top
    top.to_csv(OUTPUT_TABLES / "mutual_information_top.csv", index=False)

    return agg

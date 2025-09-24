# -*- coding: utf-8 -*-
"""
Modelagem e avaliação (validação cruzada) com pipeline scikit-learn:

- Pré-processamento:
    * Categóricas: imputação (mais frequente) + OneHotEncoder(sparse=False)
    * Numéricas: imputação (mediana) + padronização (StandardScaler)
- Modelos:
    * FAST_MODE=True: LinearRegression apenas (rápido, estável)
    * FAST_MODE=False: adiciona RidgeCV e LassoCV para comparar regularizações
- Métricas:
    * R², RMSE, MAE (média e desvio na validação K-fold)
- Gráficos:
    * Paridade (y observado vs. y previsto no hold-out)
    * Top |coef| do modelo para interpretação (Linear/Ridge/Lasso)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from .config import OUTPUT_TABLES, OUTPUT_FIGURES, RANDOM_STATE, N_SPLITS, TARGET

# Modo rápido (apenas Regressão Linear). Coloque False para incluir Ridge/Lasso.
FAST_MODE = True

def _metrics():
    """
    Define o dicionário de métricas usadas no cross_validate.
    - r2
    - rmse
    - mae
    """
    return {
        "r2": make_scorer(r2_score),
        "rmse": make_scorer(lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))),
        "mae": make_scorer(mean_absolute_error)
    }

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói o pré-processador que:
      - Imputa e codifica as categóricas (OneHot dense)
      - Imputa e escala as numéricas
    Retorna um ColumnTransformer que será usado dentro do Pipeline.
    """
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    pre = ColumnTransformer([
        # Categóricas: imputação por mais frequente + One-Hot Encoder (saída densa)
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        # Numéricas: imputação por mediana + padronização
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ])
    return pre

def evaluate_models(df: pd.DataFrame, features: List[str], target: str = TARGET) -> pd.DataFrame:
    """
    Roda validação cruzada K-fold para os modelos definidos e salva as métricas.
    Retorna um DataFrame ordenado pelo R² médio (desc).
    """
    X = df[features].copy()   # subconjunto de X com as features selecionadas
    y = df[target].values     # vetor alvo

    pre = build_preprocessor(X)  # pré-processamento

    # Define o(s) estimador(es) conforme o modo
    if FAST_MODE:
        models = { "LinearRegression": LinearRegression() }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeCV": RidgeCV(alphas=np.logspace(-2, 2, 20), cv=5),
            "LassoCV": LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, random_state=RANDOM_STATE, max_iter=20000)
        }

    # Validação cruzada com shuffle e seed fixa
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Para cada modelo: pipeline (pre + modelo) -> cross_validate -> coletar métricas
    rows = []
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        cvres = cross_validate(pipe, X, y, cv=kf, scoring=_metrics(), return_estimator=False)
        rows.append({
            "model": name,
            "r2_mean": float(np.mean(cvres["test_r2"])),
            "r2_std": float(np.std(cvres["test_r2"])),
            "rmse_mean": float(np.mean(cvres["test_rmse"])),
            "rmse_std": float(np.std(cvres["test_rmse"])),
            "mae_mean": float(np.mean(cvres["test_mae"])),
            "mae_std": float(np.std(cvres["test_mae"])),
        })

    # Monta DataFrame de métricas e salva
    res = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT_TABLES / "model_cv_metrics.csv", index=False)
    return res

def plot_parity(df: pd.DataFrame, features: List[str], target: str = TARGET, model_name: str = "LinearRegression"):
    """
    Gera gráfico de paridade (y observado vs y previsto) com um hold-out de 20%.
    Útil para visual de generalização e tendência.
    Retorna o caminho do PNG salvo.
    """
    # Prepara dados e pré-processador
    X = df[features].copy()
    y = df[target].values
    pre = build_preprocessor(X)

    # Escolha do estimador (aqui mantemos LinearRegression)
    est = LinearRegression()

    # Split treino/teste (20% teste)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Monta pipeline e ajusta
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(Xtr, ytr)

    # Prediz no teste
    yhat = pipe.predict(Xte)

    # Figura de paridade
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(yte, yhat, alpha=0.6)     # pontos observados vs previstos
    mn = min(np.min(yte), np.min(yhat))  # mínimo para a linha 45°
    mx = max(np.max(yte), np.max(yhat))  # máximo para a linha 45°
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Observado")
    ax.set_ylabel("Previsto")
    ax.set_title(f"Paridade: {model_name}")

    # Salva
    outpath = OUTPUT_FIGURES / f"paridade_{model_name}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def fit_and_plot_coefs(
    df: pd.DataFrame,
    features: List[str],
    target: str = TARGET,
    model_name: str = "LinearRegression",
    topn: int = 20
):
    """
    Ajusta o modelo no dataset completo (apenas para interpretação),
    extrai coeficientes (em espaço padronizado/codificado) e plota as top |coef|.

    Retorna o caminho do PNG salvo (ou None se o modelo não tiver coef_).
    """
    # X e y
    X = df[features].copy()
    y = df[target].values

    # Identifica colunas categóricas e numéricas (para resgatar nomes)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Pré-processador + modelo
    pre = build_preprocessor(X)
    est = LinearRegression()
    pipe = Pipeline([("pre", pre), ("model", est)])
    pipe.fit(X, y)  # ajusta no dataset inteiro para obter coeficientes

    # Recupera nomes das colunas expandidas (dummies para categóricas)
    enc = pipe.named_steps["pre"]
    cat_names = []
    if len(cat_cols) > 0:
        ohe = enc.named_transformers_["cat"].named_steps["ohe"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    feat_names = cat_names + num_cols  # ordem: primeiro dummies, depois numéricas

    # Extrai coeficientes do modelo (array 1D)
    model = pipe.named_steps["model"]
    coefs = getattr(model, "coef_", None)
    if coefs is None:
        # Modelos que não expõem coef_ (ex.: árvores) cairiam aqui
        return None

    # Monta DataFrame de coeficientes absolutos (para rankeamento)
    dfc = pd.DataFrame({"feature": feat_names, "coef": coefs.ravel()})
    dfc["abs_coef"] = dfc["coef"].abs()
    dfc = dfc.sort_values("abs_coef", ascending=False)

    # Salva a tabela completa de coeficientes
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    dfc.to_csv(OUTPUT_TABLES / f"coefficients_{model_name}.csv", index=False)

    # Gráfico com top |coef|
    top = dfc.head(topn)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(top["feature"][::-1], top["abs_coef"][::-1])
    ax.set_title(f"Top {topn} |coef| — {model_name}")
    ax.set_xlabel("|coeficiente| (padronizadas)")
    fig.tight_layout()
    outpath = OUTPUT_FIGURES / f"coef_top_abs_{model_name}.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    return outpath

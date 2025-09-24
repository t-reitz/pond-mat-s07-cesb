"""
Configurações centrais do projeto de Modelagem Multivariada (Atividade).
Foco nas entregas do Thomas: pipeline preditivo, validação, reprodutibilidade e correlações.

Este arquivo concentra:
- Caminhos padrão das pastas de dados/saídas/relatórios
- Nome do arquivo e planilha padrão do Excel
- Definição do alvo (target) do problema
- Parâmetros para EDA e seleção de features
- Parâmetros de validação (seed e K-fold)
- Listas "candidatas" de colunas por grupo (custos/manejo/solo/localização),
  que ajudam a selecionar variáveis relevantes com boa completude.
"""

from pathlib import Path

# === Caminhos principais do projeto ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]     # raiz = pasta acima de src
DATA_RAW = PROJECT_ROOT / "data" / "raw"               # dados brutos
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"   # dados processados (se usado)
OUTPUT_FIGURES = PROJECT_ROOT / "output" / "figures"   # figuras geradas para slides
OUTPUT_TABLES = PROJECT_ROOT / "output" / "tables"     # tabelas/CSVs para documentação
REPORTS = PROJECT_ROOT / "reports"                      # relatórios em Markdown

# === Fonte de dados padrão ===
DEFAULT_DATA_XLSX = DATA_RAW / "dados.xlsx"             # caminho do Excel esperado
DEFAULT_SHEET = "GERAL"                                 # nome da aba com os dados

# === Variável alvo (target) ===
# Escolhida pela boa completude e aderência ao objetivo: produtividade final (kg/ha)
TARGET = "Produtividade_Final_kg_ha"

# === Parâmetros da EDA e de seleção ===
MIN_COMPLETENESS = 0.70  # fração mínima de não-nulos para uma feature ser considerada (>= 70%)
TOP_MI = 12              # quantidade de variáveis no "top" por Informação Mútua (aplicado depois)
TOP_CORR_WITH_TARGET = 12  # quantidade de variáveis com maior |correlação| com o alvo para mostrar
N_SCATTER_TOP = 6          # número de gráficos de dispersão (alvo vs. feature) a salvar
N_BOX_TOP_CAT = 5          # número de boxplots alvo vs. categóricas a salvar
MAX_CAT_LEVELS = 8         # máximo de níveis por categórica para não poluir a visualização

# === Parâmetros de validação ===
RANDOM_STATE = 42  # semente para reprodutibilidade
N_SPLITS = 5       # número de dobras (K) da validação cruzada

# === Grupos de features candidatos ===
# Observação: ajuste os nomes conforme a presença no seu dataset.
# Estas listas ajudam a guiar a seleção automática filtrando por completude.
COST_COLS = [
    "Custos_Area_Audit",         # custo por área auditada (exemplo)
    "Custo_Mao_de_Obra",         # custo com mão de obra
    "Custo_Mecanizacao",         # custo com mecanização
    "Custo_Adubacao_Corretivos", # custo com adubação/corretivos
    "Custo_Defensivos",          # custo com defensivos
    "COLHEITA_CUSTO"             # custo de colheita
]

MGMT_COLS = [
    "Ciclo_Maturacao_Cultivar",         # ciclo/maturação da cultivar
    "Populacao_Plantas_Calculada_ha",   # população de plantas (ha)
    "Perda_Colheita_sc_ha",             # perda na colheita (sc/ha)
    "Preparo_Colheita",                 # preparação/ajuste no processo de colheita
    "Produto_Utilizado_Dessecacao"      # manejo de dessecação (exemplo)
]

SOIL_COLS = [
    # Coloque aqui nomes que existirem no seu df (exemplos usuais, comentei para não quebrar):
    # "Argila_%", "pH_Solo", "CTC", "MO_%", "P_mehlich_mg_dm3", ...
]

LOCATION_COLS = [
    "UF_Propriedade",       # estado (UF)
    "Cidade_Propriedade",   # município/cidade
    "Regiao_Propriedade",   # região
    "Latitude", "Longitude" # coordenadas
]

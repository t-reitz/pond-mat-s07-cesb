# Sumário — Pipeline Preditivo (Target: Produtividade_Final_kg_ha)

## Features selecionadas (ordem por MI nas selecionadas)

- Preparo_Colheita_******
- Preparo_Colheita_***
- Perda_Colheita_sc_ha
- Regiao_Propriedade_CENTRO-OESTE
- Ciclo_Maturacao_Cultivar_MEDIO
- Custo_Mao_de_Obra
- Regiao_Propriedade_SUL
- Custo_Mecanizacao
- Regiao_Propriedade_NORDESTE
- Longitude
- Ciclo_Maturacao_Cultivar_TARDIO
- Latitude
- Regiao_Propriedade_NORTE
- Custo_Defensivos
- Populacao_Plantas_Calculada_ha
- Custo_Adubacao_Corretivos
- Ciclo_Maturacao_Cultivar_PRECOCE
- Ciclo_Maturacao_Cultivar_SUPER_PRECOCE
- Ciclo_Maturacao_Cultivar_SUPER_TARDIO
- Regiao_Propriedade_SUDESTE


## Métricas (CV)

| model            |   r2_mean |    r2_std |   rmse_mean |   rmse_std |   mae_mean |   mae_std |
|:-----------------|----------:|----------:|------------:|-----------:|-----------:|----------:|
| LinearRegression | 0.0610444 | 0.0231425 |     890.212 |    25.2775 |    654.721 |   17.1932 |


## Figuras-chave

- Paridade: /Users/tr/Desktop/Github Projects/pond-mat-s07-cesb/output/figures/paridade_LinearRegression.png
- Top |coef|: /Users/tr/Desktop/Github Projects/pond-mat-s07-cesb/output/figures/coef_top_abs_LinearRegression.png
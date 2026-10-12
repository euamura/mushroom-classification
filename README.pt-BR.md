# Classificação de Cogumelos: Comestível vs. Venenoso

🇺🇸 [Read in English](README.md)

Comparação entre **K-Nearest Neighbors**, **Árvore de Decisão** e **Naive Bayes** na base Mushroom da UCI, avaliados com uma **métrica orientada à segurança**: o objetivo principal é nunca classificar um cogumelo venenoso como comestível.

> ⚠️ Este é um estudo de machine learning sobre uma base histórica e hipotética. **Não use para decidir se um cogumelo real pode ser consumido.**

## Resultados em resumo

| Posição | Modelo | Recall (poisonous), hold-out | Recall (poisonous), CV 5 folds | Acurácia, CV 5 folds | Venenosos não detectados (hold-out) |
|---|---|---|---|---|---|
| 🥇 | **KNN (k=5)** | **100,00%** | **100,00% ± 0,00** | **100,00%** | **0** |
| 🥈 | Árvore de Decisão (max_depth=4) | 99,74% | 99,90% ± 0,10 | 99,95% | 3 |
| 🥉 | Naive Bayes (α=1) | 92,34% | 92,49% ± 1,31 | 96,09% | 90 |

**Recomendação:** KNN, que identificou todos os cogumelos venenosos em todas as rodadas de validação. A árvore de decisão é a alternativa interpretável: sua primeira pergunta é sobre o `odor` (cheiro), o mesmo atributo da regra mais conhecida para esta base.

![Árvore de decisão](reports/figures/decision_tree.png)

## Definição do problema

| Erro | Consequência |
|---|---|
| Comestível → previsto como venenoso | Um cogumelo bom é descartado. Custo baixo. |
| **Venenoso → previsto como comestível** | **Risco à saúde. Inaceitável.** |

Por isso, os modelos são ranqueados pelo **recall da classe `poisonous`**, com a **acurácia** como desempate. As classes estão equilibradas (≈52/48), então a acurácia continua sendo um indicador secundário confiável.

## Base de dados

- **Fonte:** [UCI Machine Learning Repository: Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) (Schlimmer, 1987), baseada no *The Audubon Society Field Guide to North American Mushrooms* (1981).
- **Tamanho:** 8.124 amostras, 22 atributos categóricos e alvo binário.
- **Formato:** CSV sem cabeçalho, com cada valor codificado por uma letra. A legenda está em `data/agaricus-lepiota.names`.

## Decisões e justificativas

| # | Etapa | Decisão | Justificativa |
|---|---|---|---|
| 1 | Nomes das colunas | Mantidos em inglês | Seguem a documentação oficial, o que facilita a conferência. |
| 2 | Valores | Letras → palavras legíveis em inglês | Facilita inspeção e interpretação. Códigos não mapeados viram `NaN`, o que serve como verificação de completude. |
| 3 | `stalk_root` | **Coluna removida** | 2.480 valores faltantes (≈30%). Remover a coluna mantém as 8.124 linhas; remover as linhas descartaria um terço dos dados. |
| 4 | `veil_type` | **Removida** | Um único valor em todas as linhas, sem poder de distinção. |
| 5 | Divisão | **70/30, estratificada**, `random_state=42` | Conjunto de teste grande e confiável, proporção das classes preservada e resultado reproduzível. |
| 6 | Codificação | **One-Hot** para KNN e árvore, **Ordinal** para Naive Bayes | Códigos ordinais criariam uma ordem falsa para o KNN, que usa distâncias. One-Hot gera regras legíveis na árvore. O `CategoricalNB` espera rótulos inteiros. Os codificadores aprendem apenas com o treino. |
| 7 | KNN | **k = 5** | Padrão do scikit-learn, usado como ponto de partida sem ajuste fino. k ímpar evita empate na votação. |
| 8 | Árvore de Decisão | **max_depth = 4** | Prioriza a interpretabilidade. Mantida após ver o resultado, para não ajustar o modelo com base no teste. Trade-off: 3 venenosos não detectados. |
| 9 | Naive Bayes | **`CategoricalNB`, α = 1,0** | Suavização de Laplace padrão, que impede que categorias ausentes em uma classe zerem a probabilidade. |
| 10 | Métrica | **Recall (poisonous)** primeiro, acurácia depois | Reflete o custo assimétrico dos erros. |
| 11 | Robustez | **Validação cruzada estratificada com 5 folds**, usando pipelines | Confirma que o resultado do hold-out não foi sorte da divisão. Os pipelines evitam vazamento de dados no encoder. |

## Principais aprendizados

- **`odor` é o atributo dominante.** A árvore o escolheu sozinha como primeira pergunta, em linha com a regra publicada "cheiro diferente de almond, anise ou none → venenoso" (≈98,5% de acurácia sozinha).
- **O Naive Bayes falha por causa de atributos correlacionados.** A base descreve 23 espécies, cada uma definida por uma *combinação* de características. A suposição de independência conta a mesma evidência várias vezes, e 90 venenosos passaram despercebidos. KNN e árvore exploram as combinações diretamente.
- **A interpretabilidade tem um custo mensurável.** Limitar a árvore à profundidade 4 custa entre 0,1 e 0,3 ponto percentual de recall.

## Limitações e próximos passos

- **Dados quase separáveis:** os resultados perfeitos refletem uma base quase determinística, não o desempenho em campo.
- **Amostras hipotéticas:** os registros vêm de descrições de um guia de campo, não de espécimes medidos, e espécies de "comestibilidade desconhecida" foram somadas à classe venenosa.
- **Sinal descartado:** os faltantes de `stalk_root` se concentram em venenosos (1.760 contra 720). Tratá-los como uma categoria `missing` poderia recuperar essa informação.
- **Sem ajuste de hiperparâmetros:** `k`, `max_depth` e `α` foram definidos por raciocínio. Um grid search com validação cruzada aninhada é o próximo passo natural.
- **Uso em produção:** uma análise de limiares de probabilidade orientada a custo tornaria o trade-off de segurança explícito.

## Estrutura do projeto

```
mushroom-classification/
├── data/
│   ├── agaricus-lepiota.data        # base original
│   └── agaricus-lepiota.names       # documentação oficial
├── notebooks/
│   └── mushroom_classification.ipynb
├── reports/figures/
│   └── decision_tree.png
├── requirements.txt
├── README.md
└── README.pt-BR.md
```

## Como executar

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook mushroom_classification.ipynb
```

## Tecnologias

Python · pandas · scikit-learn · matplotlib · Jupyter

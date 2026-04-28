# Passos

## Objetivos 28/04/2026

### De manhã
- Terminar a formatação dos csv's

### De tarde
- arrumar coluna de tipos e trasnformar em 18 novas colunas 
- Começar a implementar as técnicas de aprendizado
 - Escolher duas



# Pipeline do Trabalho

Pseudocódigo:

```Python

1. Repetir 20 vezes:

    1. separar o dataset em treino, teste e validação (usar train_test_split)

    2. Para cada ténica de aprendizagem

        1. Encontrar melhor modelo (Calibração de hiperparâmetros): 8 milhoes de for aninhados
        2. Pegar a acuracia do melhor modelo e salvar numa tabela de modelo x iteração
```

Exemplo de tabela de modelo x iteração de 20 repetições:

| Iteração | KNN | ... | CNN |
| :---: | :---: | :---: | :---: |
| 0 | 0.59 | ... | 0.78 |
| ... | ... | ... | ... |
| 19 | 0.63 | ... | 0.80 |
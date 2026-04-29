# Comparação de Técnicas de Aprendizado de Máquina - Batalhas Pokémon

Este repositório foi desenvolvido para realizar uma comparação entre diferentes técnicas de aprendizado de máquina, com o objetivo de prever o vencedor de batalhas entre dois Pokémon. O modelo analisa os atributos de ambos os combatentes para determinar qual deles sairá vitorioso.

## Objetivo
O projeto visa identificar qual técnica de ML melhor prediz o resultado de um combate.

- **Dataset utilizado:** [Pokemon Dataset with Team Combat (Kaggle)](https://www.kaggle.com/datasets/tuannguyenvananh/pokemon-dataset-with-team-combat/)

## Técnicas Utilizadas

As seguintes técnicas foram implementadas e analisadas neste projeto:
- a
- 
- 

## Pré-processamento dos Dados

Pré-processamento do dataset original antes do treinamento:

1.  **Limpeza de Atributos:** Remoção dos dados ruído `Name`, `#` e remoção de colunas irrelevantes para o combate, `Generation` e `Legendary`.
2.  **Codificação de Tipos (One-Hot Encoding):** Os tipos 1 e 2 de cada Pokémon são transformados em 18 colunas binárias, representando a presença ou ausência de cada tipo elemental.
3.  **Mapeamento de Combates:** Os IDs dos Pokémon no histórico de batalhas são substituídos pelos seus atributos reais.
4.  **Transformação do Alvo:** A coluna de vencedor é convertida em um valor binário (0 se o primeiro Pokémon venceu, 1 se o segundo venceu).

## Metodologia e Pipeline

O projeto segue um pipeline rigoroso para garantir a consistência estatística, que foi 

1.  **20 Iterações:** O fluxo principal executa o experimento 20 vezes.
2.  **Divisão de Dados:** Em cada iteração, os dados são divididos em Treino (50%), Teste (25%) e Validação (25%).
3.  **Otimização:** Realiza-se a busca pelos melhores hiperparâmetros para cada modelo.
4.  **Resultados:** A acurácia de cada modelo é registrada por iteração.

## Como Executar

### Pré-requisitos
Certifique-se de ter o Python 3.x instalado e as seguintes bibliotecas:
```bash
pip install pandas scikit-learn tqdm
```

### Passo a Passo
1.  **Preparar os Dados:** Primeiramente, é necessário executar o script de pré-processamento para gerar o dataset final a partir dos arquivos brutos em `datasets_brutos/`.
    ```bash
    python -m utils.pre_processamento
    ```
    *Isso gerará o arquivo `datasets_processados/dados.csv`.*

2.  **Executar o Experimento:** Com os dados prontos, execute o script principal para iniciar as 20 iterações de treinamento.
    ```bash
    python main.py
    ```

## Saída dos Resultados
Após a conclusão, os resultados consolidados serão salvos em `out/resultados.csv`. O arquivo seguirá o formato:
| iteração | Técnica 1 | Técnica 2 | ... | Técnica N |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.85 | 0.82 | ... | 0.88 |
| ... | ... | ... | ... | ... |
| 19 | 0.84 | 0.83 | ... | 0.87 |

## Estrutura do Projeto
- `datasets_brutos/`: CSVs originais do Kaggle.
- `datasets_processados/`: Dados limpos e prontos para o modelo.
- `metodos_aprendizado/`: Implementação dos algoritmos de ML.
- `utils/`: Scripts de pré-processamento e ferramentas auxiliares.
- `out/`: Pasta onde os resultados finais são armazenados.

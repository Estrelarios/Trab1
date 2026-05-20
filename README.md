# Pokémon Battle Classifier & Ensemble Framework

Este projeto implementa uma análise de classificadores para predição de resultados em batalhas Pokémon. O sistema tem código para realizar o pré-processamento do dataset, treinar os modelos e realizar os cálculos estatísticos para comparar o desempenho a partir dos resultados.

*Nota: Este projeto foi desenvolvido originalmente para a disciplina de Aprendizagem de Máquina do curso de Ciência da Computação (UNIOESTE - Cascavel, 2026).*

## Modelos analisados

O projeto avalia:

*   **Métodos Individuais (Monolíticos):**
    *   K-Nearest Neighbors (`KNeighborsClassifier`)
    *   Árvore de Decisão (`DecisionTreeClassifier`)
    *   Naive Bayes (`GaussianNB`)
    *   Support Vector Machines (`SVC`)
    *   Multi-Layer Perceptron (`MLPClassifier`)
*   **Sistemas de Múltiplos Classificadores (SMC):**
    *   Random Forest (`RandomForestClassifier`)
    *   Bagging (`BaggingClassifier`)
    *   Boosting (`AdaBoostClassifier`)
*   **Regras de Combinação (Implementação própria):**
    *   Regra da Soma
    *   Voto Majoritário
    *   Borda Count

## 🛠️ Funcionalidades Principais

*   **Pipeline de Dados Automatizado:** O sistema realiza o merge automático entre os atributos dos Pokémon (pokemon.csv) e o histórico de batalhas (combats.csv), transformando no dataset que será utilizado no treinamento.
*   **Pré-processamento:** Tratamento de tipos via codificação One-Hot, binarização da variável alvo (vencedor) e remoção de atributos não preditivos (IDS e Nomes de pokemon).
*   **Split de dados:** Divisão estratificada em Treino (50%), Validação (25%) e Teste (25%) conforme padrões acadêmicos, garantindo conjuntos mutuamente exclusivos.
*   **Busca de Hiperparâmetros:** Otimização automatizada para os modelos KNN, Árvore de Decisão, SVM, MLP, Random Forest, Bagging e Boosting. 
    *   *Nota 2: Foram implementadas buscas customizadas (ao invés de usar funções prontas como GridSearchCV) por requisito obrigatório do trabalho para o qual esse projeto foi feito.*
*   **Execução em Paralelo:** Orquestrador capaz de disparar múltiplas iterações do pipeline simultaneamente, otimizando o tempo total de execução das 20 repetições.
*   **Avaliação Estatística:** Suporte a cálculos de média, desvio padrão e aplicação de testes não-paramétricos (Friedman e Nemenyi) com 95% de confiança para validação científica dos resultados.

## 📂 Estrutura do Projeto

Abaixo estão descritos os diretórios e arquivos mais relevantes para a avaliação:

*   **`src/`**: Código-fonte principal do projeto.
    *   **`main.py`**: Orquestrador que gerencia as 20 iterações do pipeline.
    *   **`worker.py`**: Script que executa uma única iteração (treino, validação e teste).
    *   **`processamento/`**: Contém a lógica de limpeza de dados e divisão estratificada (50% treino, 25% validação, 25% teste).
    *   **`metodos_aprendizado/`**: Implementação de todos os modelos (KNN, AD, NB, SVM, MLP) e ensembles (RF, Bagging, Boosting e Combinações).
    *   **`utils/`**: Funções auxiliares de log e formatação.
*   **`docs/`**: Documentação complementar.
    *   **`decisões.md`**: Detalha as escolhas técnicas feitas pela equipe.
*   **`datasets_processados/`**: Onde o dataset final (após limpeza e amostragem de 5k) é armazenado.
*   **`resultados/`**: Arquivos CSV gerados após as 20 iterações com as acurácias de cada método.
*   **`modelos/`**: Modelos persistidos em formato `.joblib` para reuso em combinações.

## 🚀 Como Executar

### 1. Pré-requisitos

Bibliotecas:

```bash
pip install pandas scikit-learn tqdm joblib numpy
```

### 2. Executando o Pipeline
**O projeto foi desenhado para ser executado em duas etapas**, garantindo que os modelos monolíticos sejam treinados antes das combinações que dependem deles.

**Passo A: Treinar Modelos Monolíticos**
Execute os métodos individuais (KNN, AD, NB, SVM, MLP). No arquivo `src/worker.py`, a lista `TECNICAS_PARA_RODAR` deve conter os métodos de base.
```bash
python src/main.py
```

**Passo B: Executar SMCs e Combinações**
Após os modelos individuais serem salvos na pasta `modelos/`, ative os métodos de SMC (Random Forest, Bagging, Boosting) e as combinações (Soma, Majoritária, Borda Count) no `worker.py` e execute novamente.

### 3. Modo de Teste
Para verificar se o pipeline está funcionando rapidamente sem rodar as 20 iterações completas:
```bash
python src/main.py --teste
```

## ⚙️ Guia de Configurações

O comportamento do pipeline pode ser customizado alterando variáveis específicas nos arquivos-fonte:

### 1. Orquestração e Paralelismo (`src/main.py`)
*   **Número de Iterações:** Altere a variável `num_iteracoes` (linha 39) para definir o total de repetições. (Padrão: 20)
*   **Limite de Janelas:** Altere a variável `limite_janelas` (linha 41) para definir quantos terminais rodam em paralelo. (Padrão: 4 janelas normais, 2 no modo teste)
*   **Comportamento do Terminal:** Na linha 66, o comando `cmd.exe /c` fecha a janela após o fim. Para manter a janela aberta para inspeção, altere para `cmd.exe /k`. (Padrão: `cmd.exe /c` - Fechar após finalizar)

### 2. Execução da Iteração (`src/worker.py`)
*   **Métodos Ativos:** A lista `TECNICAS_PARA_RODAR` (linha 94) define quais algoritmos serão treinados. Você pode comentar/descomentar métodos específicos para rodar apenas o que desejar. (Padrão: Métodos monolíticos + Bagging, Boosting)
*   **Tempo de Espera:** A variável `esperar` (linha 178) define quantos segundos a janela do terminal aguarda antes de fechar após o sucesso da iteração. (Padrão: 1s)

### 3. Amostragem de Dados (`src/processamento/pre_processamento.py`)
*   **Tamanho do Dataset:** Altere o valor de `n_amostras` na função `reduzir_dataset` (linha 88) para trabalhar com mais ou menos batalhas. (Padrão: 5.000 amostras)

## 📊 Análise de Resultados
Após a conclusão das 20 repetições, os resultados médios e desvios padrão podem ser analisados no CSV gerado em `resultados/`. O sistema está preparado para realizar os testes estatísticos de Friedman e Nemenyi (95% de confiança) conforme exigido no PDF.

## 💻 Hardware e Performance

O pipeline completo (20 iterações com 5.000 amostras) leva aproximadamente **3 horas** para ser concluído, utilizando as seguintes especificações de hardware como referência:

*   **Processador:** Intel Core i5-12450H (12th Gen) - 8 núcleos / 12 threads.
*   **Memória RAM:** 16 GB DDR5 4800 MT/s.
*   **Sistema Operacional:** Windows 11.

---
**Equipe:** André Gustavo Franco e Matheus Barros


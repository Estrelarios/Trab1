# Decisões de Projeto - Trabalho 1 (AM)

Este documento resume as escolhas que fizemos durante o desenvolvimento do trabalho e por que as tomamos.

## 1. Pré-processamento e Amostragem
*   **Decisão:** Reduzimos o dataset para 5.000 batalhas, usando amostragem estratificada.
*   **Justificativa:** O dataset original de 50.000 batalhas estava demorando demais para treinar (mais de 8 horas só para o SVM com C=1000). Escolhemos 5.000 instâncias para o projeto ser executável no tempo disponível, usando estratificação para garantir que a proporção de vitórias (Winner) não fosse alterada.
*   **Atributos:** Retiramos os atributos #, Nome do Pokémon, Legendary e Generation, pois não ajudam na predição de quem vence a luta. Usamos codificação One-Hot para transformar os tipos primário e secundário em colunas binárias.

## 2. Reprodutibilidade (Seeds)
*   **Decisão:** Usamos o número da iteração (1 a 20) como semente (seed) para o sorteio dos dados.
*   **Justificativa:** Isso garante que cada uma das 20 repetições exigidas pelo professor use dados diferentes (e aleatórios), mas que qualquer pessoa consiga rodar o código e chegar nos mesmos resultados que nós.

## 3. Boosting apenas com 3 estimadores
*   **Decisão:** Não incluímos KNN e MLP na lista de estimadores base do Boosting (AdaBoost).
*   **Justificativa:** O AdaBoost do scikit-learn precisa que o estimador aceite pesos nas amostras (`sample_weight`) durante o treino. Como o KNN e o MLP não suportam isso nativamente, o código quebraria com erro de `TypeError`. Por isso, focamos em Árvore de Decisão, Naive Bayes e SVM.

## 4. Bagging com MLP
*   **Decisão:** Diminuímos a quantidade de estimadores (`n_estimators`) especificamente quando o Bagging usa MLP.
*   **Justificativa:** Estavam ocorrendo muitos erros de convergência quando tentávamos rodar muitos estimadores MLP em paralelo. Reduzir para valores menores ([5, 10, 15]) resolveu o problema e permitiu que o treino finalizasse com sucesso.

## 5. Escalonamento de Dados (StandardScaler)
*   **Decisão:** Aplicamos o `StandardScaler` nos dados antes de rodar SVM, KNN e MLP.
*   **Justificativa:** Como esses modelos dependem de cálculos de distância ou gradiente, eles funcionam muito melhor se os dados estiverem na mesma escala. Usamos o `Pipeline` para garantir que o escalonamento do treino não "vazasse" informações para os conjuntos de teste e validação.

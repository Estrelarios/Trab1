# Decisões de projeto

## Pre processamento de datasets
1. Retiramos os atributos #, Nome do pokemon, Legendary e Generation
2. Usamos variável dummy para binarizar os tipos primário e secundário do pokemon

Reduzimos a quantidade de batalhas para 5k
utilizamos a seeds por iteração igual ao número da iteração

## Bagging e Boosting

Não sei se entendi bem a especificação para implementar o bagging e boosting.

O tópico "Estimadores (os classificadores empregados terão seus 
hiperparâmetros setados com os valores default)" dá a entender que não é para usar os modelos salvos já treinados.

## Metodos a executar

**Problema:** Do jeito que o projeto está (até o commit FEAT: implementa salvar modelos 8ce69cf156a0af265a9641fcb2b5c08f5c5e01fd), não há como executar um conjunto específico de metodos. Teria que executar tudo de novo para poder rodar métodos que ainda não foram implementados, o que é lento.

**Solução:** Definir quais métodos serão executados e salvar em csvs diferentes (concatenar manualmente depois).

## StandardScaler

Antes de qualquer exeução dos métodos SVM, KNN e MLP foi aplicado um standardScalar nos dados para melhorar performance.

## Boosting apenas com 3 estimadores

GEMINI disse o seguinte:

```
O algoritmo AdaBoost funciona atribuindo pesos às amostras a cada iteração (focando nas que o modelo anterior errou).
  Para isso, ele exige que o estimador base suporte o parâmetro sample_weight no método .fit().

  No scikit-learn:
   1. DecisionTreeClassifier: Suporta pesos (é o padrão do Boosting).
   2. GaussianNB: Suporta pesos.
   3. SVC (SVM): Suporta pesos.
   4. KNeighborsClassifier (KNN): Não suporta pesos de amostra.
   5. MLPClassifier (Neural Network): Não suporta pesos de amostra.

  Se eu adicionar o KNN ou o MLP na lista do Boosting, o código irá quebrar com um erro de TypeError assim que o
  AdaBoost tentar treiná-los, avisando que eles não aceitam o argumento sample_weight.```

Dito isso, não está fazendo com knn e mlp, mas não testei se da mesmo esse erro

## MLP no Bagging

MLP não estava funcionando dentro do bagging. Não descobrimos o porquê. Acredito que reduzir a quantidade de estimadores na busca de hiperparâmetros resolveu o problema. Assim, os valores para o parâmetro n_estimators do bagging foram reduzidos quando a busca era feita para o Classificador MLP.


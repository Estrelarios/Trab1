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




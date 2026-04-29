from time import time
from pandas import DataFrame

from utils.pre_processamento import pre_processar_dados
from utils.print_customizado import cprint
from utils.ler_dataset_processado import ler_datasets
from metodos_aprendizado.metodosAprendizado import MetodosAprendizado




def main():
    
    inicio = time()
    
    cprint("Iniciando fluxo principal...", label="MAIN")
    
    # Le o dataset já processado
    dados = ler_datasets("dados")

    cprint(f"Sucesso! Dataset de Pokémon processado com {dados.shape[0]} linhas.", label="MAIN")

    media = 0
    num_repeticoes = 1
    ma = MetodosAprendizado()
    for i in range(num_repeticoes):
        cprint(f"#################### Iteração {i+1}/{num_repeticoes}: ####################", label="MAIN")
        cprint("50% o conjunto de dados para treino, 25% para teste, 25% para validação.", label="MAIN")

        
        
        x_treino, y_treino, x_teste, y_teste, x_val, y_val = ma.split_dataset(dados)

        resultados = ma.disparar_comando(parametros={
            "x_treino": x_treino,
            "y_treino": y_treino,
            "x_teste":  x_teste,
            "y_teste":  y_teste,
            "x_val":    x_val,
            "y_val":    y_val,
        })
        #print(resultados)

        ############## ideia 2 ###############

        # from metodos_aprendizado import *

        # funcoes = [knn, nb, ad, ]

        # for funcao in funcoes:
        #     acuracia = funcao(x_treino, y_treino, x_teste, y_teste, x_val, y_val)


        # cprint("Executando o Random Forest...", label="MAIN")
        # cprint("Fazendo busca de hiperparametros...", label="MAIN")
    
    # cprint(f"Media KNN: {media/num_repeticoes}", label="MAIN")
    # 2. 


    # Para cada metodo de aprendizado no diretorio metodo aprendizado


if __name__ == "__main__":
    
    main()


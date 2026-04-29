from time import time
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from utils.pre_processamento import pre_processar_dados
from utils.print_customizado import cprint
from utils.ler_dataset_processado import ler_datasets
from metodos_aprendizado import KNN


def split_dataset(dados : DataFrame, tam_treino=0.5, tam_teste=0.25, tam_validacao=0.25):
    """Aplica o train_test_split duas vezes para dividir os dados em treino (50%), teste (25%) e validação (25%)

    Args:
        dataset (DataFrame): _description_
        tam_treino (float): Porcentagem do conjutno de treino
        tam_teste (float): Porcentagem do conjutno de teste
        tam_val (float): Porcentagem do conjutno de validação

    Returns:
        _type_: x_treino, y_treino, x_teste, y_teste, x_val, y_val
    """

    # Calculando tamanhos
    total = tam_teste + tam_validacao
    tam_teste = tam_teste/total
    tam_validacao = tam_validacao/total

    X = dados.iloc[:,1:]
    Y = dados.iloc[:,0]
    
    x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, train_size=tam_treino, test_size=1-tam_treino)

    x_teste, x_val, y_teste, y_val = train_test_split(x_temp, y_temp, train_size=tam_teste, test_size=tam_validacao)

    return x_treino, y_treino, x_teste, y_teste, x_val, y_val


def main():
    
    inicio = time()
    
    cprint("Iniciando fluxo principal...")
    
    # Le o dataset já processado
    df_combats = ler_datasets("dados")

    cprint(f"Sucesso! Dataset de Pokémon processado com {df_combats.shape[0]} linhas.")

    media = 0
    num_repeticoes = 20
    for i in range(num_repeticoes):
        cprint(f"#################### Iteração {i+1}/{num_repeticoes}: ####################", label="MAIN")
        cprint("50% o conjunto de dados para treino, 25% para teste, 25% para validação.", label="MAIN")

        x_treino, y_treino, x_teste, y_teste, x_val, y_val = split_dataset(df_combats, 0.50, 0.25, 0.25)

        cprint("Executando o KNN...")
        
        acuracia = KNN.knn(x_treino, y_treino, x_teste, y_teste, x_val, y_val)
        
        media += acuracia
        
        

        {}

        cprint("Executando o Random Forest...", label="MAIN")
        cprint("Fazendo busca de hiperparametros...", label="MAIN")
    
    cprint(f"Media KNN: {media/num_repeticoes}")
    # 2. 


    # Para cada metodo de aprendizado no diretorio metodo aprendizado


if __name__ == "__main__":
    
    main()


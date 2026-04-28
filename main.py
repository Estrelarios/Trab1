from time import time
from utils.pre_processamento import pre_processar_dados
from utils.print_customizado import cprint
from utils.ler_dataset_processado import ler_datasets


def main():
    
    inicio = time()
    
    cprint("Iniciando fluxo principal...")
    
    # Le o dataset já processado
    df_combats = ler_datasets()
    
    cprint(f"Sucesso! Dataset de Pokémon processado com {df_combats.shape[0]} linhas.")

    # 1. Preparar dados: train_test_split
    # 2. 


    # Para cada metodo de aprendizado no diretorio metodo aprendizado


if __name__ == "__main__":
    
    main()


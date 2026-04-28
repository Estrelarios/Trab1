from time import time
from utils.pre_processamento import pre_processar_datasets
from utils.print_customizado import cprint


def main():
    
    inicio = time()
    
    cprint("Iniciando fluxo principal...")
    
    # Chama o processamento
    df_combat, df_pokemon = pre_processar_datasets()
    
    cprint(f"Sucesso! Dataset de Pokémon processado com {df_pokemon.shape[0]} linhas.")

    # 1. Preparar dados: train_test_split
    # 2. 


    # Para cada metodo de aprendizado no diretorio metodo aprendizado








if __name__ == "__main__":
    main()

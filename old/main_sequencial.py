from time import time
from pandas import DataFrame

from utils.pre_processamento import pre_processar_dados
from utils.print_customizado import cprint
from utils.ler_dataset_processado import ler_datasets
from metodos_aprendizado.metodosAprendizado import MetodosAprendizado
import argparse





def main(modo_teste=False):

    if modo_teste:
        cprint("###########################################", label="MAIN TESTE")
        cprint("Rodando em MODO DE TESTE (Dataset reduzido)", label="MAIN TESTE")
        cprint("###########################################", label="MAIN TESTE")
    
    inicio = time()
    
    cprint("Iniciando fluxo principal...", label="MAIN")
    
    # Le o dataset já processado
    dados = ler_datasets("dados")

    cprint(f"Sucesso! Dataset de Pokémon processado com {dados.shape[0]} linhas.", label="MAIN")

    # Definindo número de repetições
    num_repeticoes = 20

    if modo_teste:
        dados = dados.sample(100)
        num_repeticoes = 2
    
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
            "modo_teste": modo_teste
        })
       


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline de Aprendizado de Máquina - Pokémon")

    # Adiciona a flag --teste.
    # action="store_true" significa: se a flag aparecer, o valor é True. Se não, é False.
    parser.add_argument("--teste", action="store_true", help="Executa o pipeline com dados reduzidos para teste rápido.")

    args = parser.parse_args()

    main(modo_teste=args.teste)


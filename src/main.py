import os
import subprocess
import time
import pandas as pd
import argparse
import sys

# Adiciona a pasta src ao path para permitir imports internos
sys.path.append(os.path.dirname(__file__))

from utils.print_customizado import cprint

def main():

    # 1. Configuração do Orquestrador
    parser = argparse.ArgumentParser(description="Orquestrador do Pipeline de ML - Pokémon")
    parser.add_argument("--teste", action="store_true", help="Executa o pipeline em modo de teste.")
    args = parser.parse_args()

    # Define o total de iterações e o limite de janelas simultâneas
    num_iteracoes = 4 if args.teste else 20
    limite_janelas = 2 if args.teste else 4
    
    # Define o arquivo de saída dependendo do modo
    nome_arquivo = "resultados_teste.csv" if args.teste else "resultados.csv"
    caminho_csv = os.path.join("resultados", nome_arquivo)

    # Cria a pasta de resultados caso não exista
    if not os.path.exists("resultados"):
        os.makedirs("resultados")

    # 2. Checkpoint: Verifica quais iterações já foram feitas
    concluidas = []
    if os.path.exists(caminho_csv):
        try:
            df = pd.read_csv(caminho_csv)
            if 'iteracao' in df.columns:
                concluidas = df['iteracao'].tolist()
        except Exception:
            pass

    # Ver quantas iterações estão pendentes
    pendentes = [i for i in range(1, num_iteracoes + 1) if i not in concluidas]
    
    cprint(f"Status: {len(concluidas)} concluídas, {len(pendentes)} pendentes.", label="CHEFE")

    if not pendentes:
        cprint("Todas as iterações já foram concluídas!", label="CHEFE")
        return

    # 3. Gerenciamento de Processos 
    processos_ativos = []

    for it in pendentes:
        while len(processos_ativos) >= limite_janelas:
            processos_ativos = [p for p in processos_ativos if p.poll() is None]
            time.sleep(1)

        cprint(f"Abrindo terminal para Iteração {it}...", label="ORQUESTRADOR")

        flag_teste = "--teste" if args.teste else ""
        comando_python = f"python src/worker.py --iteracao {it} {flag_teste}"

        # Chamamos o cmd.exe diretamente com /c para abrir o worker.
        # /c: executa o comando e TERMINA o processo (fechando a janela),
        # permitindo que o orquestrador saiba que uma vaga foi liberada.
        p = subprocess.Popen(
            ["cmd.exe", "/c", comando_python], 
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        processos_ativos.append(p)


    cprint("Todas as iterações pendentes foram disparadas.", label="CHEFE")

if __name__ == "__main__":
    main()

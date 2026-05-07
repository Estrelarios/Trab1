import os
import subprocess
import time
import pandas as pd
import argparse
import sys

# Adiciona a pasta src ao path para permitir imports internos
sys.path.append(os.path.dirname(__file__))

from utils.print_customizado import cprint

# Define a raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Orquestrador do Pipeline de ML - Pokémon")
    parser.add_argument("--teste", action="store_true", help="Executa o pipeline em modo de teste.")
    args = parser.parse_args()

    num_iteracoes = 4 if args.teste else 20
    limite_janelas = 2 if args.teste else 4
    
    nome_arquivo = "resultados_teste.csv" if args.teste else "resultados.csv"
    resultados_dir = os.path.join(BASE_DIR, "resultados")
    caminho_csv = os.path.join(resultados_dir, nome_arquivo)

    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    # Checkpoint
    concluidas = []
    if os.path.exists(caminho_csv):
        try:
            df = pd.read_csv(caminho_csv)
            if 'iteracao' in df.columns:
                concluidas = df['iteracao'].tolist()
        except Exception:
            pass

    pendentes = [i for i in range(1, num_iteracoes + 1) if i not in concluidas]
    
    cprint(f"Status: {len(concluidas)} concluídas, {len(pendentes)} pendentes.", label="CHEFE")

    if not pendentes:
        cprint("Todas as iterações já foram concluídas!", label="CHEFE")
        return

    processos_ativos = []

    for it in pendentes:
        while len(processos_ativos) >= limite_janelas:
            processos_ativos = [p for p in processos_ativos if p.poll() is None]
            time.sleep(1)

        cprint(f"Abrindo terminal para Iteração {it}...", label="ORQUESTRADOR")

        flag_teste = "--teste" if args.teste else ""
        # Chamada usando o caminho relativo à raiz do projeto
        comando_python = f"python src/worker.py --iteracao {it} {flag_teste}"

        p = subprocess.Popen(
            ["cmd.exe", "/k", comando_python], 
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=BASE_DIR # Define o diretório de trabalho como a raiz
        )
        processos_ativos.append(p)

    cprint("Todas as iterações pendentes foram disparadas.", label="CHEFE")

if __name__ == "__main__":
    main()

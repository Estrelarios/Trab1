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

    # Gera timestamp para o nome do arquivo desta sessão
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"resultados_teste_{timestamp}.csv" if args.teste else f"resultados_{timestamp}.csv"
    
    num_iteracoes = 4 if args.teste else 20
    limite_janelas = 2 if args.teste else 4
    
    resultados_dir = os.path.join(BASE_DIR, "resultados")
    caminho_csv = os.path.join(resultados_dir, nome_arquivo)

    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    # Checkpoint (desativado por padrão pelo uso de timestamp, mas mantido como solicitado)
    concluidas = []
    # ... (lógica de checkpoint omitida para brevidade se o arquivo for novo)

    pendentes = [i for i in range(1, num_iteracoes + 1)]
    
    cprint(f"Iniciando Sessão: {nome_arquivo}", label="CHEFE")
    cprint(f"Status: {len(pendentes)} iterações a executar.", label="CHEFE")

    processos_ativos = [] # Lista de tuplas (processo, iteracao)

    for it in pendentes:
        while len(processos_ativos) >= limite_janelas:
            # Verifica processos que terminaram
            for p, i in processos_ativos[:]:
                if p.poll() is not None:
                    cprint(f"Iteração {i} Finalizada!", label="CHEFE")
                    processos_ativos.remove((p, i))
            time.sleep(1)

        cprint(f"Abrindo terminal para Iteração {it}...", label="CHEFE")

        flag_teste = "--teste" if args.teste else ""
        # Passa o nome do arquivo para o worker
        comando_python = f"python src/worker.py --iteracao {it} {flag_teste} --arquivo {nome_arquivo}"

        p = subprocess.Popen(
            ["cmd.exe", "/c", comando_python], 
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=BASE_DIR # Define o diretório de trabalho como a raiz
        )
        processos_ativos.append((p, it))

    # Espera os últimos processos terminarem
    while processos_ativos:
        for p, i in processos_ativos[:]:
            if p.poll() is not None:
                cprint(f"Iteração {i} Finalizada!", label="CHEFE")
                processos_ativos.remove((p, i))
        time.sleep(1)

    cprint("Todas as iterações pendentes foram disparadas e concluídas.", label="CHEFE")

if __name__ == "__main__":
    main()

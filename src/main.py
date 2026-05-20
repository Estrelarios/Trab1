import sys
import os

sys.path.append(os.path.dirname(__file__))

import subprocess
import time
import pandas as pd
import argparse


# Adiciona a pasta src ao path para permitir imports internos
sys.path.append(os.path.dirname(__file__))

from utils.print_customizado import cprint
from testes_estatisticos.testes import teste_friedman

# Define a raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def carregar_historico(caminho_csv: str) -> dict:
    """Lê o CSV de resultados e retorna o histórico no formato esperado pelo Friedman."""

    if not os.path.exists(caminho_csv):
        cprint(f"Arquivo não encontrado: {caminho_csv}", label="ANÁLISE")
        sys.exit(1)

    df = pd.read_csv(caminho_csv)
    cprint(f"Arquivo carregado: {caminho_csv} ({len(df)} iterações)", label="ANÁLISE")

    # Converte o DataFrame para { "metodo_knn": [acc1, acc2, ...], ... }
    historico = {coluna: df[coluna].tolist() for coluna in df.columns}
    return historico


def executar_pipeline(args, nome_arquivo, caminho_csv):
    """Dispara os workers em paralelo para executar as iterações do pipeline."""

    num_iteracoes = 4 if args.teste else 20
    limite_janelas = 2 if args.teste else 4

    pendentes = list(range(1, num_iteracoes + 1))

    cprint(f"Iniciando Sessão: {nome_arquivo}", label="CHEFE")
    cprint(f"Status: {len(pendentes)} iterações a executar.", label="CHEFE")

    processos_ativos = []

    for it in pendentes:
        while len(processos_ativos) >= limite_janelas:
            for p, i in processos_ativos[:]:
                if p.poll() is not None:
                    cprint(f"Iteração {i} Finalizada!", label="CHEFE")
                    processos_ativos.remove((p, i))
            time.sleep(1)

        cprint(f"Abrindo terminal para Iteração {it}...", label="CHEFE")

        flag_teste = "--teste" if args.teste else ""
        comando_python = f"python src/worker.py --iteracao {it} {flag_teste} --arquivo {nome_arquivo}"

        p = subprocess.Popen(
            ["cmd.exe", "/c", comando_python],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=BASE_DIR
        )
        processos_ativos.append((p, it))

    while processos_ativos:
        for p, i in processos_ativos[:]:
            if p.poll() is not None:
                cprint(f"Iteração {i} Finalizada!", label="CHEFE")
                processos_ativos.remove((p, i))
        time.sleep(1)

    cprint("Todas as iterações pendentes foram disparadas e concluídas.", label="CHEFE")


def main():
    parser = argparse.ArgumentParser(description="Orquestrador do Pipeline de ML - Pokémon")
    parser.add_argument("--teste",   action="store_true", help="Executa o pipeline em modo de teste.")
    parser.add_argument("--analise", action="store_true", help="Executa apenas a análise estatística (Friedman) sobre um CSV existente.")
    args = parser.parse_args()

    resultados_dir = os.path.join(BASE_DIR, "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    # ── Modo análise ──────────────────────────────────────────────────────────
    if args.analise:
        caminho_csv = r"C:\Users\Usuario\Desktop\Unio\Machine Learning\Trab1\resultados\resultados.csv"
        historico = carregar_historico(caminho_csv)
        teste_friedman(historico, confianca=0.95)
        return

    # ── Modo pipeline normal ──────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"resultados_teste_{timestamp}.csv" if args.teste else f"resultados_{timestamp}.csv"
    caminho_csv  = os.path.join(resultados_dir, nome_arquivo)

    executar_pipeline(args, nome_arquivo, caminho_csv)


if __name__ == "__main__":
    main()
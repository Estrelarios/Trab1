import argparse
import pandas as pd
import os
import time
import traceback
import sys
from datetime import datetime

# Adiciona a pasta src ao path para permitir imports internos
# Isso permite que 'from processamento...' funcione mesmo rodando da raiz ou de src
sys.path.append(os.path.dirname(__file__))

from metodos_aprendizado.metodosAprendizado import MetodosAprendizado
from processamento.ler_dataset_processado import ler_datasets
from utils.print_customizado import cprint

# Define a raiz do projeto (um nível acima de src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log_error(iteracao, error_msg):
    """Salva o erro em um arquivo de log na raiz do projeto."""
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(log_dir, f"worker_it{iteracao}_error.log")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"--- Erro na Iteração {iteracao} em {timestamp} ---\n")
        f.write(error_msg)
        f.write("\n" + "="*50 + "\n\n")

def safe_save(df, caminho):
    """Tenta salvar os resultados no CSV com retry para evitar conflitos de escrita."""
    while True:
        try:
            header = not os.path.exists(caminho)
            df.to_csv(caminho, mode='a', index=False, header=header)
            break 
        except PermissionError:
            print(f"O arquivo {caminho} está ocupado. Tentando novamente em 1s...")
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Worker de processamento de uma única iteração.")
    parser.add_argument("--iteracao", type=int, required=True, help="Número da iteração atual.")
    parser.add_argument("--teste", action="store_true", help="Ativa o modo de teste.")
    args = parser.parse_args()

    try:
        cprint(f"Iniciando Iteração {args.iteracao}...", label=f"CLT {args.iteracao}")

        # Carga de dados (o loader já deve lidar com caminhos internos)
        dados = ler_datasets("dados")
        
        if args.teste:
            dados = dados.sample(1000)

        ma = MetodosAprendizado()
        ma.modo_teste = args.teste
        
        x_treino, y_treino, x_teste, y_teste, x_val, y_val = ma.split_dataset(dados)
        
        resultados = ma.disparar_comando(parametros={
            "x_treino": x_treino, "y_treino": y_treino,
            "x_teste":  x_teste,  "y_teste":  y_teste,
            "x_val":    x_val,    "y_val":    y_val,
            "modo_teste": args.teste
        })

        # Formatação do Resultado
        linha = {"iteracao": args.iteracao}
        linha.update(resultados)
        df_result = pd.DataFrame([linha])

        # Caminho de Salvamento (Sempre na raiz/resultados)
        nome_arquivo = "resultados_teste.csv" if args.teste else "resultados.csv"
        resultados_dir = os.path.join(BASE_DIR, "resultados")
        if not os.path.exists(resultados_dir):
            os.makedirs(resultados_dir)
            
        caminho_csv = os.path.join(resultados_dir, nome_arquivo)
        
        safe_save(df_result, caminho_csv)
        
        cprint(f"Iteração {args.iteracao} finalizada com sucesso!", label=f"CLT {args.iteracao}")

        esperar = 10
        cprint(f"Fechando em {esperar}s")
        time.sleep(esperar)

    except Exception:
        error_msg = traceback.format_exc()
        cprint(f"ERRO FATAL na Iteração {args.iteracao}! Verifique logs/", label=f"ERR {args.iteracao}")
        
        log_error(args.iteracao, error_msg)
        time.sleep(10)
        

if __name__ == "__main__":
    main()

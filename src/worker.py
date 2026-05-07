import argparse
import pandas as pd
import os
import time
import traceback
import sys
from datetime import datetime

# Adiciona a pasta src ao path para permitir imports internos
sys.path.append(os.path.dirname(__file__))

from metodos_aprendizado.metodosAprendizado import MetodosAprendizado
from processamento.ler_dataset_processado import ler_datasets
from utils.print_customizado import cprint

def log_error(iteracao, error_msg):
    """Salva o erro em um arquivo de log para análise posterior."""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join("logs", f"worker_it{iteracao}_error.log")
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- Erro na Iteração {iteracao} em {timestamp} ---\n")
        f.write(error_msg)
        f.write("\n" + "="*50 + "\n\n")

def safe_save(df, caminho):
    """
    Tenta salvar os resultados no CSV.
    Como múltiplos terminais podem tentar escrever ao mesmo tempo,
    usamos um bloco try-except para capturar erros de permissão (arquivo ocupado).
    """
    while True:
        try:
            # Verifica se o arquivo já existe para decidir se escreve o cabeçalho (header)
            header = not os.path.exists(caminho)
            
            # mode='a' (append): adiciona a nova linha ao final do arquivo sem apagar o que já existe
            df.to_csv(caminho, mode='a', index=False, header=header)
            break 
        except PermissionError:
            # Se outro processo estiver escrevendo no CSV, o Windows bloqueia o acesso.
            # Esperamos 1 segundo e tentamos novamente.
            print(f"O arquivo {caminho} está ocupado. Tentando novamente em 1s...")
            time.sleep(1)

def main():
    # 1. Configuração de Argumentos
    parser = argparse.ArgumentParser(description="Worker de processamento de uma única iteração.")
    parser.add_argument("--iteracao", type=int, required=True, help="Número da iteração atual.")
    parser.add_argument("--teste", action="store_true", help="Ativa o modo de teste (dados reduzidos).")
    args = parser.parse_args()

    try:
        cprint(f"Iniciando Iteração {args.iteracao}...", label=f"CLT {args.iteracao}")

        # 2. Carga e Preparação de Dados
        dados = ler_datasets("dados")
        
        if args.teste:
            dados = dados.sample(1000)

        # 3. Execução da Lógica de Aprendizado
        ma = MetodosAprendizado()
        ma.modo_teste = args.teste
        
        x_treino, y_treino, x_teste, y_teste, x_val, y_val = ma.split_dataset(dados)
        
        resultados = ma.disparar_comando(parametros={
            "x_treino": x_treino, "y_treino": y_treino,
            "x_teste":  x_teste,  "y_teste":  y_teste,
            "x_val":    x_val,    "y_val":    y_val,
            "modo_teste": args.teste
        })

        # 4. Formatação do Resultado
        linha = {"iteracao": args.iteracao}
        linha.update(resultados)
        df_result = pd.DataFrame([linha])

        # 5. Salvamento Incremental (Checkpoint)
        nome_arquivo = "resultados_teste.csv" if args.teste else "resultados.csv"
        caminho_csv = os.path.join("resultados", nome_arquivo)
        
        safe_save(df_result, caminho_csv)
        
        cprint(f"Iteração {args.iteracao} finalizada com sucesso!", label=f"CLT {args.iteracao}")

    except Exception:
        error_msg = traceback.format_exc()
        cprint(f"ERRO FATAL na Iteração {args.iteracao}! Verifique logs/worker_it{args.iteracao}_error.log", label=f"ERR {args.iteracao}")
        log_error(args.iteracao, error_msg)
        # Em caso de erro, esperamos um pouco para o usuário ver a mensagem no console antes de fechar
        time.sleep(5)

if __name__ == "__main__":
    main()

import argparse
import pandas as pd
import os
import time
from metodos_aprendizado.metodosAprendizado import MetodosAprendizado
from utils.ler_dataset_processado import ler_datasets
from utils.print_customizado import cprint

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
    # Define o arquivo de saída dependendo do modo
    nome_arquivo = "resultados_teste.csv" if args.teste else "resultados.csv"
    caminho_csv = os.path.join("resultados", nome_arquivo)
    
    safe_save(df_result, caminho_csv)
    
    cprint(f"Iteração {args.iteracao} finalizada com sucesso!", label=f"CLT {args.iteracao}")

if __name__ == "__main__":
    main()

import ast
import pandas as pd

def ler_datasets(dados):
    
    if dados == "dados":
        df_proc = pd.read_csv("datasets_processados/dados.csv")
    #print(df_proc)
    elif dados == "resultados":
        df_proc = pd.read_csv("datasets_processados/resultados.csv")
    
    else:
        print("Opção inválida. Escolha 'dados' ou 'resultados'.")
        return None
    return df_proc
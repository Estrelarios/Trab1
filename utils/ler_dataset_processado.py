import ast
import pandas as pd

def ler_datasets():
    
    df_proc = pd.read_csv("datasets_processados/dados.csv")
    #print(df_proc)
    return df_proc
import ast
import pandas as pd

def ler_datasets():
    
    df_proc = pd.read_csv("datasets_processados/dados.csv")
    #print(df_proc)
    df_proc["First_Tipos"] = df_proc["First_Tipos"].apply(ast.literal_eval)  # converts string to actual list
    df_proc["Second_Tipos"] = df_proc["Second_Tipos"].apply(ast.literal_eval)  # converts string to actual list
    return df_proc
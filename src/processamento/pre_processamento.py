import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from time import sleep
from tqdm import tqdm

# Adiciona o diretório 'src' ao sys.path para permitir imports relativos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.print_customizado import cprint

CAMINHO_COMBATS = "datasets_brutos/combats.csv"
CAMINHO_POKEMON = "datasets_brutos/pokemon.csv"

def ler_datasets_brutos():
    """
    Carrega os conjuntos de dados brutos a partir dos arquivos CSV.
    
    Returns:
        tuple: (df_combats, df_pokemon) contendo os DataFrames carregados.
    """
    df_combats = pd.read_csv(CAMINHO_COMBATS)
    df_pokemon = pd.read_csv(CAMINHO_POKEMON)
    return df_combats, df_pokemon

def pre_processar_dataset_pokemon(df_pokemon):
  """Realiza a limpeza e codificação de atributos do dataset de Pokémon.
  
  Etapas:
  1. Crop dos atributos #, Name, Generation e Legendary
  2. Codificação One-Hot dos tipos 1 e 2, transformando em vetores binários e por fim em colunas.
  
  Args:
      df_pokemon (pd.DataFrame): DataFrame original de Pokémon.
      
  Returns:
      pd.DataFrame: DataFrame processado com atributos numéricos e tipos codificados.
  """
  # Remoção de colunas desnecessárias
  colunas_desnecessarias = ["Name", "#", "Generation", "Legendary"]
  df_proc = df_pokemon.drop(colunas_desnecessarias, axis='columns')

  # Identifica todos os tipos possíveis para manter consistência nos índices
  # Criamos uma lista ordenada de todos os tipos únicos (Fogo, Água, etc.)
  tipos_possiveis = sorted(list(set(df_proc['Type 1'].dropna().unique()) | set(df_proc['Type 2'].dropna().unique())))

  # Codificação categórica para manter o alinhamento das colunas no get_dummies
  df_proc['Type 1'] = pd.Categorical(df_proc['Type 1'], categories=tipos_possiveis)
  df_proc['Type 2'] = pd.Categorical(df_proc['Type 2'], categories=tipos_possiveis)

  vetor1 = pd.get_dummies(df_proc['Type 1']).astype(int)
  vetor2 = pd.get_dummies(df_proc['Type 2']).astype(int)

  # Transforma os 2 vetores em 18 colunas binárias
  tipos_combinados = vetor1.combine(vetor2, lambda a, b: a.combine(b, max))

  df_proc = df_proc.drop(['Type 1', 'Type 2'], axis='columns')
  df_proc = pd.concat([df_proc, tipos_combinados], axis=1)

  return df_proc

def pre_processar_dataset_combats(df_combats, df_pokemon_proc):
  """Mapeia os resultados das batalhas e associa os atributos dos Pokémon envolvidos.
  
  Args:
      df_combats (pd.DataFrame): Histórico de batalhas.
      df_pokemon_proc (pd.DataFrame): Atributos de Pokémon já processados.
      
  Returns:
      pd.DataFrame: Conjunto de dados pronto para treinamento.
  """
  # Substitui strings: 1 se o segundo Pokémon venceu, 0 caso contrário
  df_combats["Winner"] = (df_combats["Winner"] == df_combats["Second_pokemon"]).astype(int)

  # O index do dataset de Pokémon original inicia em 1 (ID do Pokémon)
  df_pokemon_proc.index = df_pokemon_proc.index + 1

  # Para cada coluna do pokemon, mapeia o valor do First e Second pokemon
  for coluna in df_pokemon_proc.columns:
    df_combats[f"First_{coluna}"] = df_combats["First_pokemon"].map(df_pokemon_proc[coluna])
    df_combats[f"Second_{coluna}"] = df_combats["Second_pokemon"].map(df_pokemon_proc[coluna])

  # Remoção de IDs originais
  df_combats = df_combats.drop(["First_pokemon", "Second_pokemon"], axis="columns")
  
  return df_combats

def reduzir_dataset(df, n_amostras=5000):
  """Reduz o tamamho do conjunto de dados com estratificação.
  
  Args:
      df (pd.DataFrame): DataFrame completo.
      n_amostras (int): Número de instâncias desejadas na amostra.
      
  Returns:
      pd.DataFrame: Amostra reduzida mantendo a distribuição original das classes.
  """
  cprint(f"Reduzindo dataset para {n_amostras} amostras (Estratificado)...")
  
  df_reduzido, _ = train_test_split(
      df, 
      train_size=n_amostras, 
      stratify=df['Winner'], 
      random_state=42 # Seed fixa para consistência
  )

  cprint(f"Redução concluida!")
  
  return df_reduzido

def executar_pipeline_processamento():
  """
  Orquestra o fluxo completo de pré-processamento de dados.
  """
  df_combats, df_pokemon = ler_datasets_brutos()

  df_pokemon_proc = pre_processar_dataset_pokemon(df_pokemon)
  df_combats_proc = pre_processar_dataset_combats(df_combats, df_pokemon_proc)

  # Reduzir para 5k batalhas
  df_combats_proc = reduzir_dataset(df_combats_proc, n_amostras=5000)

  # Salva o dataset processado
  output_path = "datasets_processados/dados.csv"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  df_combats_proc.to_csv(output_path, index=False)

def main():
  cprint("Iniciando pré-processamento dos datasets...")
  executar_pipeline_processamento()
  cprint("Processamento concluído!")

if __name__ == "__main__":
    main()

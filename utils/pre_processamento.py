import pandas as pd
from utils.print_customizado import cprint

"""# Preparação de dados

1. Unir datasets
2. Substituir Ids e nomes de pokemon por algum arranjo dos atributos
3. ban no is_legendary e geração

## Ideias

Pandas.getdummy() para vetorizar os tipos de pokemon
"""

CAMINHO_COMBATS = "datasets_brutos/combats.csv"
CAMINHO_POKEMON = "datasets_brutos/pokemon.csv"

def ler_datasets():
  """Lê os datasets e retorna em dataframe pandas
  """
  df_combats = pd.read_csv(CAMINHO_COMBATS)
  df_pokemon = pd.read_csv(CAMINHO_POKEMON)
  return df_combats, df_pokemon

def pre_processar_dataset_pokemon(df_pokemon):
  """Realiza o pré-processamento do dataset de pokemon, realizando:
  1. Crop dos atributos #, Name, Generation e Legendary
  2. Transforma Type 1 e Type 2 em colunas de vetores binários

  Returns:
    Dataframe : df_pokemon_proc
  """

  # 1. Remover colunas desnecessárias
  colunas_desnecessarias = ["Name", "#", "Generation", "Legendary"]
  df_proc = df_pokemon.drop(colunas_desnecessarias, axis='columns')

  # 2. Identificar todos os tipos possíveis para manter consistência nos índices
  # Criamos uma lista ordenada de todos os tipos únicos (Fogo, Água, etc.)
  tipos_possiveis = sorted(list(set(df_proc['Type 1'].dropna().unique()) | set(df_proc['Type 2'].dropna().unique())))

  # 3. Criar os vetores binários usando Categorical e get_dummies
  df_proc['Type 1'] = pd.Categorical(df_proc['Type 1'], categories=tipos_possiveis)
  df_proc['Type 2'] = pd.Categorical(df_proc['Type 2'], categories=tipos_possiveis)

  vetor1 = pd.get_dummies(df_proc['Type 1']).astype(int)
  vetor2 = pd.get_dummies(df_proc['Type 2']).astype(int)

  # 4. Converter as linhas de dummies em listas (o "vetor binário" que você pediu)
  df_proc['Type 1'] = vetor1.values.tolist()
  df_proc['Type 2'] = vetor2.values.tolist()

  # Salva o dataset processado
  # a lista será salva como uma string "[0, 1, 0...]" no csv
  df_proc.to_csv("datasets_processados/pokemon_proc.csv", index=False)

  df_pokemon_proc = df_proc

  return df_pokemon_proc

def pre_processar_datasets():
  
  df_combats, df_pokemon = ler_datasets()

  df_pokemon_proc = pre_processar_dataset_pokemon(df_pokemon)

  return df_combats, df_pokemon_proc

def main():
  
    cprint("Iniciando Pré-Processamento dos datasets")

    df_combats, df_pokemon = pre_processar_datasets()

    cprint("Processamento concluído!")
    cprint(f"Exemplo do vetor do primeiro Pokémon (Tipo 1): {df_pokemon['Type 1'].iloc[0]}")
    cprint(f"Tamanho do vetor: {len(df_pokemon['Type 1'].iloc[0])} tipos mapeados.")

if __name__ == "__main__":
  main()
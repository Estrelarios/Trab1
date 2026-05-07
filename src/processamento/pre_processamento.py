import pandas as pd
from utils.print_customizado import cprint
from processamento.ler_dataset_processado import ler_datasets
from time import sleep

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
  3. Junta os 2 tipos em um único vetor binário

  Returns:
    Dataframe : df_pokemon_proc
  """

  # 1. Remover colunas desnecessárias
  colunas_desnecessarias = ["Name", "#", "Generation", "Legendary", "First_Defense", "Second_Defense"]
  df_proc = df_pokemon.drop(colunas_desnecessarias, axis='columns')

  # 2. Identificar todos os tipos possíveis para manter consistência nos índices
  # Criamos uma lista ordenada de todos os tipos únicos (Fogo, Água, etc.)
  tipos_possiveis = sorted(list(set(df_proc['Type 1'].dropna().unique()) | set(df_proc['Type 2'].dropna().unique())))

  # 3. Criar os vetores binários usando Categorical e get_dummies
  df_proc['Type 1'] = pd.Categorical(df_proc['Type 1'], categories=tipos_possiveis)
  df_proc['Type 2'] = pd.Categorical(df_proc['Type 2'], categories=tipos_possiveis)

  print("AAAAAAAAAAAAAAA")
  print(df_proc['Type 1'])

  vetor1 = pd.get_dummies(df_proc['Type 1']).astype(int)
  vetor2 = pd.get_dummies(df_proc['Type 2']).astype(int)

  # 4. Combinar os dois vetores com max e criar 18 colunas separadas
  tipos_combinados = vetor1.combine(vetor2, lambda a, b: a.combine(b, max))

  # 5. Anexar as 18 colunas ao dataframe e remover Type 1 e Type 2
  df_proc = df_proc.drop(['Type 1', 'Type 2'], axis='columns')
  df_proc = pd.concat([df_proc, tipos_combinados], axis=1)

  # df_proc = df_proc.replace(1, 255)

  df_pokemon_proc = df_proc

  return df_pokemon_proc

def pre_processar_dataset_combats(df_combats, df_pokemon):
  """Pré-processa o dataset de combats, realizando:

  1. Para cada batalha, substitui o vencedor da terceira coluna por 0 ou 1 dependendo se o vencedor é o primeiro ou segundo pokemon, respectivamente.
  2. Troca dos ids do pokemon pelos atributos do dataset pokemon
  
  Returns:
    Dataframe: df_combat_proc
  """
  # 1. Substituir o vencedor por 0 ou 1

  df_combats["Winner"] = (df_combats["Winner"] == df_combats["Second_pokemon"]).astype(int)

  df_pokemon.index = df_pokemon.index + 1

    # Para cada coluna do pokemon, mapeia o valor do First e Second pokemon
  for coluna in df_pokemon.columns:
      df_combats[f"First_{coluna}"] = df_combats["First_pokemon"].map(df_pokemon[coluna])
  for coluna in df_pokemon.columns:
      df_combats[f"Second_{coluna}"] = df_combats["Second_pokemon"].map(df_pokemon[coluna])

  # Remove as colunas de ID originais
  df_combats = df_combats.drop(["First_pokemon", "Second_pokemon"], axis="columns")
  
  return df_combats


def pre_processar_dados():
  
  df_combats, df_pokemon = ler_datasets()

  df_pokemon_proc = pre_processar_dataset_pokemon(df_pokemon)

  df_combats_proc = pre_processar_dataset_combats(df_combats, df_pokemon_proc)

  # Salva o dataset processado
  df_combats_proc.to_csv("datasets_processados/dados.csv", index=False)

  return 0


from tqdm import tqdm

def main():
  
    cprint("Iniciando Pré-Processamento dos datasets")

    pre_processar_dados()
    print("Lendo os datasets processados...")
    for _ in tqdm(range(3), desc="Carregando"):
        sleep(.5)
    dados = ler_datasets()

    cprint("Processamento concluído!")
    # cprint(f"{dados.head()}")

if __name__ == "__main__":
  main()
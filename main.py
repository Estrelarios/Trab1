import sklearn as sk
import pandas as pd
"""# Preparação de dados

1. Unir datasets
2. Substituir Ids e nomes de pokemon por algum arranjo dos atributos
3. ban no is_legendary e geração

## Ideias

Pandas.getdummy() para vetorizar os tipos de pokemon
"""

def main():
  # Paths
  path_combats = "/combats.csv"
  path_pokemons = "/pokemon.csv"

  # Transforma em Dataframe Pandas
  df_combats = pd.read_csv(path_combats)
  df_pokemon = pd.read_csv(path_pokemons)

  # print(df_combats)
  # print(df_pokemon)

  """# Crop nos atributos"""

  df_pokemon_drop = df_pokemon.drop(["Name", "#", "Generation", "Legendary"], axis=1)

  print(df_pokemon_drop)

  """# Dummy

  vetorizar os tipos de pokemons com get_dummy

  ex:
  """

  dummies_1 = pd.get_dummies(df_pokemon_drop['Type 1'])
  dummies_2 = pd.get_dummies(df_pokemon_drop['Type 2'])
  #print(dummies_1)

  #df_pokemon_drop = pd.concat([df_pokemon_drop, dummies_1, dummies_2], axis=1)

  print(df_pokemon_drop)
    # df_pokemon_drop[0][0] = dummies_1[0]

  


if __name__ == "__main__":
  main()
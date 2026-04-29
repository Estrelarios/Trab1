import sklearn as sk
import joblib as jb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

def knn(x_treino, y_treino, x_teste, y_teste, x_val, y_val):

    print("Fazendo busca de hiperparametros...")
    maiorAcc = -1
    for j in ("distance", "uniform"):
        for i in tqdm(range(1, 51), desc=f"weights='{j}'"):
            KNN = KNeighborsClassifier(n_neighbors=i, weights=j, n_jobs=-1)
            KNN.fit(x_treino, y_treino)
            opiniao = KNN.predict(x_teste)
            Acc = accuracy_score(y_teste, opiniao)
            if (Acc > maiorAcc):
                maiorAcc = Acc
                melhor_modelo = KNN
                
    # print("Salvando o modelo...")
    jb.dump(melhor_modelo, "ultimo_modelo_KNN")
    print("\nMelhor configuração para o KNN")
    print(melhor_modelo.get_params())
    
    print("\nMelhor configuração:")
    print("K: ",melhor_modelo.n_neighbors,"Weights: ",melhor_modelo.weights)
    

    """**Aplicando o melhor modelo sobre o conjunto de teste**"""

    # print("\n\nDesempenho sobre o conjunto de teste")
    opiniao = melhor_modelo.predict(x_val)
    print("\nK: ",melhor_modelo.n_neighbors," Acurácia sobre o teste: ",accuracy_score(y_val, opiniao))

    # cm = confusion_matrix(y_val, opiniao)
    # print(f"\nMatriz de confusão:{cm}")

    return accuracy_score(y_val, opiniao)


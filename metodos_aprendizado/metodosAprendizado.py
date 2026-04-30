import sklearn as sk
import joblib as jb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import inspect
from utils.pre_processamento import pre_processar_dados
from utils.print_customizado import cprint
from utils.ler_dataset_processado import ler_datasets




class MetodosAprendizado:
    def __init__(self):
        self.modo_teste = False

    def disparar_comando(self, parametros: dict = None, callback=None):
        """
        Args:
            parametros (dict): Dicionário com todos os parâmetros disponíveis para os métodos.
            callback (optional): Callback para métodos específicos.
        """
    
        if parametros is None:
            parametros = {}

        if parametros["modo_teste"]:
            self.modo_teste = True
        
        resultados = {}
        metodos = [m for m in dir(self) if m.startswith("metodo") and callable(getattr(self, m))]
        
        for nome_metodo in metodos:
            metodo = getattr(self, nome_metodo)
            
            # Descobre quais parâmetros o método precisa (excluindo 'self')
            assinatura = inspect.signature(metodo)
            params_necessarios = [
                p for p in assinatura.parameters 
                if p != "self"
            ]
            
            # Extrai do dicionário apenas os parâmetros que o método precisa
            kwargs = {p: parametros[p] for p in params_necessarios if p in parametros}
            
            if callback and "callback" in assinatura.parameters:
                resultados[nome_metodo] = metodo(**kwargs, callback=callback)
            else:
                resultados[nome_metodo] = metodo(**kwargs)
        
        return resultados
    
    def split_dataset(self, dados : DataFrame, tam_treino=0.5, tam_teste=0.25, tam_validacao=0.25):
        """Aplica o train_test_split duas vezes para dividir os dados em treino (50%), teste (25%) e validação (25%)

        Args:
            dataset (DataFrame): _description_
            tam_treino (float): Porcentagem do conjutno de treino
            tam_teste (float): Porcentagem do conjutno de teste
            tam_val (float): Porcentagem do conjutno de validação

        Returns:
            _type_: x_treino, y_treino, x_teste, y_teste, x_val, y_val
        """

        # Calculando tamanhos
        total = tam_teste + tam_validacao
        tam_teste = tam_teste/total
        tam_validacao = tam_validacao/total

        X = dados.iloc[:,1:]
        Y = dados.iloc[:,0]
        
        x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, train_size=tam_treino, test_size=1-tam_treino)

        x_teste, x_val, y_teste, y_val = train_test_split(x_temp, y_temp, train_size=tam_teste, test_size=tam_validacao)

        return x_treino, y_treino, x_teste, y_teste, x_val, y_val
    

#    def divisoes_dataset(self)


    def metodo_knn(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):

        cprint("Executando o KNN...", label="KNN")

        # Definido o range (reduzido para modo teste)
        k_range = range(1,51)
        if self.modo_teste:
            k_range = range(1,2)
        
        cprint("Fazendo busca de hiperparametros...", label="KNN")
        maiorAcc = -1
        for j in ("distance", "uniform"):
            for i in tqdm(k_range, desc=f"weights='{j}'"):
                KNN = KNeighborsClassifier(n_neighbors=i, weights=j, n_jobs=-1)
                KNN.fit(x_treino, y_treino)
                opiniao = KNN.predict(x_teste)
                Acc = accuracy_score(y_teste, opiniao)
                if (Acc > maiorAcc):
                    maiorAcc = Acc
                    melhor_modelo = KNN
                    

        print()
        cprint(f"Melhor configuração para o KNN", label="KNN")
        cprint(f"{melhor_modelo.get_params()}", label="KNN")
        
        print()
        cprint("Melhor configuração:", label="KNN")
        cprint(f"K: {melhor_modelo.n_neighbors} , Weights: {melhor_modelo.weights}", label="KNN")
        

        """**Aplicando o melhor modelo sobre o conjunto de teste**"""

        # print("\n\nDesempenho sobre o conjunto de teste")
        opiniao = melhor_modelo.predict(x_val)
        cprint(f"K: {melhor_modelo.n_neighbors}, Acurácia sobre o teste: {accuracy_score(y_val, opiniao)}", label="KNN")

        # cm = confusion_matrix(y_val, opiniao)
        # print(f"\nMatriz de confusão:{cm}")

        return accuracy_score(y_val, opiniao)
    

    def metodo_arvoreDecisao(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        
        cprint("Executando a Árvore de decisão...", label="AD")

        # Definido os ranges (reduzido para modo teste)
        i_range = range(1,11)
        k_range = range(1,11)
        l_range = range(2,16)

        # Altera os ranges para modo teste
        if self.modo_teste:
            i_range = range(1,2)
            k_range = range(1,2)
            l_range = range(2,3)

        cprint("Fazendo busca de hiperparametros...", label="AD")

        maior = -1
        for j in ("entropy","gini"):  #criterion
            for i in tqdm(i_range, ascii=True, desc=f"[ AD ] criteion = {j} "):      #max_depth
                for k in tqdm(k_range, desc=f"[ AD ] max_depth = {i}", leave=False, ascii=True):    #min_samples_leaf
                    for l in l_range:  #min_samples_split
                        for m in ('best','random'): #splitter
                            AD = DecisionTreeClassifier(criterion=j,max_depth=i,min_samples_leaf=k,min_samples_split=l,splitter=m)
                            AD.fit(x_treino,y_treino)
                            opiniao = AD.predict(x_val)
                            Acc = accuracy_score(y_val, opiniao)
                            # print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
                            if (Acc > maior):
                                #cprint(f"Nova melhor configuração encontrada: {Acc}", label="AD")
                                maior = Acc
                                melhor_modelo = AD

        cprint("\nMelhor configuração para a AD", label="AD")
        cprint(f"Criterion: {melhor_modelo.criterion}, max_depth: {melhor_modelo.max_depth}, min_samples_leaf: {melhor_modelo.min_samples_leaf}, min_samples_split: {melhor_modelo.min_samples_split}, splitter: {melhor_modelo.splitter}, Acc: {maior}", label="AD")

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""

        opiniao = melhor_modelo.predict(x_teste)
        cprint(f"Acurácia sobre o teste: {accuracy_score(y_teste, opiniao)}", label="AD")
        
        return accuracy_score(y_teste, opiniao)
    

    def metodo_svm(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando o SVM...", label="SVM")
        pass

    def metodo_randomForest(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando a Random Forest...", label="RF")
        pass

    def metodo_naiveBayes(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando o Naive Bayes...", label="NB")
        pass

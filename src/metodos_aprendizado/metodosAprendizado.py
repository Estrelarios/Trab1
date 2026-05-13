import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pandas import DataFrame
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import inspect

from processamento.pre_processamento import pre_processar_dados
from utils.print_customizado import cprint
from processamento.ler_dataset_processado import ler_datasets

from warnings import filterwarnings; filterwarnings("ignore", category="ConvergenceWarning") # Filtra os warnings do MLP




class MetodosAprendizado:
    def __init__(self):
        self.modo_teste = False

        self.modelos = {}


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
        modelos = {}
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
                resultados[nome_metodo], modelos[nome_metodo] = metodo(**kwargs, callback=callback)
            else:
                resultados[nome_metodo], modelos[nome_metodo] = metodo(**kwargs)
        
        return resultados, modelos
    
    def split_dataset(self, dados : DataFrame, tam_treino=0.5, tam_teste=0.25, tam_validacao=0.25, seed=None):
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
        
        x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, train_size=tam_treino, test_size=1-tam_treino, random_state=seed)

        x_teste, x_val, y_teste, y_val = train_test_split(x_temp, y_temp, train_size=tam_teste, test_size=tam_validacao, random_state=seed)

        return x_treino, y_treino, x_teste, y_teste, x_val, y_val

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
                opiniao = KNN.predict(x_val) # Valida no conjunto de validação
                Acc = accuracy_score(y_val, opiniao)
                if (Acc > maiorAcc):
                    maiorAcc = Acc
                    melhor_modelo = KNN

        print()
        cprint(f"Melhor configuração encontrada na validação:", label="KNN")
        cprint(f"K: {melhor_modelo.n_neighbors} , Weights: {melhor_modelo.weights}, Acc: {maiorAcc}", label="KNN")

        
        """**Aplicando o melhor modelo sobre o conjunto de teste**"""

        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)

        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="KNN")

        return acuracia_teste, melhor_modelo
    

    def metodo_arvoreDecisao(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        
        cprint("Executando a Árvore de decisão...", label="AD")

        # Definido os ranges
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
        acuracia = accuracy_score(y_teste, opiniao)
        cprint(f"Acurácia sobre o teste: {acuracia}", label="AD")
        
        return acuracia, melhor_modelo
    
    def metodo_naiveBayes(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando o Naive Bayes...", label="NB")

        melhor_modelo = GaussianNB()
        melhor_modelo.fit(x_treino, y_treino)

        # Validação (embora GNB não tenha muitos hiperparâmetros, seguimos o padrão)
        opiniao_val = melhor_modelo.predict(x_val)
        acuracia_val = accuracy_score(y_val, opiniao_val)
        cprint(f"Acurácia sobre a validação: {acuracia_val}", label="NB")

        # Teste
        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="NB")

        return acuracia_teste, melhor_modelo

    def metodo_svm(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando o SVM...", label="SVM")

        # Buscar:
        # ● Tipo de kernel (linear, polinomial, RBF, sigmoid)
        # ● Parâmetro de regularização (C) (Valor do erro)

        kernels = ['linear'] if self.modo_teste else ['linear', 'poly', 'rbf', 'sigmoid']

        # Valores comuns para C em escala logarítimica, revisar
        c_range = [1] if self.modo_teste else [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500,1000]   

        maior = -1

        for kernel in tqdm(kernels, ascii=True, leave=False): # itera buscando o melhor kernel
            for C in tqdm(c_range, ascii=True): # itera buscando o melhor C para o kernel atual
                cprint(f"Testando kernel='{kernel}', C={C}...", label="SVM")
                SVM = SVC(kernel=kernel, C=C)
                SVM.fit(x_treino, y_treino) # treina
                opiniao = SVM.predict(x_val) # valida no conjunto de validação
                acuracia = accuracy_score(y_val, opiniao) 

                # Se o modelo foi melhor do que o melhor até agora, salva ele
                if acuracia > maior:
                    cprint(f"Novo melhor Kernel: {kernel}, C: {C}, Acurácia sobre a validação: {acuracia}", label="SVM")
                    maior = acuracia
                    melhor_modelo = SVM

        # Salva na classe o melhor modelo encontrado
        self.modelos["SVM"] = melhor_modelo

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""
        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="SVM")

        # Retorna o melhor resultado encontrado no teste
        return acuracia_teste, melhor_modelo

    def metodo_randomForest(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando a Random Forest...", label="RF")

        # ● Número de estimadores 
        # ● Critério de seleção dos atributos 
        # ● Profundidade máxima 
        # ● Número mínimo para realizar a divisão do nó 
        # ● Número mínimo de elementos por folha

        i_range = range(10,100,5)

        if self.modo_teste:
            i_range = range(10,20,5)

        if self.modelos.get("AD") is None:
            cprint("Executando a Árvore de decisão para obter os melhores hiperparâmetros...", label="RF")
            self.metodo_arvoreDecisao(x_treino, y_treino, x_teste, y_teste, x_val, y_val)
        modelo = self.modelos["AD"]

        melhor_modelo = modelo
        maior = -1

        for i in tqdm(i_range, ascii=True, desc=f"[ RF ] n_estimators =  "):      #n_estimators
            RandomForest = RandomForestClassifier(n_estimators=i, criterion=modelo.criterion, max_depth=modelo.max_depth, min_samples_split=modelo.min_samples_split, min_samples_leaf=modelo.min_samples_leaf)
            RandomForest.fit(x_treino, y_treino)
            opiniao = RandomForest.predict(x_val)
            acuracia = accuracy_score(y_val, opiniao)

            if acuracia > maior:
                melhor_modelo = RandomForest
                cprint(f"Acurácia sobre a validação melhorou: {acuracia} > {maior}", label="RF")
                maior = acuracia

        # Salva na classe o melhor modelo encontrado
        self.modelos["RF"] = melhor_modelo

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""
        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="RF")

        return acuracia_teste, melhor_modelo


    def metodo_mlp(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):

        cprint("Executando o Multi-Layer Perceptron...", label="MLP")
        
        # Buscar:
        # ● Número de épocas de treino
        # ● Taxa de aprendizagem
        # ● Número de camadas escondidas
        # ● Quantidade de neurônios em cada camada escondida
        # ● Função de ativação
        # ● Tamanho do batch

        # Definido os ranges
        epocas_range = [30, 50, 100, 200, 500]
        lr_range = [0.0001, 0.001, 0.01, 0.1] # Learning rate
        num_camadas_escondidas_range = [2] # Mais que 2 é dar overclock no mouse
        neuronios_por_camada_range = [44,55,66,77,88] # Deve estar entre o numero de atributos e o seu dobro
        funcao_ativacao_range = ['relu', 'tanh', 'logistic'] # logistic é bom pra classificação binaria
        tamanho_batch_range = [32, 64, 128, 256]

        # Altera os ranges caso modo teste
        if self.modo_teste:
            epocas_range = [100] # testa só 100 epocas
            lr_range = [0.1] # Learning rate
            num_camadas_escondidas_range = [1] # Mais que 2 é dar overclock no mouse
            neuronios_por_camada_range = [4] # Deve estar entre o numero de neuronios de entrada e o de saida
            funcao_ativacao_range = ['logistic'] # logistic é bom pra classificação binaria
            tamanho_batch_range = [256]

        cprint("Fazendo busca de hiperparametros...", label="MLP")

        maior = -1
        for epoca in tqdm(epocas_range, ascii=True, leave=False):
            for tamanho_batch in tqdm(tamanho_batch_range, ascii=True):
                for taxa_aprendizado in lr_range:
                    for num_camadas in num_camadas_escondidas_range:
                        for num_neuronios_1 in neuronios_por_camada_range:
                            camadas_iter = neuronios_por_camada_range if num_camadas == 2 else [None]
                            for num_neuronios_2 in camadas_iter:
                                config_camadas = (num_neuronios_1,) if num_camadas == 1 else (num_neuronios_1, num_neuronios_2)
                                for func_ativacao in funcao_ativacao_range:
                                

                                    MLP = MLPClassifier(
                                        max_iter=epoca,
                                        batch_size=tamanho_batch,
                                        hidden_layer_sizes=config_camadas,
                                        learning_rate_init=taxa_aprendizado,
                                        activation=func_ativacao
                                    )
                                                    
                                    MLP.fit(x_treino,y_treino)
                                    opiniao = MLP.predict(x_val)
                                    Acc = accuracy_score(y_val, opiniao)
                                    # print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
                                    if (Acc > maior):
                                        #cprint(f"Nova melhor configuração encontrada: {Acc}", label="AD")
                                        maior = Acc
                                        melhor_modelo = MLP

        cprint("\nMelhor configuração para a MLP", label="MLP")
        # cprint(f"Criterion: {melhor_modelo.}, max_depth: {melhor_modelo.max_depth}, min_samples_leaf: {melhor_modelo.min_samples_leaf}, min_samples_split: {melhor_modelo.min_samples_split}, splitter: {melhor_modelo.splitter}, Acc: {maior}", label="MLP")

        # Salvando melhor modelo
        self.modelos["MLP"] = melhor_modelo

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""

        opiniao = melhor_modelo.predict(x_teste)

        acuracia = accuracy_score(y_teste, opiniao)
        cprint(f"Acurácia sobre o teste: {acuracia}", label="MLP")
        
        return acuracia, melhor_modelo




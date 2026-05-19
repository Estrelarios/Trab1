import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import inspect
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
filterwarnings("ignore", category=ConvergenceWarning) # Filtra os warnings do MLP

from utils.print_customizado import cprint
from processamento.ler_dataset_processado import ler_datasets

def um_menos_dist_norm(distances):
    # distances shape: (n_amostras, n_vizinhos)
    # Pegamos a maior distância de cada linha para normalizar
    max_dist = np.max(distances, axis=1, keepdims=True)

    # Criamos os pesos: 1 - (dist/max_dist)
    # Usamos np.where para evitar divisão por zero caso a maior distância seja 0
    # (isso acontece se todos os vizinhos estiverem no mesmo local do ponto de consulta)
    weights = np.where(max_dist > 0, 1 - (distances / max_dist), 1.0)

    # Garante que não existam pesos negativos e que a soma nunca seja zero
    weights = np.clip(weights, 0, 1)
    weights[np.all(weights <= 1e-10, axis=1)] = 1.0

    return weights

class MetodosAprendizado:
    def __init__(self):
        self.modo_teste = False
        self.modelo_AD = None


    def disparar_comando(self, parametros: dict = None, lista_tecnicas: list = None, modelos_carregados: dict = None):
        """
        Args:
            parametros (dict): Dicionário com todos os parâmetros disponíveis para os métodos.
            lista_tecnicas (list): Lista de strings com os nomes dos métodos a serem executados (ex: ["metodo_knn"]).
                                  Se None, executa todos os que começam com "metodo_".
            modelos_carregados (dict): Dicionário de modelos pré-treinados para uso em métodos de combinação.
        """
    
        if parametros is None:
            parametros = {}

        if "modo_teste" in parametros and parametros["modo_teste"]:
            self.modo_teste = True
        
        resultados = {}
        modelos = {}

        # Define quais métodos executar
        if lista_tecnicas is None:
            metodos_nomes = [m for m in dir(self) if m.startswith("metodo_") and callable(getattr(self, m))]
        else:
            metodos_nomes = lista_tecnicas

        for nome_metodo in metodos_nomes:
            # Verifica se o metodo exsite na classe
            if not hasattr(self, nome_metodo):
                cprint(f"Aviso: Método {nome_metodo} não encontrado na classe.", label="MA")
                continue

            metodo = getattr(self, nome_metodo)
            
            # Descobre quais parâmetros o método precisa (excluindo 'self')
            assinatura = inspect.signature(metodo)
            params_necessarios = list(assinatura.parameters.keys())
            
            # Prepara os argumentos para o método
            kwargs = {}
            for p in params_necessarios:
                if p == "self": continue
                if p == "modelos_carregados":
                    kwargs[p] = modelos_carregados
                elif p in parametros:
                    kwargs[p] = parametros[p]
            
            try:
                # Chama o método e armazena resultados
                res, mod = metodo(**kwargs)
                resultados[nome_metodo] = res
                modelos[nome_metodo] = mod
            
            except Exception as e: 
                cprint(f"Erro ao executar {nome_metodo}: {str(e)}", label="ERRO")
                import traceback
                traceback.print_exc()
        
        return resultados, modelos
    
    def carregar_modelos(self, iteracao, teste=False):
        """Carrega todos os modelos .joblib da iteração especificada.
        
        Returns:
            modelos_lidos (dict): { "knn": modelo, "arvoreDecisao": modelo, ... }
        """
        import joblib as jb
        import os

        dir_modelos = "modelos/teste/" if teste else "modelos/"
        modelos_lidos = {}

        if not os.path.exists(dir_modelos):
            cprint(f"Diretório de modelos não encontrado: {dir_modelos}", label="MA")
            return modelos_lidos

        prefixo = f"{iteracao:02}_"
        arquivos = [f for f in os.listdir(dir_modelos) if f.startswith(prefixo) and f.endswith(".joblib")]

        if not arquivos:
            cprint(f"Nenhum modelo encontrado para a iteração {iteracao} em {dir_modelos}", label="MA")
            return modelos_lidos

        cprint(f"Carregando {len(arquivos)} modelos da iteração {iteracao}...", label="MA")
        for arq in arquivos:
            nome_tecnica = arq.replace(prefixo, "").replace(".joblib", "")

            # só carrega modelos mono
            if nome_tecnica in ["randomForest", "bagging", "boosting", "combSoma", "combMajoritaria", "combBordaCount"]:
                continue

            caminho = os.path.join(dir_modelos, arq)
            try:
                modelos_lidos[nome_tecnica] = jb.load(caminho)
            except Exception as e:
                cprint(f"Falha ao carregar {arq}: {e}", label="MA")

        return modelos_lidos

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
        
        x_treino, x_temp, y_treino, y_temp = train_test_split(X, Y, train_size=tam_treino, test_size=1-tam_treino, random_state=seed, stratify=Y)

        x_teste, x_val, y_teste, y_val = train_test_split(x_temp, y_temp, train_size=tam_teste, test_size=tam_validacao, random_state=seed, stratify=y_temp)

        return x_treino, y_treino, x_teste, y_teste, x_val, y_val

    def metodo_knn(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
                
        cprint("Executando o KNN...", label="KNN")

        # Escalonamento prévio
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)

        # Definido o range (reduzido para modo teste)
        # O KNN não tem a função nativa de 1 - distancia normalizada para os pesos
        # Mas permite que seja inserida uma função customizada para isso
        tipos_pesamento = ["distance", "uniform", um_menos_dist_norm]
        k_range = range(1,51)
        if self.modo_teste:
            k_range = range(1,2)
        
        cprint("Fazendo busca de hiperparametros...", label="KNN")
        maiorAcc = -1
        melhor_k = 1
        melhor_weights = "uniform"

        for j in tipos_pesamento:
            for i in tqdm(k_range, desc=f"weights='{j}'"):
                KNN = KNeighborsClassifier(n_neighbors=i, weights=j, n_jobs=3)
                KNN.fit(x_treino_s, y_treino)
                opiniao = KNN.predict(x_val_s) # Valida no conjunto de validação
                Acc = accuracy_score(y_val, opiniao)
                if (Acc > maiorAcc):
                    maiorAcc = Acc
                    melhor_k = i
                    melhor_weights = j

        # Pipeline final
        melhor_modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=melhor_k, weights=melhor_weights, n_jobs=3))
        ])
        melhor_modelo.fit(x_treino, y_treino)

        print()
        cprint(f"Melhor configuração encontrada na validação:", label="KNN")
        cprint(f"K: {melhor_k} , Weights: {melhor_weights}, Acc: {maiorAcc}", label="KNN")

        
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

        # Salva o melhor modelo para o RandomForest
        self.modelo_AD = melhor_modelo
        
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

        # Escalonamento manual antes do loop para performance
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)

        # Buscar:
        # ● Tipo de kernel (linear, polinomial, RBF, sigmoid)
        # ● Parâmetro de regularização (C) (Valor do erro)

        kernels = ['linear'] if self.modo_teste else ['linear', 'poly', 'rbf', 'sigmoid']

        # Valores comuns para C em escala logarítimica, revisar
        c_range = [1] if self.modo_teste else [0.001, 0.01, 0.1, 1, 10, 100, 1000]   

        maior = -1
        melhor_kernel = "linear"
        melhor_C = 1

        for kernel in tqdm(kernels, ascii=True, leave=False): # itera buscando o melhor kernel
            for C in tqdm(c_range, ascii=True): # itera buscando o melhor C para o kernel atual
                cprint(f"Testando kernel='{kernel}', C={C}...", label="SVM")
                SVM = SVC(kernel=kernel, C=C, probability=True)
                SVM.fit(x_treino_s, y_treino) # treina
                opiniao = SVM.predict(x_val_s) # valida no conjunto de validação
                acuracia = accuracy_score(y_val, opiniao) 

                # Se o modelo foi melhor do que o melhor até agora, salva ele
                if acuracia > maior:
                    cprint(f"Novo melhor Kernel: {kernel}, C: {C}, Acurácia sobre a validação: {acuracia}", label="SVM")
                    maior = acuracia
                    melhor_kernel = kernel
                    melhor_C = C

        # Cria o modelo final com Pipeline para incluir o scaler
        melhor_modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=melhor_kernel, C=melhor_C, probability=True)),
        ])
        melhor_modelo.fit(x_treino, y_treino)

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""
        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="SVM")

        # Retorna o melhor resultado encontrado no teste
        return acuracia_teste, melhor_modelo

    def metodo_mlp(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):

        cprint("Executando o Multi-Layer Perceptron...", label="MLP")
        
        # Escalonamento manual antes do loop para performance
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)

        # Buscar:
        # ● Número de épocas de treino
        # ● Taxa de aprendizagem
        # ● Quantidade de neurônios em cada camada escondida
        # ● Função de ativação
        # ● Tamanho do batch

        # Definido os ranges
        epocas_range = [30, 50, 100, 200, 500]
        taxa_aprendizado_range = [0.0001, 0.001, 0.01, 0.1] # Learning rate
        neuronios_por_camada_range = [44,55,66,77,88] # Deve estar entre o numero de atributos e o seu dobro
        funcao_ativacao_range = ['relu', 'tanh', 'logistic'] # logistic é bom pra classificação binaria
        tamanho_batch_range = [32, 64, 128, 256]

        # Altera os ranges caso modo teste
        if self.modo_teste:
            epocas_range = [100] # testa só 100 epocas
            taxa_aprendizado_range = [0.1] # Learning rate
            neuronios_por_camada_range = [4] # Deve estar entre o numero de neuronios de entrada e o de saida
            funcao_ativacao_range = ['logistic'] # logistic é bom pra classificação binaria
            tamanho_batch_range = [256]

        cprint("Fazendo busca de hiperparametros...", label="MLP")

        maior = -1
        melhor_modelo_base = None

        for epoca in tqdm(epocas_range, ascii=True, leave=True, desc="num epocas"):
            for tamanho_batch in tqdm(tamanho_batch_range, ascii=True, desc="tamanho batch"):
                for taxa_aprendizado in taxa_aprendizado_range:
                    for num_neuronios in neuronios_por_camada_range:
                        for func_ativacao in funcao_ativacao_range:
                            
                            MLP = MLPClassifier(
                                max_iter=epoca,
                                batch_size=tamanho_batch,
                                hidden_layer_sizes=(num_neuronios,),
                                learning_rate_init=taxa_aprendizado,
                                activation=func_ativacao
                            )
                                            
                            MLP.fit(x_treino_s, y_treino)
                            opiniao = MLP.predict(x_val_s)
                            Acc = accuracy_score(y_val, opiniao)
                            # print("Criterion: ",j," max_depth: ",i," min_samples_leaf: ",k," min_samples_split: ",l," splitter: ",m," Acc: ",Acc)
                            if (Acc > maior):
                                #cprint(f"Nova melhor configuração encontrada: {Acc}", label="AD")
                                maior = Acc
                                melhor_modelo_base = MLP

        cprint("\nMelhor configuração para a MLP", label="MLP")
        # cprint(f"Criterion: {melhor_modelo.}, max_depth: {melhor_modelo.max_depth}, min_samples_leaf: {melhor_modelo.min_samples_leaf}, min_samples_split: {melhor_modelo.min_samples_split}, splitter: {melhor_modelo.splitter}, Acc: {maior}", label="MLP")

        # Cria o modelo final com Pipeline para incluir o scaler
        melhor_modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', melhor_modelo_base)
        ])
        melhor_modelo.fit(x_treino, y_treino)

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""
        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="MLP")
        
        return acuracia_teste, melhor_modelo

    def metodo_randomForest(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        cprint("Executando a Random Forest...", label="RF")

        # ● Número de estimadores 
        # ● Critério de seleção dos atributos 
        # ● Profundidade máxima 
        # ● Número mínimo para realizar a divisão do nó 
        # ● Número mínimo de elementos por folha
        
        # Definido os ranges
        n_estimadores_range = [10, 50, 100]
        profundidade_range = range(1, 11)
        min_divisao_nos_range = range(2, 11)
        min_elementos_folha_range = range(1, 11)

        # Altera os ranges para modo teste
        if self.modo_teste:
            n_estimadores_range = [10]
            profundidade_range = range(1, 2)
            min_divisao_nos_range = range(2, 3)
            min_elementos_folha_range = range(1, 2)

        cprint("Fazendo busca de hiperparametros...", label="RF")

        maior = -1
        melhor_modelo = None

        for criterion in ("entropy", "gini"):
            for n_estimadores in tqdm(n_estimadores_range, ascii=True, desc=f"[ RF ] criterion = {criterion}"):
                for profundidade in tqdm(profundidade_range, desc=f"[ RF ] n_estimadores = {n_estimadores}", leave=False, ascii=True):
                    for min_elementos_folha in min_elementos_folha_range:
                        for min_divisao_nos in min_divisao_nos_range:
                            RF = RandomForestClassifier(
                                n_estimators=n_estimadores,
                                criterion=criterion,
                                max_depth=profundidade,
                                min_samples_leaf=min_elementos_folha,
                                min_samples_split=min_divisao_nos,
                                n_jobs=3,
                            )


                            RF.fit(x_treino, y_treino)
                            opiniao = RF.predict(x_val)
                            Acc = accuracy_score(y_val, opiniao)
                            
                            if (Acc > maior):
                                maior = Acc
                                melhor_modelo = RF

        cprint("\nMelhor configuração para a RF", label="RF")
        cprint(f"Criterion: {melhor_modelo.criterion}, n_estimadores: {melhor_modelo.n_estimators}, profundidade: {melhor_modelo.max_depth}, min_samples_leaf: {melhor_modelo.min_samples_leaf}, min_samples_split: {melhor_modelo.min_samples_split}, Acc: {maior}", label="RF")

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""
        opiniao_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="RF")

        return acuracia_teste, melhor_modelo

    def metodo_bagging(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):

        """
        Lembrar:

        Para escolher os subconjuntos, deve-se fazer seleção aleatorio com reposição:
            • Um mesmo elemento pode ser escolhido várias vezes
            para um mesmo conjunto
            • Um mesmo elemento pode ser escolhido para
            diferentes subconjuntos

        """

        cprint("Executando o Bagging...", label="Bagging")

        # Escalonamento prévio para os estimadores que precisam (KNN, SVM, MLP)
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)
        x_teste_s = scaler.transform(x_teste)

        # Buscar
        # ● Número de estimadores
        # ● Número de amostras para cada subconjunto (0.1 0.5 0.7 1.0)
        # ● Estimadores (os classificadores empregados terão seus hiperparâmetros setados com os valores default)

        n_estimadores_range = [50, 100, 500] 
        n_amostras_range = [0.1, 0.5, 0.7, 1.0] 
        estimador_range = [KNeighborsClassifier, SVC, DecisionTreeClassifier, MLPClassifier, GaussianNB]
        # estimador_range = [MLPClassifier]

        if self.modo_teste:
            n_estimadores_range = [1] 
            n_amostras_range = [0.1] 
            estimador_range = [KNeighborsClassifier, SVC, DecisionTreeClassifier, MLPClassifier, GaussianNB]

        cprint("Fazendo busca de hiperparametros...", label="Bagging")

        maiorAcc = -1
        melhor_modelo = None

        for classe_base in tqdm(estimador_range, desc="Estimadores", ascii=True):
            
            # Esconde o MLPClassifier dentro de um Pipeline
            if classe_base == MLPClassifier:
                n_estimadores_range_uso = [5, 10, 15] 
            else:
                n_estimadores_range_uso = n_estimadores_range 
            
            for n_est in tqdm(n_estimadores_range_uso, desc=f"n_estimators ({classe_base.__name__})", leave=False, ascii=True):
                for n_samp in n_amostras_range:
           
                    # Cria o Bagging
                    bag = BaggingClassifier(estimator=classe_base(), n_estimators=n_est, max_samples=n_samp, n_jobs=3)

                    bag.fit(x_treino_s, y_treino)
                    opiniao = bag.predict(x_val_s)
                    Acc = accuracy_score(y_val, opiniao)
                    
                    if (Acc > maiorAcc):
                        maiorAcc = Acc
                        melhor_modelo = bag

        if melhor_modelo is None:
            cprint("Aviso: Nenhum modelo de Bagging foi gerado.", label="Bagging")
            return 0, None

        # Identifica o nome do melhor estimador base para o log
        nome_base = type(getattr(melhor_modelo, 'estimator', getattr(melhor_modelo, 'base_estimator', None))).__name__

        cprint("\nMelhor configuração para o Bagging", label="Bagging")
        cprint(f"Estimador: {nome_base}, n_estimators: {melhor_modelo.n_estimators}, max_samples: {melhor_modelo.max_samples}, Acc: {maiorAcc}", label="Bagging")

        """**Aplicando o melhor modelo sobre o conjunto de teste**"""
        opiniao_teste = melhor_modelo.predict(x_teste_s)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="Bagging")

        # Pipeline final para salvar o scaler junto
        melhor_modelo_final = Pipeline([
            ('scaler', scaler),
            ('bagging', melhor_modelo)
        ])

        return acuracia_teste, melhor_modelo_final


    def metodo_boosting(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):

        cprint("Executando o Boosting (AdaBoost)...", label="Boosting")

        # Escalonamento prévio para os estimadores que precisam (KNN, SVM, MLP)
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)
        x_teste_s = scaler.transform(x_teste)

        # Buscar:
        # ● Número de estimadores (50 100 500)
        # ● Taxa de aprendizado (0.01 0.1 1.0)
        # ● Estimadores (os classificadores empregados terão seus hiperparâmetros setados com os valores default)

        n_estimadores_range = [50, 100, 500] 
        learning_rate_range = [0.01, 0.1, 1.0]
        
        # Nota: AdaBoost requer que o estimador base suporte pesos (sample_weight).
        # DecisionTree, GaussianNB e SVC suportam. KNN e MLP não suportam no scikit-learn.
        estimador_range = [DecisionTreeClassifier, GaussianNB, SVC] 

        if self.modo_teste:
            n_estimadores_range = [10] 
            learning_rate_range = [0.1] 
            estimador_range = [DecisionTreeClassifier]

        cprint("Fazendo busca de hiperparametros...", label="Boosting")

        maiorAcc = -1
        melhor_modelo = None

        for classe_base in tqdm(estimador_range, desc="Estimadores", ascii=True):
            # Define se esta classe base precisa de dados escalonados
            precisa_escalonar = classe_base in [SVC]
            x_treino_uso = x_treino_s if precisa_escalonar else x_treino
            x_val_uso = x_val_s if precisa_escalonar else x_val

            for n_est in tqdm(n_estimadores_range, desc=f"n_estimators ({classe_base.__name__})", leave=False, ascii=True):
                for lr in learning_rate_range:
                    # Instancia o classificador base com valores default
                    base = classe_base()
                    
                    # AdaBoostClassifier
                    boost = AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr)
                    
                    boost.fit(x_treino_uso, y_treino)
                    opiniao = boost.predict(x_val_uso)
                    Acc = accuracy_score(y_val, opiniao)
                    
                    if (Acc > maiorAcc):
                        maiorAcc = Acc
                        melhor_modelo = boost
                        modelo_precisou_escalonar = precisa_escalonar

        if melhor_modelo is None:
            cprint("Aviso: Nenhum modelo de Boosting foi gerado.", label="Boosting")
            return 0, None

        # Identifica o nome do melhor estimador base para o log
        nome_base = type(getattr(melhor_modelo, 'estimator', getattr(melhor_modelo, 'base_estimator', None))).__name__

        cprint("\nMelhor configuração para o Boosting", label="Boosting")
        cprint(f"Estimador: {nome_base}, n_estimators: {melhor_modelo.n_estimators}, learning_rate: {melhor_modelo.learning_rate}, Acc: {maiorAcc}", label="Boosting")

        """**Aplicando o melhor modelo sobre o conjunto de teste**"""
        x_teste_uso = x_teste_s if modelo_precisou_escalonar else x_teste
        opiniao_teste = melhor_modelo.predict(x_teste_uso)
        acuracia_teste = accuracy_score(y_teste, opiniao_teste)
        cprint(f"Acurácia sobre o teste: {acuracia_teste}", label="Boosting")

        # Se o modelo precisou de escalonamento, retorna um Pipeline com o scaler
        if modelo_precisou_escalonar:
            melhor_modelo_final = Pipeline([
                ('scaler', scaler),
                ('boosting', melhor_modelo)
            ])
        else:
            melhor_modelo_final = melhor_modelo

        return acuracia_teste, melhor_modelo_final

    def metodo_combSoma(self, x_treino, x_teste, y_treino, y_teste, modelos_carregados : dict):

        # modelos_carregados tem a estrutura { "knn": modelo, "arvoreDecisao": modelo, ... }
        cprint("Executando o combinação majoritária ...", label="SOMA")

        estimators = []
        soma_probs = None
        for label, modelo in modelos_carregados.items():

            probs = modelo.predict_proba(x_teste)

            # Vai somando as probabilidades
            if soma_probs is None:
                soma_probs = probs
            else:
                soma_probs += probs

        opiniao = []
        # Para cada amostra pega o vitorioso
        # argmax([x,y]) verifica qual é maior (x ou y) e retorna seu indice 
        for amostra_prob in soma_probs:
            opiniao.append(np.argmax(amostra_prob))

        acuracia = accuracy_score(y_teste, opiniao)
        cprint(f"Acurácia sobre o teste: {acuracia}", label="SOMA")

        return acuracia, None
      
    def metodo_combMajoritaria(self, x_treino, x_teste, y_treino, y_teste, modelos_carregados : dict):

        cprint("Executando o combinação majoritária ...", label="MAJOR")

        estimators = []
        probs = []
        for label, modelo in modelos_carregados.items():
            # add todas as probabilidade em uma lista só
            # probs terá o formato [modelo1[probs], modelo2[probs]...] 
            probs.append(modelo.predict_proba(x_teste))

        opiniao = []
        # Para cada amostra
        for i in range(len(probs[0])): 
            
            # votacao será um int
            # se for votado no pokemon 1, votacao--. Se for votado no 2, votacao++
            # ao final, se votacao < 0, pokemon 1 foi o mais votado.
            # se votacao > 0, pokemon 2 foi o mais votado
            votacao = 0
            # para cada modelo
            for j in range(len(probs)):
                
                # confiança de cada modelo (prob. de ser winner pokemon 1 ou 2 )
                p_1 = probs[j][i][0]
                p_2 = probs[j][i][1]
                
                if p_1 > p_2:
                    votacao -= 1
                else:
                    votacao += 1 

            if votacao < 0: # 1 foi mais votado
                opiniao.append(0)
            elif votacao > 0: # 2 foi mais votado
                opiniao.append(1)
            else:
                # Erro, votação deu 0
                cprint("Erro: alguma votação tem valor 0. Isso significa que ou ninguém votou ou há um número igual de votos para o pokemon 1 e para o 2, o que significa que alguém não votou.", label="MAJOR")
            
        acuracia = accuracy_score(y_teste, opiniao)
        cprint(f"Acurácia sobre o teste: {acuracia}", label="MAJOR")

        return acuracia, None 

    # def metodo_combBordaCount(self, x_teste, y_teste, modelos_carregados : dict[str, KNeighborsClassifier]):

    #     prob_estimators = {}
        
    #     primeiro_pokemon = 0 # primeira posicao da tupla
    #     segundo_pokemon = 0 # segunda posicao da tupla

    #     # Dict para guardar probabilidades dos estimadores
    #     # { "knn" : [p_0, p_1] } 0 = pokemon 1 venceu, 1 = pokemon 2 venceu

    #     # Matriz para guardar as probabilidades
    #     # [knn,AD,...]
    #     # amostra1 -> [p_1,p_1, p_1]
    #     # amostra2 -> ...

    #     for label, modelo in modelos_carregados.items():
    #             prob_estimators[label] = modelo.predict_proba(x_teste)

    #     for amostra_i in range(len(x_teste)):

    #         modelos =  {}

    #         for label, batalhas in prob_estimators.items():

    #             # para cada batalha da amostra_i, pega a probabilidade de cada pokemon vencer
    #             prob_batalha = batalhas[amostra_i] # [p_0, p_1]
                
                
            

            
    #             # pega o ranking baseado na opinião de cada modelo
    #             # faz o teste de qm tem mais confiança
    #             # salva
            
    #         # prob_estimators[knn] = probs_knn
    #         # prob_estimators[svm] = probs_svm

    #     # Ter a seguinte estrutra:

    #     # Probs = [Modelo[Amostra[P_0, P_1]]]
    #     # Probs = [
    #     #          [ # KNN
    #     #           [0.85, 0.15], # amostra 1
    #     #           [0.5, 0.5], # amostra 2
    #     #           [P0, P1], # amostra n
    #     #           ...
    #     #          ],

    #     #          [ # SVM
    #     #          ... 
    #     #          ] #
    #     #

    #     # Para cada amostra
    #     for amostra_i in range(len(x_teste)):

    #         # para cada prob_estimators:
                


            
    #     # testa as opiniões contra y_teste
    #     # pega a acuracia do borda count
    #     # return acuracia

    #     for label, modelo in modelos_carregados.items(): # salva a probabilidade de cada modelo 
            
            
    #         prob_estimators[label] = modelo.predict_proba(x_teste)

    #         # prob_estimators["knn"] = [
    #         # [0.85, 0.15],  # Batalha 1
    #         # [0.30, 0.70],  # Batalha 2
    #         # [p_0, p_1]   # Batalha n
    #         # ]

    #         # Para cada amostra do x_teste, precisamos pegar a probabilidade que cada modelo
    #         # prevê e fazer um ranking reverso (menor probabilidade ficará em primeiro e
    #         # maior probabilidade receberá o 2º lugar) 

    #         opiniao = {}
    #         opiniao_primeiro = 0
    #         opiniao_segundo = 0
    
    #         for prob in prob_estimators[label]: # itera pelas 5000 batalhas de cada modelo, que é capturado pela key do dict
                
    #             p_0, p_1 = prob

    #             if p_0 > p_1:
    #                 opiniao_primeiro += 2
    #                 opiniao_segundo += 1
    #             elif p_1 > p_0:
    #                 opiniao_segundo += 2
    #                 opiniao_primeiro += 1

    #             previsao_knn = [1,0]
                

    #             # resulta em  2 valores, a pontuação do primeiro pokemon e a pontuação do segundo pokemon, para cada modelo
    #             # Salva como knn : [pontuação primeiro pokemon, pontuação segundo pokemon]

    #             opiniao[label] = [opiniao_primeiro, opiniao_segundo] 

    #             opiniao = [1,0,1,1,0,1]
    #             y_teste

    #         opiniao.append(classe_preditiva)

    #     for  modelo in prob_estimators.items():

    #         borda = prob_estimators[modelo]

    #         if borda[0] > borda[1]:
    #             primeiro_pokemon += 1
    #         elif borda[1] > borda[0]:
    #             segundo_pokemon += 1

    #     return acuracia_teste, melhor_modelo
    
# Boa tarde professor, tenho uma duvida perante o trabalho 1 de AM
# Meu dataset é o de batalhas pokemon, onde o objetivo é prever o vencedor da batalha com base nas estatísticas dos pokemons.
# Minha duvida reside na implementação dos métodos de combinação de classificadores devido ao seguinte impasse:
# As classes utilizadas para o treinamento não contém por si mesmas a resposta que buscamos, qual pokemon dentre os 2 tem a maior probabilidade de vencer a batalha.
# Desta forma, como implementamos o borda count? Podemos atribuir ranks para cada uma das estatísticas dos pokemons, mas não vejo utilidade nisso, visto que tais não contém a resposta que buscamos.
# Tentamos implementar o borda count utilizando as probabilidades de cada modelo para cada pokemon, ou seja, cada modelo atribui uma probabilidade de vitória para cada pokemon, e o pokemon com a maior probabilidade recebe um ponto. 
# No entanto, isso se assemelha mais a uma votação majoritária do que ao borda count tradicional, Além de não ser possível objetificar o que mandar para a instância de teste, como escrito no slide "A classe que possuir o maior somatório de ranks é atribuída à instância de teste".
# Não sabemos como encaixar essa situação no borda count
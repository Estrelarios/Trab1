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

    def split_dataset(self, dados, tam_treino=0.5, tam_teste=0.25, tam_validacao=0.25, seed=None):
        """
        Divide o conjunto de dados em subconjuntos de treinamento, teste e validação 
        mantendo a proporção original das classes (estratificação).

        Args:
            dados (pd.DataFrame): DataFrame contendo as instâncias e o alvo (Winner).
            tam_treino (float): Proporção destinada ao conjunto de treinamento (default 50%).
            tam_teste (float): Proporção destinada ao conjunto de teste (default 25%).
            tam_validacao (float): Proporção destinada ao conjunto de validação (default 25%).
            seed (int): Semente para reprodutibilidade da divisão aleatória.

        Returns:
            tuple: (x_treino, y_treino, x_teste, y_teste, x_val, y_val)
        """

        # Cálculo das proporções relativas para a segunda divisão
        total_restante = tam_teste + tam_validacao
        prop_teste = tam_teste / total_restante
        prop_val = tam_validacao / total_restante

        X = dados.iloc[:, 1:]
        Y = dados.iloc[:, 0]
        
        x_treino, x_temp, y_treino, y_temp = train_test_split(
            X, Y, 
            train_size=tam_treino, 
            test_size=1-tam_treino, 
            random_state=seed, 
            stratify=Y
        )

        x_teste, x_val, y_teste, y_val = train_test_split(
            x_temp, y_temp, 
            train_size=prop_teste, 
            test_size=prop_val, 
            random_state=seed, 
            stratify=y_temp
        )

        return x_treino, y_treino, x_teste, y_teste, x_val, y_val

    def metodo_knn(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação do classificador K-Nearest Neighbors (KNN) com busca de hiperparâmetros.
        Avalia diferentes valores de K e estratégias de ponderação (uniforme, distância e 
        1 - distância normalizada) sobre o conjunto de validação.
        """
        cprint("Executando o KNN...", label="KNN")

        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)

        # Definido o range (reduzido para modo teste)
        # O KNN não tem a função nativa de 1 - distancia normalizada para os pesos
        # Mas permite que seja inserida uma função customizada para isso
        tipos_pesamento = ["distance", "uniform", um_menos_dist_norm]
        k_range = range(1, 51)
        if self.modo_teste:
            k_range = range(1, 2)
        
        cprint("Fazendo busca de hiperparâmetros...", label="KNN")
        maior_acc = -1
        melhor_k = 1
        melhor_weights = "uniform"

        for weights in tipos_pesamento:
            desc_weights = weights if isinstance(weights, str) else "custom" # Apenas pra printar o nome bonitniho
            for k in tqdm(k_range, desc=f"weights='{desc_weights}'"):
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=3)
                knn.fit(x_treino_s, y_treino)
                pred_val = knn.predict(x_val_s)
                acc = accuracy_score(y_val, pred_val)
                if acc > maior_acc:
                    maior_acc = acc
                    melhor_k = k
                    melhor_weights = weights

        # Pipeline para salvar o Scaler e o melhor modelo
        melhor_modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=melhor_k, weights=melhor_weights, n_jobs=3))
        ])
        melhor_modelo.fit(x_treino, y_treino)

        cprint(f"Melhor configuração (Validação): K={melhor_k}, Weights={melhor_weights}, Acc={maior_acc}", label="KNN")

        """**Aplicando o melhor modelo sobre o conjunto de teste**"""
        pred_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="KNN")

        return acuracia_teste, melhor_modelo
    
    def metodo_arvoreDecisao(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação da Árvore de Decisão com otimização de hiperparâmetros (profundidade, 
        critério de divisão, amostras por folha/nó e estratégia de splitter).
        """
        cprint("Executando a Árvore de Decisão...", label="AD")

        # Definição do espaço de busca
        depth_range = range(1, 11)
        leaf_range = range(1, 11)
        split_range = range(2, 16)

        if self.modo_teste:
            depth_range = range(1, 2)
            leaf_range = range(1, 2)
            split_range = range(2, 3)

        cprint("Iniciando busca de hiperparâmetros...", label="AD")
        maior_acc = -1
        melhor_modelo = None

        for criterion in ("entropy", "gini"):
            for depth in tqdm(depth_range, ascii=True, desc=f"[AD] crit={criterion}"):
                for leaf in tqdm(leaf_range, desc=f"depth={depth}", leave=False, ascii=True):
                    for min_split in split_range:
                        for splitter in ('best', 'random'):
                            ad = DecisionTreeClassifier(
                                criterion=criterion,
                                max_depth=depth,
                                min_samples_leaf=leaf,
                                min_samples_split=min_split,
                                splitter=splitter
                            )
                            ad.fit(x_treino, y_treino)
                            pred_val = ad.predict(x_val)
                            acc = accuracy_score(y_val, pred_val)
                            
                            if acc > maior_acc:
                                maior_acc = acc
                                melhor_modelo = ad

        cprint(f"Melhor configuração (Validação): Criterion={melhor_modelo.criterion}, MaxDepth={melhor_modelo.max_depth}, Acc={maior_acc}", label="AD")

        # Avaliação final sobre o conjunto de teste
        pred_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="AD")
        
        return acuracia_teste, melhor_modelo
    
    def metodo_naiveBayes(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação do classificador Naive Bayes Gaussiano.
        """
        cprint("Executando o Naive Bayes...", label="NB")

        modelo = GaussianNB()
        modelo.fit(x_treino, y_treino)

        # Avaliação em validação e teste
        pred_val = modelo.predict(x_val)
        acc_val = accuracy_score(y_val, pred_val)
        
        pred_teste = modelo.predict(x_teste)
        acc_teste = accuracy_score(y_teste, pred_teste)
        
        cprint(f"Acurácia (Validação): {acc_val}", label="NB")
        cprint(f"Acurácia (Teste): {acc_teste}", label="NB")

        return acc_teste, modelo

    def metodo_svm(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação de Support Vector Machines (SVM) com busca de hiperparâmetros.
        Avalia diferentes kernels e parâmetros de regularização (C).
        """
        cprint("Executando o SVM...", label="SVM")

        # Escalonamento manual antes do loop para performance
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)

        kernels = ['linear'] if self.modo_teste else ['linear', 'poly', 'rbf', 'sigmoid']

        # Valores comuns para C em escala logarítimica, revisar
        c_range = [1] if self.modo_teste else [0.001, 0.01, 0.1, 1, 10, 100, 1000]   

        maior_acc = -1
        melhor_kernel = "linear"
        melhor_c = 1

        for kernel in tqdm(kernels, ascii=True, leave=False):
            for c_val in tqdm(c_range, ascii=True, desc=f"kernel='{kernel}'"):
                svm = SVC(kernel=kernel, C=c_val, probability=True)
                svm.fit(x_treino_s, y_treino)
                pred_val = svm.predict(x_val_s)
                acc = accuracy_score(y_val, pred_val) 

                if acc > maior_acc:
                    maior_acc = acc
                    melhor_kernel = kernel
                    melhor_c = c_val

        # Pipeline final consolidando o escalonamento
        melhor_modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=melhor_kernel, C=melhor_c, probability=True)),
        ])
        melhor_modelo.fit(x_treino, y_treino)

        pred_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="SVM")

        return acuracia_teste, melhor_modelo

    def metodo_mlp(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação do Multi-Layer Perceptron (MLP) com busca exaustiva de hiperparâmetros.
        Otimiza arquitetura da rede, taxa de aprendizado, funções de ativação e tamanho de lote.
        """
        cprint("Executando o Multi-Layer Perceptron...", label="MLP")
        
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)

        # Espaço de busca para hiperparâmetros conforme requisitos acadêmicos
        epocas_range = [30, 50, 100, 200, 500]
        taxa_aprendizado_range = [0.0001, 0.001, 0.01, 0.1]
        neuronios_por_camada_range = [44, 55, 66, 77, 88] 
        funcao_ativacao_range = ['relu', 'tanh', 'logistic']
        tamanho_batch_range = [32, 64, 128, 256]

        if self.modo_teste:
            epocas_range = [100] # testa só 100 epocas
            taxa_aprendizado_range = [0.1] # Learning rate
            neuronios_por_camada_range = [4] # Deve estar entre o numero de neuronios de entrada e o de saida
            funcao_ativacao_range = ['logistic'] # logistic é bom pra classificação binaria
            tamanho_batch_range = [256]

        cprint("Fazendo busca de hiperparâmetros...", label="MLP")
        maior_acc = -1
        melhor_modelo_base = None

        for epoca in tqdm(epocas_range, ascii=True, desc="épocas"):
            for batch in tqdm(tamanho_batch_range, ascii=True, desc="batch size", leave=False):
                for taxa in taxa_aprendizado_range:
                    for neuronios in neuronios_por_camada_range:
                        for ativacao in funcao_ativacao_range:
                            mlp = MLPClassifier(
                                max_iter=epoca,
                                batch_size=batch,
                                hidden_layer_sizes=(neuronios,),
                                learning_rate_init=taxa,
                                activation=ativacao
                            )
                            mlp.fit(x_treino_s, y_treino)
                            pred_val = mlp.predict(x_val_s)
                            acc = accuracy_score(y_val, pred_val)
                            
                            if acc > maior_acc:
                                maior_acc = acc
                                melhor_modelo_base = mlp

        # Pipeline final com escalonamento
        melhor_modelo = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', melhor_modelo_base)
        ])
        melhor_modelo.fit(x_treino, y_treino)

        pred_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="MLP")
        
        return acuracia_teste, melhor_modelo

    def metodo_randomForest(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação de Random Forest com busca de hiperparâmetros.
        Avalia número de estimadores, critério, profundidade e restrições de nós/folhas.
        """
        cprint("Executando a Random Forest...", label="RF")

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

        cprint("Iniciando busca de hiperparâmetros...", label="RF")
        maior_acc = -1
        melhor_modelo = None

        for criterion in ("entropy", "gini"):
            for n_estimadores in tqdm(n_estimadores_range, ascii=True, desc=f"[ RF ] criterion = {criterion}"):
                for profundidade in tqdm(profundidade_range, desc=f"[ RF ] n_estimadores = {n_estimadores}", leave=False, ascii=True):
                    for min_elementos_folha in min_elementos_folha_range:
                        for min_divisao_nos in min_divisao_nos_range:
                            rf = RandomForestClassifier(
                                n_estimators=n_estimadores,
                                criterion=criterion,
                                max_depth=profundidade,
                                min_samples_leaf=min_elementos_folha,
                                min_samples_split=min_divisao_nos,
                                n_jobs=3,
                            )
                            rf.fit(x_treino, y_treino)
                            pred_val = rf.predict(x_val)
                            acc = accuracy_score(y_val, pred_val)
                            
                            if acc > maior_acc:
                                maior_acc = acc
                                melhor_modelo = rf

        cprint(f"Melhor configuração (Validação): Criterion={melhor_modelo.criterion}, N_Est={melhor_modelo.n_estimators}, Acc={maior_acc}", label="RF")

        """Aplicando a melhor configuração sobre o **Conjunto de Teste**"""
        pred_teste = melhor_modelo.predict(x_teste)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="RF")

        return acuracia_teste, melhor_modelo

    def metodo_bagging(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação de Bagging com suporte a múltiplos estimadores base.
        Realiza seleção aleatória com reposição (bootstrap) para formação de subconjuntos.
        """
        cprint("Executando o Bagging...", label="Bagging")

        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)
        x_teste_s = scaler.transform(x_teste)

        # Parâmetros de busca
        n_estimadores_range = [50, 100, 500] 
        n_amostras_range = [0.1, 0.5, 0.7, 1.0] 
        estimador_range = [KNeighborsClassifier, SVC, DecisionTreeClassifier, MLPClassifier, GaussianNB]

        if self.modo_teste:
            n_estimadores_range = [1] 
            n_amostras_range = [0.1] 
            estimador_range = [KNeighborsClassifier, SVC, DecisionTreeClassifier, MLPClassifier, GaussianNB]

        cprint("Iniciando busca de hiperparâmetros...", label="Bagging")
        maior_acc = -1
        melhor_modelo = None

        for classe_base in tqdm(estimador_range, desc="Estimadores Base", ascii=True):
            n_est_uso = [5, 10, 15] if classe_base == MLPClassifier else n_estimadores_range
            
            for n_est in tqdm(n_est_uso, desc=f"n_estimators ({classe_base.__name__})", leave=False, ascii=True):
                for max_samp in n_amostras_range:
                    bag = BaggingClassifier(estimator=classe_base(), n_estimators=n_est, max_samples=max_samp, n_jobs=3)
                    bag.fit(x_treino_s, y_treino)
                    pred_val = bag.predict(x_val_s)
                    acc = accuracy_score(y_val, pred_val)
                    
                    if acc > maior_acc:
                        maior_acc = acc
                        melhor_modelo = bag

        if melhor_modelo is None:
            cprint("Aviso: Nenhum modelo de Bagging foi gerado.", label="Bagging")
            return 0, None

        # Identifica o nome do melhor estimador base para o log
        nome_base = type(getattr(melhor_modelo, 'estimator', getattr(melhor_modelo, 'base_estimator', None))).__name__
        cprint(f"Melhor configuração (Validação): Estimador={nome_base}, Acc={maior_acc}", label="Bagging")

        pred_teste = melhor_modelo.predict(x_teste_s)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="Bagging")

        # Pipeline final para salvar o scaler junto
        melhor_modelo_final = Pipeline([
            ('scaler', scaler),
            ('bagging', melhor_modelo)
        ])

        return acuracia_teste, melhor_modelo_final

    def metodo_boosting(self, x_treino, y_treino, x_teste, y_teste, x_val, y_val):
        """
        Implementação de Boosting (AdaBoost) otimizando estimadores base e taxa de aprendizado.
        """
        cprint("Executando o Boosting (AdaBoost)...", label="Boosting")

        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s = scaler.transform(x_val)
        x_teste_s = scaler.transform(x_teste)

        n_estimadores_range = [50, 100, 500] 
        learning_rate_range = [0.01, 0.1, 1.0]
        
        # Nota: AdaBoost requer que o estimador base suporte pesos (sample_weight).
        # DecisionTree, GaussianNB e SVC suportam. KNN e MLP não suportam no scikit-learn.
        estimador_range = [DecisionTreeClassifier, GaussianNB, SVC] 

        if self.modo_teste:
            n_estimadores_range = [10] 
            learning_rate_range = [0.1] 
            estimador_range = [DecisionTreeClassifier]

        cprint("Fazendo busca de hiperparâmetros...", label="Boosting")
        maior_acc = -1
        melhor_modelo = None
        modelo_precisou_escalonar = False

        for classe_base in tqdm(estimador_range, desc="Estimadores Base", ascii=True):
            precisa_escalonar = (classe_base == SVC)
            x_treino_uso = x_treino_s if precisa_escalonar else x_treino
            x_val_uso = x_val_s if precisa_escalonar else x_val

            for n_est in tqdm(n_estimadores_range, desc=f"n_est ({classe_base.__name__})", leave=False, ascii=True):
                for lr in learning_rate_range:
                    boost = AdaBoostClassifier(estimator=classe_base(), n_estimators=n_est, learning_rate=lr)
                    boost.fit(x_treino_uso, y_treino)
                    pred_val = boost.predict(x_val_uso)
                    acc = accuracy_score(y_val, pred_val)
                    
                    if acc > maior_acc:
                        maior_acc = acc
                        melhor_modelo, modelo_precisou_escalonar = boost, precisa_escalonar

        if melhor_modelo is None:
            return 0, None

        nome_base = type(getattr(melhor_modelo, 'estimator', getattr(melhor_modelo, 'base_estimator', None))).__name__
        cprint(f"Melhor configuração (Validação): Estimador={nome_base}, Acc={maior_acc}", label="Boosting")

        """**Aplicando o melhor modelo sobre o conjunto de teste**"""
        x_teste_uso = x_teste_s if modelo_precisou_escalonar else x_teste
        pred_teste = melhor_modelo.predict(x_teste_uso)
        acuracia_teste = accuracy_score(y_teste, pred_teste)
        cprint(f"Acurácia (Teste): {acuracia_teste}", label="Boosting")

        if modelo_precisou_escalonar:
            return acuracia_teste, Pipeline([('scaler', scaler), ('boosting', melhor_modelo)])
        return acuracia_teste, melhor_modelo

    def metodo_combSoma(self, x_treino, x_teste, y_treino, y_teste, modelos_carregados: dict):
        """
        Combinação de classificadores pela Regra da Soma das probabilidades.
        """
        cprint("Executando combinação pela Regra da Soma...", label="SOMA")

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
      
    def metodo_combMajoritaria(self, x_treino, x_teste, y_treino, y_teste, modelos_carregados: dict):
        """
        Combinação de classificadores por Voto Majoritário.
        """
        cprint("Executando combinação por Voto Majoritário...", label="MAJOR")

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

    def metodo_combBordaCount(self, x_treino, x_teste, y_treino, y_teste, modelos_carregados: dict):

        cprint("Executando a combinação Borda Count...", label="BORDA")

        # Coleta as probabilidades de cada modelo
        # probs_por_modelo[i] = array (n_amostras, n_classes) do modelo i
        probs_por_modelo = []
        for label, modelo in modelos_carregados.items():
            probs_por_modelo.append(modelo.predict_proba(x_teste))

        n_amostras = len(x_teste)
        n_classes = 2  # pokemon 1 (idx 0) ou pokemon 2 (idx 1)

        opiniao = []

        for i in range(n_amostras):

            # Acumula os ranks de cada classe para a amostra i
            borda_scores = np.zeros(n_classes)

            for probs in probs_por_modelo: # para cada batalha no dataset 

                p_classe_0 = probs[i][0]
                p_classe_1 = probs[i][1]

                # O mais confiante recebe rank 2, o menos confiante recebe rank 1
                if p_classe_0 > p_classe_1:
                    borda_scores[0] += 2
                    borda_scores[1] += 1
                else:
                    borda_scores[0] += 1
                    borda_scores[1] += 2

            # A classe com maior somatório de ranks vence
            opiniao.append(np.argmax(borda_scores))

        acuracia = accuracy_score(y_teste, opiniao)
        cprint(f"Acurácia sobre o teste: {acuracia}", label="BORDA")


        return acuracia, None
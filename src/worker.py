import argparse
import pandas as pd
import os
import time
import traceback
import sys
import joblib as jb
from datetime import datetime

# Adiciona a pasta src ao path para permitir imports internos
# Isso permite que 'from processamento...' funcione mesmo rodando da raiz ou de src
sys.path.append(os.path.dirname(__file__))

from metodos_aprendizado.metodosAprendizado import MetodosAprendizado
from processamento.ler_dataset_processado import ler_datasets
from utils.print_customizado import cprint

# Define a raiz do projeto (um nível acima de src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def log_error(iteracao, error_msg):
    """Salva o erro em um arquivo de log na raiz do projeto."""
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(log_dir, f"worker_it{iteracao}_error.log")
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"--- Erro na Iteração {iteracao} em {timestamp} ---\n")
        f.write(error_msg)
        f.write("\n" + "="*50 + "\n\n")

def salvar_resultados(df, caminho):
    """Salva os resultados no CSV garantindo o alinhamento das colunas e evitando condições de corrida."""
    lock_path = caminho + ".lock"
    max_tentativas = 100
    atraso = 0.5

    for _ in range(max_tentativas):
        try:
            # Tenta criar o arquivo de lock de forma atômica
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                if os.path.exists(caminho):
                    df_antigo = pd.read_csv(caminho)
                    df_novo = pd.concat([df_antigo, df], ignore_index=True)
                    df_novo.to_csv(caminho, index=False)
                else:
                    df.to_csv(caminho, index=False)
                return # Sucesso
            finally:
                os.close(fd)
                if os.path.exists(lock_path):
                    os.remove(lock_path)
        except (FileExistsError, PermissionError):
            # Outro processo está escrevendo ou o arquivo está bloqueado
            time.sleep(atraso)
    
    cprint(f"Erro: Não foi possível obter trava para {caminho} após {max_tentativas} tentativas.", label="ERR")

def salvar_modelos(modelos : dict, iteracao, teste):
    """Salva os modelos gerados a partir da iteração"""

    dir_salvamento = "modelos/teste/" if teste else "modelos/"
    if not os.path.exists(dir_salvamento):
        os.makedirs(dir_salvamento)

    for chave, modelo in modelos.items():
        if modelo is None:
            continue
            
        nome_modelo = chave.removeprefix("metodo_")
        nome_arquivo = f"{iteracao:02}_{nome_modelo}.joblib"
        caminho_salvamento = os.path.join(dir_salvamento, nome_arquivo)
        jb.dump(modelo, caminho_salvamento)



def main():
    parser = argparse.ArgumentParser(description="Worker de processamento de uma única iteração.")
    parser.add_argument("--iteracao", type=int, required=True, help="Número da iteração atual.")
    parser.add_argument("--teste", action="store_true", help="Ativa o modo de teste.")
    parser.add_argument("--arquivo", type=str, help="Nome do arquivo de resultados.")
    args = parser.parse_args()

    # --- CONFIGURAÇÃO MANUAL DE TÉCNICAS ---
    
    TECNICAS_PARA_RODAR = [
        "metodo_knn",
        "metodo_arvoreDecisao",
        "metodo_naiveBayes",
        "metodo_svm",
        "metodo_mlp",
        "metodo_bagging",
        "metodo_boosting",
        # "metodo_randomForest",
        # "metodo_combSoma",
        # "metodo_combMajoritaria",
        # "metodo_combBordaCount"
    ]

    try:
        cprint(f"Iniciando Iteração {args.iteracao}...", label=f"CLT {args.iteracao}")

        # Carga de dados (o loader já deve lidar com caminhos internos)
        dados = ler_datasets("dados")

        if args.teste:
            dados = dados.sample(1000)

        ma = MetodosAprendizado()
        ma.modo_teste = args.teste
        
        x_treino, y_treino, x_teste, y_teste, x_val, y_val = ma.split_dataset(dados, seed=int(args.iteracao))
        
        # Verifica se precisa carregar modelos (se houver métodos de combinação ou RF na lista)
        modelos_carregados = None
        metodos_que_precisam_de_carga = [
            "metodo_combSoma", 
            "metodo_combMajoritaria", 
            "metodo_combBordaCount", 
            "metodo_randomForest"
        ]
        
        # Só carrega se o método estiver na lista para rodar
        if any(m in TECNICAS_PARA_RODAR for m in metodos_que_precisam_de_carga):
            # Se for RF, só precisa carregar se a AD não for rodar agora (pois a AD em memória é preferível)
            precisa_carregar = True
            if "metodo_randomForest" in TECNICAS_PARA_RODAR and "metodo_arvoreDecisao" in TECNICAS_PARA_RODAR:
                # Se ambos estão na lista, o RF usará o self.modelo_AD gerado na hora
                # Mas se houver outros métodos de combinação, ainda precisa carregar.
                outros_comb = [m for m in metodos_que_precisam_de_carga if m != "metodo_randomForest"]
                if not any(m in TECNICAS_PARA_RODAR for m in outros_comb):
                    precisa_carregar = False
            
            if precisa_carregar:
                modelos_carregados = ma.carregar_modelos(args.iteracao, args.teste)

        # Dispara os comandos selecionados
        resultados, modelos = ma.disparar_comando(
            parametros={
                "x_treino": x_treino, "y_treino": y_treino,
                "x_teste":  x_teste,  "y_teste":  y_teste,
                "x_val":    x_val,    "y_val":    y_val,
                "modo_teste": args.teste
            },
            lista_tecnicas=TECNICAS_PARA_RODAR,
            modelos_carregados=modelos_carregados
        )

        # Formatação do Resultado
        linha = {"iteracao/seed": args.iteracao}
        linha.update(resultados)
        df_result = pd.DataFrame([linha])

        # Caminho de Salvamento
        nome_arquivo = args.arquivo if args.arquivo else ("resultados_teste.csv" if args.teste else "resultados.csv")
        resultados_dir = os.path.join(BASE_DIR, "resultados")
        if not os.path.exists(resultados_dir):
            os.makedirs(resultados_dir)
            
        caminho_csv = os.path.join(resultados_dir, nome_arquivo)
        
        salvar_resultados(df_result, caminho_csv)

        # Salvamento de modelos (apenas os que foram treinados nesta rodada)
        if modelos:
            cprint("Salvando modelos...", label=f"CLT {args.iteracao}")
            salvar_modelos(modelos, args.iteracao, args.teste)
            cprint("Modelos salvos!", label=f"CLT {args.iteracao}")
        
        cprint(f"Iteração {args.iteracao} finalizada com sucesso!", label=f"CLT {args.iteracao}")

        esperar = 1
        cprint(f"Fechando em {esperar}s")
        time.sleep(esperar)

    except Exception:
        error_msg = traceback.format_exc()
        cprint(f"ERRO FATAL na Iteração {args.iteracao}! Verifique logs/", label=f"ERR {args.iteracao}")
        
        log_error(args.iteracao, error_msg)
        time.sleep(10)
        

if __name__ == "__main__":
    main()

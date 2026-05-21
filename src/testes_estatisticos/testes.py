from scipy.stats import friedmanchisquare
import pandas as pd
from utils.print_customizado import cprint
import scikit_posthocs as sp

def teste_friedman(historico_resultados: dict, confianca=0.95):
    """
    Aplica o teste de Friedman para detectar se há diferença
    entre o desempenho dos classificadores.

    Args:
        historico_resultados (dict): {
            "metodo_knn": [acc_iter1, acc_iter2, ...],
            "metodo_arvoreDecisao": [acc_iter1, acc_iter2, ...],
            ...
        }
        confianca (float): Nível de confiança (padrão 0.95)
    """


    alpha = 1 - confianca

    nomes_legiveis = {
        "metodo_knn":             "KNN",
        "metodo_arvoreDecisao":   "Árvore de Decisão",
        "metodo_naiveBayes":      "Naive Bayes",
        "metodo_svm":             "SVM",
        "metodo_mlp":             "MLP",
        "metodo_randomForest":    "Random Forest",
        "metodo_bagging":         "Bagging",
        "metodo_boosting":        "Boosting",
        "metodo_combSoma":        "Comb. Soma",
        "metodo_combMajoritaria": "Comb. Majoritária",
        "metodo_combBordaCount":  "Comb. Borda Count",
    }

    df = pd.DataFrame(historico_resultados)

    # Remove colunas de controle
    df = df.drop(columns=["iteracao/seed", "iteracao", "seed"], errors="ignore")

    df.rename(columns=nomes_legiveis, inplace=True)


    print("\n" + "="*60)
    print("  TESTE DE FRIEDMAN (não paramétrico, confiança 95%)")
    print("="*60)
    print("\nH₀: Todos os classificadores têm desempenho equivalente")
    print("H₁: Pelo menos um classificador difere dos demais")

    print("\nAcurácias por iteração:")
    print(df.to_string())

    print("\nEstatísticas descritivas:")
    print(df.describe().loc[["mean", "std", "min", "max"]].to_string())

    grupos = [df[col].values for col in df.columns]
    stat, p_valor = friedmanchisquare(*grupos)

    print(f"\nEstatística de Friedman : {stat:.4f}")
    print(f"p-valor                 : {p_valor:.2e}")
    print(f"Alpha (1 - confiança)   : {alpha:.6f}")

    if p_valor < alpha:
        print(f"\n✅ Rejeitamos H₀ (p={p_valor:.2e} < α={alpha:.6f})")
        print("→ Há diferença significativa entre pelo menos um par de classificadores.")
    else:
        print(f"\n❌ Não rejeitamos H₀ (p={p_valor:.2e} ≥ α={alpha:.6f})")
        print("→ Não há evidência de diferença significativa entre os classificadores.")

    print("="*60 + "\n")

    return stat, p_valor


def teste_nemenyi(historico_resultados: dict, confianca=0.95):
    """
    Aplica o teste post-hoc de Nemenyi para comparar os classificadores dois a dois.
    Deve ser aplicado apenas se o teste de Friedman rejeitar H₀.

    Args:
        historico_resultados (dict): {
            "metodo_knn": [acc_iter1, acc_iter2, ...],
            ...
        }
        confianca (float): Nível de confiança (padrão 0.95)
    """
 

    alpha = 1 - confianca

    nomes_legiveis = {
        "metodo_knn":             "KNN",
        "metodo_arvoreDecisao":   "Árvore de Decisão",
        "metodo_naiveBayes":      "Naive Bayes",
        "metodo_svm":             "SVM",
        "metodo_mlp":             "MLP",
        "metodo_randomForest":    "Random Forest",
        "metodo_bagging":         "Bagging",
        "metodo_boosting":        "Boosting",
        "metodo_combSoma":        "Comb. Soma",
        "metodo_combMajoritaria": "Comb. Majoritária",
        "metodo_combBordaCount":  "Comb. Borda Count",
    }

    df = pd.DataFrame(historico_resultados)

    # Remove colunas de controle
    df = df.drop(columns=["iteracao/seed", "iteracao", "seed"], errors="ignore")

    df.rename(columns=nomes_legiveis, inplace=True)

    print("\n" + "="*60)
    print("  TESTE DE NEMENYI (post-hoc, confiança 95%)")
    print("="*60)
    print("H₀ (por par): Os dois classificadores têm desempenho equivalente")
    print("H₁ (por par): Os dois classificadores têm desempenhos diferentes")

    p_nemenyi = sp.posthoc_nemenyi_friedman(df)

    print("\nMatriz de p-valores (Nemenyi):")
    print(p_nemenyi.to_string())

    print(f"\nPares com diferença significativa (p < {alpha:.6f}):")
    cols = df.columns.tolist()
    encontrou = False
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p_val = p_nemenyi.iloc[i, j]
            if p_val < alpha:
                media_i = df[cols[i]].mean()
                media_j = df[cols[j]].mean()
                melhor = cols[i] if media_i > media_j else cols[j]
                print(f"  • {cols[i]} vs {cols[j]}: p={p_val:.2e} → melhor: {melhor} "
                      f"(média {max(media_i, media_j):.4f})")
                encontrou = True

    if not encontrou:
        print("  Nenhum par apresentou diferença significativa.")

    print("\n🏆 Ranking por média de acurácia:")
    ranking = df.mean().sort_values(ascending=False)
    for pos, (nome, media) in enumerate(ranking.items(), 1):
        print(f"  {pos}º {nome}: {media:.4f} ± {df[nome].std():.4f}")

    print("="*60 + "\n")

    return p_nemenyi



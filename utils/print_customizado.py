

def cprint(texto, tempo=None):
    """Custom Print

    Print customizado que printa o [Sys] na frente para melhor visualização
    
    Args:
        texto (str): Texto que será printado
    """

    if tempo:
        print(f"[{tempo}s] {texto}")
    else:
        print(f"[Sys] {texto}")


def cprint(texto, label=None, num_espacos=0):
    """Custom Print

    Print customizado que printa o [Sys] na frente para melhor visualização
    
    Args:
        texto (str): Texto que será printado
    """
    string = ""

    string += " "*num_espacos

    if label:
        string += f"[ {label} ] "
    else:
        string += f"[ Sys ] "

    string += texto

    print(string)

if __name__ == "__main__":

    for i in range(10):
        cprint("Texto bem gamer", label="GAMER", num_espacos=i)
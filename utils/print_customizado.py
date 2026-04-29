

def cprint(texto, label=None, num_espacos=0, jump_line=True):
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
        string += f"[ SYS ] "

    string += texto

    if jump_line:
        print(string)
    else:
        print(string, end="")

if __name__ == "__main__":

    for i in range(10):
        cprint("Texto bem gamer", label="GAMER", num_espacos=i)
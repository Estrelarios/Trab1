import time

def cprint(texto, label=None, num_idents=0, jump_line=True):
    """Custom Print

    Args:
        texto (str): Texto a ser printado
        label (str, optional): Texto que vai dentro dos colchetes. Defaults to None.
        num_idents (int, optional): Espaços antes do print. Defaults to 0.
        jump_line (bool, optional): Pode prefir por não pular a linha. Defaults to True.
    
    Exemplo:

    (num_idents espaços)[ LABEL HH:MM:SS ] texto (jump_line)
    
    """

    string = ""

    # Hora atual
    hora_atual = time.strftime("%H:%M:%S")

    # Label
    prefixo = label if label else "SYS"

    # Formato: [ LABEL hh:mm:ss ]
    tag = f"[ {prefixo} {hora_atual} ]"

    ident = " " * num_idents

    string = f"{ident}{tag} {texto}"

    if jump_line:
        print(string)
    else:
        print(string, end="")

if __name__ == "__main__":

    for i in range(10):
        cprint("Texto bem gamer", label="GAMER", num_idents=i)

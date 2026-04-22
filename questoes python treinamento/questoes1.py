#1. Função que lê os dados de um arquivo txt. A entrada é o nome com o caminho até o 
# arquivo txt e a saída é uma lista onde cada item é uma linha do arquivo txt.

def ler_arquivo():
    nome = input("Digite o caminho do arquivo: ")
    
    lista = []

    with open(nome, "r") as arquivo:
        for linha in arquivo:
            lista.append(linha.strip())

    return lista


# 2. Função que escreve os dados de uma lista de strings em um arquivo txt. 
# A entrada é o nome com o caminho até o arquivo txt além da lista de strings. 
# Não há saída. Cada linha do arquivo é um item da lista.

def escrever_dados():
    nome = input("Digite o caminho do arquivo: ")

    lista = []

    n = int(input("Quantas linhas vc deseja inserir? "))

    for i in range(n):
        linha = input("Digite a linha: ")
        lista.append(linha)
    
    with open (nome , "w") as arquivo:
        for linha in lista:
            arquivo.write(linha + "\n")
    

#  3. Função que lê os dados de um arquivo csv. A entrada é o nome com o caminho até o arquivo csv 
# e a saída é uma lista de listas onde cada item é uma linha do arquivo csv, 
# e cada item é uma lista de outros itens da coluna.

def ler_csv():
    nome = input ("Digite o caminho do arquivo: ")

    listaLinha = []
    #listaColuna = []

    with open (nome, "r") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            listaColuna = linha.split(",")
            listaLinha.append(listaColuna)
    
    return listaLinha

# 4. Função que escreve os dados de uma lista de listas de strings em um arquivo csv.
#  A entrada é o nome com o caminho até o arquivo csv e a lista de listas de strings e não há saída. 
# Cada linha do arquivo é um item da lista, e cada coluna é um item dessa lista.

def escrever_csv(nome, lista) : 
    with open (nome, "w") as arquivo:
        for linha in lista:
            linhaTexto = ",".join(linha)  
            arquivo.write(linhaTexto + "\n")


#5. Função que recebe o caminho até uma pasta e retorna uma lista com todos os arquivos dentro dessa pasta e suas subpastas.


import os

def pasta(caminho):
    lista = []

    for raiz, dirs, arquivos in os.walk(caminho):
        for arquivo in arquivos:
            caminho_completo = os.path.join(raiz, arquivo)
            lista.append(caminho_completo)

    return lista
#teste2
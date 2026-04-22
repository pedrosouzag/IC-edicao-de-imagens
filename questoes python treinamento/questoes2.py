#1. Criar função para salvar em um arquivo txt pares de números aleatórios. 
# A função deve receber o valor n (número de pares). A saída é o arquivo txt com os n pares de números aleatórios de zero a um. 
# O padrão de escrita para o par é número1;número2. Cada par deve estar em uma linha.

import random

def salvar_arquivo(n):
    lista = []

    for i in range(n):
        x = random.random()
        y = random.random()

        lista.append((x,y))

    with open(pares.txt, "w") as arquivo:
        for x,y in lista:
             arquivo.write(f"{x};{y}\n")



# #2. Criar função para salvar em um arquivo csv pares de números aleatórios. 
# A função deve receber o valor n (número de pares). 
# A saída é o arquivo csv com os n pares de números aleatórios de zero a um. 
# Cada elemento do par fica em uma coluna. Cada par deve estar em uma linha.


def salvar_arquivo_csv(n):
    lista1 = []

    for i in range(n):
        x = random.random()
        y = random.random()

        lista1.append((x, y))
    
    
    with open("pares.csv", "w") as arquivo:
        for x,y in lista1:
            arquivo.write(f"{x},{y}\n")

import random
import csv

def salvar_arquivo_csv2(n):
    with open("pares.csv", "w", newline="") as arquivo:
        writer = csv.writer(arquivo)

        for i in range(n):
            x = random.random()
            y = random.random()
            writer.writerow([x, y])
    

# 3. Criar função para salvar em um arquivo txt pares com um número não aleatório e um número aleatório. 
# A função deve receber o valor n (número de pares). 
# A saída é o arquivo txt com os n pares, onde o primeiro é referente ao termo 
# da sequência e o segundo um número aleatório de zero a um. O padrão de escrita para o par é número1;número2. 
# Cada par deve estar em uma linha. O primeiro termo vai de 1 até n.


def aleatoeioEnaoAleatorioTXT(n):

    with open("pares.txt", "w") as arquivo:
        for i in range(n):
            x = random.uniform(0,1)
            arquivo.write((f"{i + 1};{x}\n"))



# 4. Criar função para salvar em um arquivo csv pares com um número não aleatório e um número aleatório.
# A função deve receber o valor n (número de pares). A saída é o arquivo csv com os n pares,
# onde o primeiro é referente ao termo da sequência e o segundo um número aleatório de zero a um. 
# Cada elemento do par fica em uma coluna. Cada par deve estar em uma linha. O primeiro termo vai de 1 até n.

def aleatoeioEnaoAleatorioCSV(n):
    with open("pares.csv", "w") as arquivo:
        for i in range(n):
            x = random.uniform(0,1)
            arquivo.write((f"{i + 1},{x}\n"))

# 5. Criar uma função que lê de um csv de pares de números aleatórios e plota esses pontos no espaço. 
# Cada par está em uma linha, e cada elemento do par em uma coluna diferente. Dica: use a biblioteca matplotlib.

import matplotlib.pyplot as plt

def lerPares():
    listaX = []
    listaY = []

    with open("pares.csv", "r") as arquivo:
        for linha in arquivo:
            x, y = linha.strip().split(",")
            listaX.append(float(x))
            listaY.append(float(y))

    plt.scatter(listaX, listaY)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pares aleatórios")
    plt.show()



# 6. Criar uma função que lê de um txt de pares de números aleatórios e plota esses pontos no espaço. 
# Cada par está em uma linha, e cada elemento do par está separado por um ponto e vírgula (;). Dica: use a biblioteca matplotlib.

def lerPares2():
    listaX = []
    listaY = []

    with open("pares.txt", "r") as arquivo:
        for linha in arquivo:
            x, y = linha.strip().split(";")
            listaX.append(float(x))
            listaY.append(float(y))

    plt.scatter(listaX, listaY)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pares aleatórios")
    plt.show()
# 7. Criar uma função que lê de um txt de pares onde o primeiro número é o termo da sequência e o 
# segundo é um número aleatório e plota esses pontos no espaço em forma de um grafo linha. 
# Cada par está em uma linha, e cada elemento do par é separado por ponto e vírgula (;). Dica: use a biblioteca matplotlib.

def lerParesAleatorio():
    listaX = []
    listaY = []

    with open("pares.txt", "r") as arquivo:
        for linha in arquivo:
            x, y = linha.strip().split(";")
            listaX.append(float(x))
            listaY.append(float(y))

    plt.plot(listaX, listaY)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pares aleatórios")
    plt.show()



# 8. Criar uma função que lê de um csv de pares onde o primeiro número é o termo da sequência e 
# o segundo é um número aleatório e plota esses pontos no espaço em forma de um grafo linha. 
# Cada par está em uma linha, e cada elemento do par em uma coluna diferente. Dica: use a biblioteca matplotlib.


def lerParesAleatorioCSV():
    listaX = []
    listaY = []

    with open("pares.csv", "r") as arquivo:
        for linha in arquivo:
            x, y = linha.strip().split(",")
            listaX.append(float(x))
            listaY.append(float(y))

    plt.plot(listaX, listaY)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Pares aleatórios")
    plt.show()
# 9. Listar todos os documentos de uma pasta e suas subpastas

import os

def listarDocumentos(caminho):
    lista = []

    for raiz, pastas, arquivos in os.walk(caminho):
        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(raiz, nome_arquivo)
            lista.append(caminho_completo)

    return lista
import nltk

""" descargando el corpus brown y exportando en un txt"""
nltk.download('brown')

#no me dejó descargar el corpus axolotl
#nltk.download('axolotl') 

from nltk.corpus import brown

# exportando el corpus a un archivo txt para bpe
with open('brown_corpus.txt', 'w', encoding='utf-8') as file:
    for word in brown.words():
        file.write(word + ' ')

# guardando el corpus en un String par word-level
with open('brown_corpus.txt', 'r', encoding='utf-8') as file:
    brown_corpus = file.read()


""" entrenando el modelo bpe y tokenizando corpus"""
import subprocess

#metodo para entrenar un modelo bpe
def trainBPE(text: str, model: str):
    # Entrenando el modelo BPE
    subprocess.run(['subword-nmt', 'learn-bpe', '-i', text, '-o', model])

#metodo para tokenizar un corpus dado un modelo
def tokenizeBPE(model: str, corpus: str, text: str):
    # Tokenizando brown_corpus usando el modelo BPE
    comando = "subword-nmt apply-bpe -c "+model+" < "+corpus+" > "+text
    subprocess.run(comando, shell=True)

#metodo para pasar un corpus tokenizado a una lista
def get_corpus_list(directory: str):
    #obteniendo el corpus tokenizado en una lista
    with open(directory, "r") as archivo:
        contenido = archivo.read()
        tokens = contenido.split()
    return tokens


""" calculando la entropia """    
from collections import Counter
import math
def calc_entropy(corpus: list[str]) -> float:
    types = Counter(corpus)
    tokens = len(corpus)
    proba = {word: count / tokens for word, count in types.items()}
    entropy = - sum(p*math.log2(p) for p in proba.values())
    return entropy

"""tokenizando word-level con nltk"""
from nltk.tokenize import word_tokenize

#metodo para tokenizar word level un corpus
def tokenize_word_level(corpus: str):
    tk = word_tokenize(brown_corpus)
    return tk

"""-------------------------------------------------------------------"""
""" Haciendo la practica para el corpus brown """
#entrenando el modelo bpe
trainBPE("brown_corpus.txt","bpe_model")

#tokenizando el corpus con el modelo bpe entrenado
tokenizeBPE("bpe_model","brown_corpus.txt","brown_tokenized.txt")

#pasando el txt del corpus tokenizado a una list
brown_bpe = get_corpus_list("brown_tokenized.txt")

#tokenizando con word-level
brown_wl = tokenize_word_level(brown_corpus)

#calculando la entropía en ambos casos
entropy_bpe = calc_entropy(brown_bpe)
entropy_wl = calc_entropy(brown_wl)

print("entropia de brown con:\nbpe: "+str(entropy_bpe)+"\nwl: "+str(entropy_wl))

"""para axolotl son los mismos pasos, no pude descargar el corpus :c"""

"""
¿aumentó o disminuyó la entropía?

para el corpus brown disminuyó


¿qué significa que la entropía aumente o disminuya en un texto?

si la entropía en un texto aumenta quiere decir que hay mas incertidumbre, es decir que hay mas variedad en los simbolos o caracteres utilizados, se podría deber a una mayor diversidad lexica

si la entropia en un texto disminuye entonces hay menos variedad y por lo tanto podría ser mas predecible

¿cómo influye la tokenización en a entropía de un texto?

al granular los tokens de un texto reducimos la variedad de los mismos, por lo que disminuimos la entropía.

"""

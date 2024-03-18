# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import os
import re
from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import random
import string
from collections import Counter

corpus_freqs = pd.read_csv('crea_full.csv', delimiter='\t', encoding="latin-1")

freq_dict = corpus_freqs.set_index('words')['freq'].to_dict()

nltk.download('stopwords')

stopwords = stopwords.words('spanish')

freq_stop = {}
for i in stopwords:
    if i in freq_dict.keys():
        freq_stop[i] = freq_dict[i]


def makeImage(text):

    wc = WordCloud(background_color="white", max_words=250)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# +
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

makeImage(freq_dict)

# +
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

makeImage(freq_stop)
# -

# # Las nubes de palabras se parecen mucho
# En mi opinión obtenemos lo mismo salvo algunas excepciones.
# parece ser que las stopwords son la base de la estructura pero no del contexto de la lengua,
# por lo que necesitamos de ellas para darle el significado deseado a cada oración y por eso 
# se ocupan mucho.

# # Parte 2: Creación de lenguaje artificial.
# El lenguaje se llama Poissañol y curiosamente siempre empieza las oraciones con una preposición.
# Y tanto la longitud de las oraciones como las de las palabras siguen una distribución poisson.

# +
#Creamos el vocabulario
vocabulario = {}
todo = ['verbos', 'sustantivos', 'adjetivos', 'adverbios', 'preposiciones']
prob_crear = [.30, .35, .2, .1, .05]

for i in range(10000):
    long = np.random.poisson(lam = 6)
    s = ''
    while len(s) <= long:
        s += random.choice(string.ascii_lowercase)
    if s not in vocabulario:
        cat = random.choices(todo, weights = prob_crear, k = 1)[0]
        vocabulario[s] = cat
lista = list(vocabulario.keys())
verbos = [i for i in vocabulario.keys() if vocabulario[i] == 'verbos']
sustantivos = [i for i in vocabulario.keys() if vocabulario[i] == 'sustantivos']
adjetivos = [i for i in vocabulario.keys() if vocabulario[i] == 'adjetivos']
adverbios = [i for i in vocabulario.keys() if vocabulario[i] == 'adverbios']
preposiciones = [i for i in vocabulario.keys() if vocabulario[i] == 'preposiciones']
# -

#Las reglas gramaticales y probabilidad de tener un tipo de palabra después de otra.
prob_sig = [[0, .3, .2, .2, .3], [.4, .1, 0, .2, .3], [0, .5, 0, .4, .1], [0, 0, .9, .1, 0], [.2, .2, .2, .2, .2]] 
def SiguientePalabra(palabra):
    if vocabulario[palabra] == 'verbos':
        return random.choice(globals()[random.choices(todo, prob_sig[0], k = 1)[0]])
    if vocabulario[palabra] == 'sustantivos':
        return random.choice(globals()[random.choices(todo, prob_sig[1], k = 1)[0]])
    if vocabulario[palabra] == 'adjetivos':
        return random.choice(globals()[random.choices(todo, prob_sig[2], k = 1)[0]])
    if vocabulario[palabra] == 'adverbios':
        return random.choice(globals()[random.choices(todo, prob_sig[3], k = 1)[0]])
    if vocabulario[palabra] == 'preposiciones':
        return random.choice(globals()[random.choices(todo, prob_sig[4], k = 1)[0]])
        


#Función que crea las oraciones.
def CrearOracion():
    oracion = [random.choice(preposiciones)]
    long = np.random.poisson(11)
    for i in range(long):
        oracion.append(SiguientePalabra(oracion[-1]))
    return oracion


#Creamos el corpus
minicorpus = []
for i in range(1000000):
    minicorpus.append(CrearOracion())

#Añadimos las palabras a una lista
todas_palabras = []
for oracion in minicorpus:
    for palabra in oracion:
        todas_palabras.append(palabra)
        

#Contamos las palabras.
vocabulary = Counter(todas_palabras)

len(vocabulary)

vocabulary.most_common(10)

#Ejemplo de frase:
ejemplo = ''
for i in minicorpus[1]:
    ejemplo += i + " "
print(ejemplo)


# +
def get_frequencies(vocabulary: Counter, n: int) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]

def plot_frequencies(frequencies: list, title="Freq of words", log_scale=False):
    x = list(range(1, len(frequencies)+1))
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)


# -

# Visualizamos la distribución de las palabras que parece que sigue zipf
frequencies = get_frequencies(vocabulary, 100)
plot_frequencies(frequencies)

plot_frequencies(frequencies, log_scale=True)

# +
from scipy.optimize import minimize

def calculate_alpha(ranks: np.array, frecs: np.array) -> float:
    # Inicialización
    a0 = 1
    # Función de minimización:
    func = lambda a: sum((np.log(frecs)-(np.log(frecs[0])-a*np.log(ranks)))**2)
    # Minimización: Usando minimize de scipy.optimize:
    return minimize(func, a0).x[0] 

ranks = np.array(corpus_freqs.index) + 1
frecs = np.array(corpus_freqs['freq'])

a_hat = calculate_alpha(ranks, frecs)

print('alpha:', a_hat)


# -

def plot_generate_zipf(alpha: np.float64, ranks: np.array, freqs: np.array) -> None:
    plt.plot(np.log(ranks), -a_hat*np.log(ranks) + np.log(frecs[0]), color='r', label='Aproximación Zipf')


plot_generate_zipf(a_hat, ranks, frecs)
plt.plot(np.log(ranks), np.log(frecs), color='b', label='Distribución original')
plt.xlabel('log ranks')
plt.ylabel('log frecs')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()



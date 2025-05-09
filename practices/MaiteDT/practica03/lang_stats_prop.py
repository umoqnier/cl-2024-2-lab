# -*- coding: utf-8 -*-
"""Lang_Stats_Prop.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ipvHNy_v1neGAaanjtVS1-nhHNEOnJSz

# Comparando Stopwords

Incorporando el Corpus CREA
"""

# Bibliotecas
import nltk
#from nltk.corpus import stopwords
nltk.download('stopwords')
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 4]
from re import sub
import numpy as np
import pandas as pd
from itertools import chain
import csv

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

corpus_csv = pd.read_csv('crea_full.csv', delimiter='\t', encoding="latin-1")
crea_dict = pd.Series(corpus_csv.freq.values, index=corpus_csv.words).to_dict()
most_common_crea = Counter(crea_dict)

"""A partir de este corpus, con Zpif, vamos a determinar las stopwords"""

zipf_cloud = WordCloud()

zipf_cloud.generate_from_frequencies(most_common_crea)

plt.figure(figsize=(8,6), dpi=120)
plt.imshow(zipf_cloud)
plt.axis("off")
plt.show()

"""Importando las Stopwords de la paquetería nltk"""

stopword_es = nltk.corpus.stopwords.words('spanish')
stopword_freq = Counter(stopword_es)

"""Generamos una nube de palabras de las Stopwords obtenidas con la paquetería nltk"""

nltk_cloud = WordCloud()

nltk_cloud.generate_from_frequencies(stopword_freq)

plt.figure(figsize=(8,6), dpi=120)
plt.imshow(nltk_cloud)
plt.axis("off")
plt.show()

"""

*   ¿Obtenemos el mismo resultado? ¿Por qué?

A simple vista, podemos decir que sí, las nubes son muy parecidas, podemos ver palabras como "de", "la", "que", "las", sin embargo, debemos de considerar que el resultado puede tener unas ligeras alteraciones ya que la paquetería nltk nos brinda una lista de stopwords, que si bien es muy útil para cuando queremos filtrar las palabras de nuestro texto, no nos brinda información acerca de las frequencias, ya que es una lista"""

stopword_es[:10]

"""Entonces, cuando estamos haciendo el Counter que utilizamos para la creación de la nube, tenemos que la frecuencia de todas las palbras será 1"""

stopword_freq.most_common(10)

"""# Zipf en lenguaje artificial

Primer paso: Creación del lenguaje artificial
"""

import random

def generate_random_corpus(alfa_1, alfa_2, num_iteraciones):
  corpus = {}
  for x in range(num_iteraciones):
    frecuencia_random = random.randint(10,4000)
    len_random = random.randint(3, 17)
    part = len_random%5 +1
    new_word = ""
    for i in range(len_random):
      if ((i+1)%len_random)%2 == 0:
        new_word += alfa_1[random.randint(0, len(alfa_1)-1)]
      else:
        new_word += alfa_2[random.randint(0, len(alfa_2)-1)]
    corpus[new_word] = frecuencia_random
  return corpus

roman = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

kata = ['カ', 'キ', 'ク', 'ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ',
        'テ', 'ト', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'マ',
        'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ']

art_corpus = generate_random_corpus(kata, roman, 3000)
list_to_show = list(art_corpus)
list_to_show[:10]

"""Así hemos inventado el ✨**japañol**✨, con el que podríamos tener una oración como sigue, esperando que haga sentido si es que lo tiene:

xハク yヨbカmトyリgラ nケtヨ yテム

A este lenguaje inventado, también le creamos frecuencias aleatorias y veamos si es que así se cumpliría la ley de Zipf
"""

sorted_corpus = sorted(art_corpus.items(), key = lambda x:x [1], reverse = True)
japanol_dict = dict(sorted_corpus)
print(japanol_dict)

my_language = pd.DataFrame({"palabra": japanol_dict.keys(), "freq": japanol_dict.values()})
print(my_language)
my_language["freq"].plot(marker="o")
plt.title('Ley de Zipf en el Japañol')
plt.xlabel('rank')
plt.ylabel('freq')
plt.show()

"""Asombrosamente, se ve que cumple con la distribución de Zipf, la verdad pensé que no lo iba a hacer porque a fin de cuentas se está generando aleatoriamente tanto la longitud de las palabras como su frecuencia, pero al parecer incluso así Zipf se cumple 😱"""
# Bibliotecas
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 4]
from re import sub
import numpy as np
import pandas as pd
from itertools import chain
from scipy.optimize import minimize

corpus_freqs = pd.read_csv('CREA_total.TXT', delimiter='\t', encoding="latin-1")
# Pasar strings a números y quitar espacios
corpus_freqs['Frec.absoluta '] = corpus_freqs['Frec.absoluta '].apply(lambda x: int(sub(r',', '', x)) if isinstance(x, str) else x)
corpus_freqs['     Orden'] = corpus_freqs['     Orden'].apply(lambda x: str(x).strip())

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('spanish'))

# Tomar las palabras más frecuentes
zipf = set(corpus_freqs['     Orden'].head(300))

# Comparar con las stopwords
interseccion = zipf.intersection(stop_words)

print(f"\nHay {len(interseccion)} de {len(stop_words)} stopwords en las palabras más frecuentes")

if(input("¿Desea generar la nube de palabras? (s/n) ") == 's'):
    from wordcloud import WordCloud

    # Convertir el DataFrame a un diccionario con palabras como claves y frecuencias como valores
    word_frequencies = dict(zip(corpus_freqs['     Orden'].str.strip().head(300), corpus_freqs['Frec.absoluta ']))

    # Crear la nube de palabras
    nube = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

    # Mostrar la nube de palabras
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(nube)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()


# Ejercicio 2
# Leer el archivo
corpus = open('Pikachu.txt', 'r', encoding='latin-1').read()
corpus_freqs = {}
for word in corpus.split():
    if word in corpus_freqs:
        corpus_freqs[word] += 1
    else:
        corpus_freqs[word] = 1

# Ordenar el diccionario
corpus_freqs = dict(sorted(corpus_freqs.items(), key=lambda item: item[1], reverse=True))
corpus_freqs = pd.DataFrame({'Palabra': list(corpus_freqs.keys()), 'Frecuencia': list(corpus_freqs.values())})

def calculate_alpha(ranks: np.array, frecs: np.array) -> float:
    # Inicialización
    a0 = 1
    # Función de minimización:
    func = lambda a: sum((np.log(frecs)-(np.log(frecs[0])-a*np.log(ranks)))**2)
    # Minimización: Usando minimize de scipy.optimize:
    return minimize(func, a0).x[0] 

ranks = np.array(corpus_freqs.index) + 1
frecs = np.array(corpus_freqs['Frecuencia'])

a_hat = calculate_alpha(ranks, frecs)

def plot_generate_zipf(alpha: np.float64, ranks: np.array, freqs: np.array) -> None:
    plt.plot(np.log(ranks), -a_hat*np.log(ranks) + np.log(frecs[0]), color='r', label='Aproximación Zipf')

plot_generate_zipf(a_hat, ranks, frecs)
plt.plot(np.log(ranks), np.log(frecs), color='b', label='Distribución original')
plt.title('Idioma Pikachu')
plt.xlabel('Rango (log)')
plt.ylabel('Frecuencia (log)')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# Oración de ejemplo
import random
print("\nOración de ejemplo:")
for i in range(7):
    word = corpus.split()[random.randint(0, len(corpus.split()))]
    print(word, end=' ')
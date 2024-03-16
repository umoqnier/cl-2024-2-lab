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

# + [markdown] editable=true id="QybO5wX9cmJW" slideshow={"slide_type": "slide"}
# # 3. Propiedades estadísticas del lenguaje natural

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ## Objetivo

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Explorar propiedades estadísticas del lenguaje natural
# - Observar si se cumplen propiedades como:
#     - La distribución de Zipf
#     - La distribución de Heap
# - Observar como impacta la normalización

# + [markdown] editable=true id="__j_u2KcF3hz" slideshow={"slide_type": "subslide"}
# ## Perspectivas formales

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Fueron el primer acercamiento al procesamiento del lenguaje natural. Sin embargo tienen varias **desventajas**
# - Requieren **conocimiento previo de la lengua**
# - Las herramientas son especificas de la lengua
# - Los fenomenos que se presentan son muy amplios y dificilmente se pueden abarcar con reglas formales (muchos casos especiales)
# - Las reglas tienden a ser rigidas y no admiten incertidumbre en el resultado

# + [markdown] editable=true id="9eXBkslNwd7-" slideshow={"slide_type": "subslide"}
# ## Perspectiva estadística

# + [markdown] editable=true id="BTrnxln6wt--" slideshow={"slide_type": "fragment"}
# - Puede integrar aspectos de la perspectiva formal
# - Lidia mejor con la incertidumbre y es menos rigida que la perspectiva formal
# - No requiere conocimiento profundo de la lengua. Se pueden obtener soluciones de forma no supervisada

# + [markdown] editable=true id="axPN_dt4xtnp" slideshow={"slide_type": "subslide"}
# ### Modelos estadísticos

# + [markdown] editable=true id="IL53Cz22xv_K" slideshow={"slide_type": "fragment"}
# - Las **frecuencias** juegan un papel fundamental para hacer una descripción acertada del lenguaje
# - Las frecuencias nos dan información de la **distribución de tokens**, de la cual podemos estimar probabilidades.
# - Existen **leyes empíricas del lenguaje** que nos indican como se comportan las lenguas a niveles estadísticos
# - A partir de estas leyes y otras reglas estadísticas podemos crear **modelos del lenguaje**; es decir, asignar probabilidades a las unidades lingüísticas

# + editable=true id="N9JDvDSHub2V" slideshow={"slide_type": ""}
# Bibliotecas
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 4]
from re import sub
import numpy as np
import pandas as pd
from itertools import chain

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 141, "status": "ok", "timestamp": 1695680112256, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="zP8gw3oLch_I" outputId="65c27951-9a98-45ad-b282-90d7e5933b08" slideshow={"slide_type": ""}
mini_corpus = """Humanismo es un concepto polisémico que se aplica tanto al estudio de las letras humanas, los
estudios clásicos y la filología grecorromana como a una genérica doctrina o actitud vital que
concibe de forma integrada los valores humanos. Por otro lado, también se denomina humanis-
mo al «sistema de creencias centrado en el principio de que las necesidades de la sensibilidad
y de la inteligencia humana pueden satisfacerse sin tener que aceptar la existencia de Dios
y la predicación de las religiones», lo que se aproxima al laicismo o a posturas secularistas.
Se aplica como denominación a distintas corrientes filosóficas, aunque de forma particular,
al humanismo renacentista1 (la corriente cultural europea desarrollada de forma paralela al
Renacimiento a partir de sus orígenes en la Italia del siglo XV), caracterizado a la vez por su
vocación filológica clásica y por su antropocentrismo frente al teocentrismo medieval
"""
words = mini_corpus.replace("\n", " ").split(" ")
len(words)
# -

words[:10]

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 189, "status": "ok", "timestamp": 1695680133678, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="UyVe47bxzVKn" outputId="05badd03-2abb-48ae-f0b2-8b080a741a21"
vocabulary = Counter(words)
vocabulary.most_common(10)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 145, "status": "ok", "timestamp": 1695680188436, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="lcByyl9wz4Bf" outputId="27f40707-09f9-4e9b-8bc5-858ac2a7eb21"
len(vocabulary)


# + id="X-_K4QTKdYkF"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 467} editable=true executionInfo={"elapsed": 448, "status": "ok", "timestamp": 1695680259751, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="Rrq5cTSL0LTg" outputId="987e63c2-a8c0-460d-ec42-3435d9184516" slideshow={"slide_type": ""}
frequencies = get_frequencies(vocabulary, 100)
plot_frequencies(frequencies)
# -

plot_frequencies(frequencies, log_scale=True)

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### ¿Qué pasará con más datos? 📊
# -

# ## Ley Zipf

# + [markdown] id="zhff2_1Gl-fj"
# Exploraremos el Corpus de Referencia del Español Actual [CREA](https://www.rae.es/banco-de-datos/crea/crea-escrito)
# -

# <center><img src="img/crea.png"></center>

# !head corpora/zipf/crea_full.csv

# + colab={"base_uri": "https://localhost:8080/", "height": 363} executionInfo={"elapsed": 1446, "status": "ok", "timestamp": 1709598025605, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ORaN-mExS07f" outputId="44d791da-a7c6-4301-d5df-64ae353e155f"
corpus_freqs = pd.read_csv('corpora/zipf/crea_full.csv', delimiter='\t', encoding="latin-1")
#N = len(set(chain(*[list(str(w)) for w in corpus_freqs['words'].to_list()])))
corpus_freqs.head(10)
# -

corpus_freqs.iloc[10]

corpus_freqs[corpus_freqs["words"] == "barriga"]

corpus_freqs["freq"].plot(marker="o")
plt.title('Ley de Zipf en el CREA')
plt.xlabel('rank')
plt.ylabel('freq')
plt.show()

corpus_freqs['freq'].plot(loglog=True, legend=False)
plt.title('Ley de Zipf en el CREA (log-log)')
plt.xlabel('log rank')
plt.ylabel('log frecuencia')
plt.show()

# + [markdown] editable=true id="O03JMfuSfYJl" slideshow={"slide_type": "slide"}
# ### Ley de Zipf

# + [markdown] editable=true id="E8Jozc6MgYa9" slideshow={"slide_type": "fragment"}
# - Notamos que las frecuencias entre lenguas siguen un patrón
# - Pocas palabras (tipos) son muy frecuentes, mientras que la mayoría de palabras ocurren pocas veces
#
# De hecho, la frecuencia de la palabra que ocupa la posición r en el rank, es proporcional a $\frac{1}{r}$ (La palabra más frecuente ocurrirá aproximadamente el doble de veces que la segunda palabra más frecuente en el corpus y tres veces más que la tercer palabra más frecuente del corpus, etc)
#
# $$f(w_r) \propto \frac{1}{r^α}$$
#
# Donde:
# - $r$ es el rank que ocupa la palabra en el corpus
# - $f(w_r)$ es la frecuencia de la palabra en el corpus
# - $\alpha$ es un parámetro, el valor dependerá del corpus o fenómeno que estemos observando

# + [markdown] editable=true id="LQwxUIU8jjpD" slideshow={"slide_type": "subslide"}
# ### Formulación de la Ley de Zipf:

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# $f(w_{r})=\frac{c}{r^{\alpha }}$
#
# En la escala logarítimica:
#
# $log(f(w_{r}))=log(\frac{c}{r^{\alpha }})$
#
# $log(f(w_{r}))=log (c)-\alpha log (r)$

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### ❓ ¿Cómo estimar el parámetro $\alpha$?

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 13976, "status": "ok", "timestamp": 1709598069231, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="SzdvuX_vf-Rm" outputId="5930a33e-079b-4893-dea5-9fc27d0db59c" slideshow={"slide_type": ""}
# Ver cápitulo 12. Distribución de Zipf de Victor Mijangos
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


# + editable=true slideshow={"slide_type": ""}
def plot_generate_zipf(alpha: np.float64, ranks: np.array, freqs: np.array) -> None:
    plt.plot(np.log(ranks), -a_hat*np.log(ranks) + np.log(frecs[0]), color='r', label='Aproximación Zipf')


# + colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 1157, "status": "ok", "timestamp": 1709598075410, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="CwUNoLdLiPw0" outputId="d945622f-cdae-4253-e92f-cb57d6e41f66"
plot_generate_zipf(a_hat, ranks, frecs)
plt.plot(np.log(ranks), np.log(frecs), color='b', label='Distribución original')
plt.xlabel('log ranks')
plt.ylabel('log frecs')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ## Ley de Heap

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# Relación entre el número de **tokens** y **tipos** de un corpus
#
# $$T \propto N^b$$
#
# Dónde:
#
# - $T = $ número de tipos
# - $N = $ número de tokens
# - $b = $ parámetro  

# + [markdown] editable=true id="EgBcTeNEcfgG" slideshow={"slide_type": "fragment"}
# - **TOKENS**: Número total de palabras dentro del texto (incluidas repeticiones)
# - **TIPOS**: Número total de palabras únicas en el texto
# -

# Obtenemos los tipos y tokens
total_tokens = corpus_freqs["freq"].sum()
total_types = len(corpus_freqs)

total_tokens, total_types

# +
# Ordenamos el corpus por frecuencia
corpus_freqs_sorted = corpus_freqs.sort_values(by='freq', ascending=False)

# Calculamos la frecuencia acumulada
corpus_freqs_sorted['cum_tokens'] = corpus_freqs_sorted['freq'].cumsum()

# Calculamos el número acumulado de tipos
corpus_freqs_sorted['cum_types'] = range(1, total_types + 1)
# -

corpus_freqs_sorted.head()

# Plot de la ley de Heap
plt.plot(corpus_freqs_sorted['cum_types'], corpus_freqs_sorted['cum_tokens'])
plt.xscale("log")
plt.yscale("log")
plt.xlabel('Types')
plt.ylabel('Tokens')
plt.title('Ley de Heap')
plt.show()

# + [markdown] editable=true id="3Lx8R7XF8ByR" slideshow={"slide_type": "slide"}
# ## ¿Otros idiomas? 🇧🇴 🇨🇦 🇲🇽

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Presentando `pyelotl` 🌽

# + editable=true slideshow={"slide_type": "fragment"}
# !pip install elotl

# + [markdown] editable=true id="D3VPGDxn9G6-" slideshow={"slide_type": "fragment"}
# - [Documentación](https://pypi.org/project/elotl/)
# - Paquete para desarrollo de herramientas de NLP enfocado en lenguas de bajos recursos digitales habladas en México

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 210, "status": "ok", "timestamp": 1695680798411, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="H-PWk45K8IZj" outputId="fe6762b3-afbd-43ac-df50-b40ba17c8390" slideshow={"slide_type": "subslide"}
from elotl import corpus as elotl_corpus


print("Name\t\tDescription")
for row in elotl_corpus.list_of_corpus():
    print(row)

# + [markdown] editable=true id="8suXldYhbX-a" slideshow={"slide_type": "fragment"}
# Cada corpus se pueden visualizar y navegar a través de interfaz web.
# - [Axolotl](https://axolotl-corpus.mx/)
# - [Tsunkua](https://tsunkua.elotl.mx/)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 443, "status": "ok", "timestamp": 1695680866391, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="TOmKkwj2-HbQ" outputId="93887ce3-f944-4cb8-9e68-fb728ebb88f7" slideshow={"slide_type": ""}
axolotl = elotl_corpus.load("axolotl")
for row in axolotl:
    print("Lang 1 (es) =", row[0])
    print("Lang 2 (nah) =", row[1])
    print("Variante =", row[2])
    print("Documento de origen =", row[3])
    break

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 213, "status": "ok", "timestamp": 1695680887340, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="r0tuk82I-2M1" outputId="7f057687-a587-41cb-9b91-d7ede0aa3baf"
tsunkua = elotl_corpus.load("tsunkua")
for row in tsunkua:
    print("Lang 1 (es) =", row[0])
    print("Lang 2 (oto) =", row[1])
    print("Variante =", row[2])
    print("Documento de origen =", row[3])
    break


# +
def extract_words_from_sentence(sentence: str) -> list:
    return sub(r'[^\w\s\']', ' ', sentence).lower().split()

def get_words(corpus: list) -> tuple[list, list]:
    words_l1 = []
    words_l2 = []
    for row in corpus:
        words_l1.extend(extract_words_from_sentence(row[0]))
        words_l2.extend(extract_words_from_sentence(row[1]))
    return words_l1, words_l2


# + id="_FBETQjlwN1K"
spanish_words_na, nahuatl_words = get_words(axolotl)
spanish_words_oto, otomi_words = get_words(tsunkua)
# -

spanish_words_na[:10]

nahuatl_words[:10]

# + [markdown] id="9HFEw6XJPjkA"
# ### Tokens

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 144, "status": "ok", "timestamp": 1695681129287, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="zjUmpt4AccEH" outputId="61a51735-d569-4c06-d7a4-9a7737dd22dd" slideshow={"slide_type": ""}
print("Número total de palabras en náhuatl (corpus 1):", len(nahuatl_words))
print("Número total de palabras en español (corpus 1):", len(spanish_words_na))
print("Número total de palabras en otomí (corpus 2):", len(otomi_words))
print("Número total de palabras en español (corpus 2):", len(spanish_words_oto))

# + [markdown] editable=true id="dQIvSFmkcIa-" slideshow={"slide_type": ""}
# ### ❓ ¿Porqué si son textos paralelos (traducciones) el número de palabras cambia tanto?

# + [markdown] id="qgv6d4QfPdRT"
# De manera general, por las diferencias inherentes de las lenguas para expresar los mismos conceptos, referencias, etc. De manera particular, estas diferencias revelan características morfológicas de las lenguas. El náhuatl es una lengua con tendencia aglutinante/polisintética, por lo tanto, tiene menos palabras pero con morfología rica que les permite codificar cosas que en lenguas como el Español aparecen en la sintaxis. Ejemplo:
#
# > titamaltlakwa - Nosotros comemos tamales

# + [markdown] id="Oq2ijWa_Pyx5"
# ### Tipos

# + id="7K_5WknZy5S-"
nahuatl_vocabulary = Counter(nahuatl_words)
nahuatl_es_vocabulary = Counter(spanish_words_na)
otomi_vocabulary = Counter(otomi_words)
otomi_es_vocabulary = Counter(spanish_words_oto)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 143, "status": "ok", "timestamp": 1695681422555, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="k3KiQCcROlC3" outputId="ef5d0ec5-a9ed-4e93-d880-8afd5ec0fd26"
otomi_vocabulary.most_common(20)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 160, "status": "ok", "timestamp": 1695681448329, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="Rk_oRR-tO7rg" outputId="30c6bd8c-53ee-4e11-c12a-a5be9160b7c7"
print("Tamaño del vocabulario de nahúatl:", len(nahuatl_vocabulary))
print("Tamaño del vocabulario de español (corpus 1):", len(nahuatl_es_vocabulary))
print("Tamaño del vocabulario de otomí:", len(otomi_vocabulary))
print("Tamaño del vocabulario de español (corpus 2):", len(otomi_es_vocabulary))

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### ❓ ¿Cómo cambiarían estas estadísticas si no filtramos los signos de puntuación?
# -

# Si no normalizamos aumenta el número de tipos lo cual "ensucia" los datos con los que vamos a trabajar. 
#
# Ejemplo: `algo != algo,`

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### ❓ ¿Cómo afecta la falta de normalización ortográfica en lenguas como el náhuatl
# -

# En lenguas como el nahúatl, la falta de normalización ortográfica y las variaciones diacrónicas del corpus, provocan que haya grafías diferentes que corresponden a una misma palabra. Ejemplo:
#
# - Yhuan-ihuan
#
# - Yn-in

print(nahuatl_vocabulary["in"])
print(nahuatl_vocabulary["yn"])

# ### Normalizador para el Nahúatl

# +
from elotl.nahuatl import orthography

normalizer = orthography.Normalizer("inali")
# -

help(normalizer)

nahuatl_words_normalized = [normalizer.normalize(word) for word in nahuatl_words]

nahuatl_norm_vocabulary = Counter(nahuatl_words_normalized)
print("Tamaño del vocabulario (tipos) ANTES de normalizar:", len(nahuatl_vocabulary))
print("Tamaño del vocabulario (tipos) DESPUÉS de normalizar:", len(nahuatl_norm_vocabulary))


# + id="g0AIsl6CQD1u"
def avg_len(tokens: list) -> float:
    return sum(len(token) for token in tokens) / len(tokens)


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 168, "status": "ok", "timestamp": 1695682715742, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="SAYvR6PrUWy7" outputId="bde55d9c-787d-48e6-8991-9a3849e30ad0"
print("Longitud promedio de palabras en nahúatl:", avg_len(nahuatl_words))
print("Longitud promedio de palabras en nahúatl (NORM):", avg_len(nahuatl_words_normalized))
print("Longitud promedio de palabras en otomí:", avg_len(otomi_words))
print("Longitud promedio de palabras en español (corpus 1):", avg_len(spanish_words_na))
print("Longitud promedio de palabras en español (corpus 2):", avg_len(spanish_words_oto))

# + [markdown] id="1WKtSLMGte__"
# #### Ejercicio: Obtener la palabra más larga de cada lista de palabras (15 min) (0.5 pt extra 🔥)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 146, "status": "ok", "timestamp": 1695683918598, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="Qx2pXnJ8oVHW" outputId="7a4be667-fee4-4341-9fed-b3fc7d2f1fcb"
print("Nahúatl", max(nahuatl_words, key=len))
print("Nahúatl (Español)", max(spanish_words_na, key=len))
print("Otomí", max(otomi_words, key=len))
print("Otomí (Español)", max(spanish_words_oto, key=len))


# + [markdown] id="tQS4_T6YtXwM"
# ### Comparación de longitudes promedio
#
# Calcular la longitud promedio de las palabras en la "cabeza" y en la "cola" de la distribución

# + id="jNC2tdecuhaZ"
def get_words_from_vocabulary(vocabulary: Counter, n: int, most_common=True) -> list:
    pairs = vocabulary.most_common(n) if most_common else vocabulary.most_common()[:-n-1:-1]
    return [pair[0] for pair in pairs]

words_head = get_words_from_vocabulary(otomi_vocabulary, 20)
words_tail = get_words_from_vocabulary(otomi_vocabulary, 20, most_common=False)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 245, "status": "ok", "timestamp": 1695684394482, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="yNbV0WL36CDt" outputId="cfe2459f-c548-4a39-e662-1bde46cf64d2"
words_tail

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 160, "status": "ok", "timestamp": 1695684405585, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="_PZMcNgn6EBE" outputId="11d2f1d8-6921-4599-bf3f-36014dbbb31a"
words_head

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 172, "status": "ok", "timestamp": 1695684412534, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="Zp8ITqTC4xPP" outputId="115ad096-da47-4cb9-b59b-8870a7780a1e"
print ("Longitud promedio de las palabras más frecuentes:", avg_len(words_head))
print ("Longitud promedio de las palabras menos frecuentes:", avg_len(words_tail))

# + [markdown] editable=true id="uUx3hte36KNy" slideshow={"slide_type": ""}
# ### ❓ ¿Por qué las palabras más frecuentes son más cortas?

# + [markdown] id="tVJMbsKW6QVT"
# Probablemente por cuestiones de eficiencia/economía del lenguaje. Representa menor "esfuerzo" ocupar un código pequeño para las palabras que tenemos que usar frecuentemente. Esto también se puede entender en términos de codificación óptima en teoría de la información. [Brevity Law](https://en.wikipedia.org/wiki/Brevity_law)

# + [markdown] id="--Q2tS5Vee6r"
# ### Data viz 📊

# + colab={"base_uri": "https://localhost:8080/", "height": 458} executionInfo={"elapsed": 497, "status": "ok", "timestamp": 1695684052992, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="g2VvHaJLeiz2" outputId="88ffdac2-e2d4-418b-8f3e-461ba0d9ad7b"
most_common_count = 300
nahuatl_freqs = get_frequencies(nahuatl_vocabulary, most_common_count)
plot_frequencies(nahuatl_freqs, f"Frequencies for Nahúatl {most_common_count} most common", log_scale=True)
# -

nahuatl_freqs = get_frequencies(nahuatl_norm_vocabulary, len(nahuatl_norm_vocabulary))
na_freqs = np.array(nahuatl_freqs)
na_ranks = np.array(range(1, len(na_freqs)+1))
alpha_na = calculate_alpha(ranks=na_ranks, frecs=na_freqs)

plot_generate_zipf(alpha_na, ranks=na_ranks, freqs=na_freqs)
plt.plot(np.log(na_ranks), np.log(na_freqs), color='b', label='Distribución original')
plt.xlabel('log ranks')
plt.ylabel('log frecs')
plt.title("Distribución de Zip para Nahuátl")
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 461} executionInfo={"elapsed": 493, "status": "ok", "timestamp": 1695684076639, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="2YVcsGA0fsZx" outputId="42835fb3-00f0-4c11-92d2-b8d1e547e7a5"
otomi_freqs = get_frequencies(otomi_vocabulary, most_common_count)
plot_frequencies(otomi_freqs, f"Frequencies for Otomí {most_common_count} most common", log_scale=True)
# -

otomi_freqs = get_frequencies(otomi_vocabulary, len(otomi_vocabulary))
oto_freqs = np.array(otomi_freqs)
oto_ranks = np.array(range(1, len(oto_freqs)+1))
alpha_oto = calculate_alpha(ranks=oto_ranks, frecs=oto_freqs)

plot_generate_zipf(alpha_oto, ranks=oto_ranks, freqs=oto_freqs)
plt.plot(np.log(oto_ranks), np.log(oto_freqs), color='b', label='Distribución original')
plt.xlabel('log ranks')
plt.ylabel('log frecs')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 457} executionInfo={"elapsed": 638, "status": "ok", "timestamp": 1695684082980, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ODHIhYWYf85G" outputId="4d11641f-7c36-48ea-cb03-8e3ae1b182b4"
spanish_freqs = get_frequencies(nahuatl_es_vocabulary, most_common_count)
plot_frequencies(spanish_freqs, f"Frequencies for Spanish (Corpus 1) {most_common_count} most common", log_scale=True)

# + [markdown] editable=true id="XkCFc70vmQyk" slideshow={"slide_type": ""}
# ### Práctica 3: Stop! stop! 🚏 My Zipf's distribution can talk!!! 🙀

# + [markdown] id="-k9g8PNisX0f"
# **Fecha de entrega: Domingo 17 de Marzo 2024 - 11:59pm**
#
# -

# - Comprobar si las *stopwords* que encontramos en paqueterias de *NLP* coinciden con las palabras más comúnes obtenidas en Zipf
#     - Utilizar el [corpus CREA](https://corpus.rae.es/frec/CREA_total.zip)
#     - Realizar una nube de palabras usando las stopwords de paqueteria y las obtenidas através de Zipf
#     - Responder las siguientes preguntas:
#         - ¿Obtenemos el mismo resultado? Si o no y ¿Porqué?
# - Comprobar si Zipf se cumple para un lenguaje artificial creado por ustedes
#   - Deberán darle un nombre a su lenguaje
#   - Mostrar una oración de ejemplo
#   - Pueden ser una secuencia de caracteres aleatorios
#   - Tambien pueden definir el tamaño de las palabras de forma aleatoria



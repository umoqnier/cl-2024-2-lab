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

# + [markdown] id="UndAvhCx62c5"
# # 6. Modelos del lenguaje

# + [markdown] id="mmMeLR-EcPCW"
# ## Objetivos

# + [markdown] id="yWZRbpatcQeA"
# - Crear modelos del lenguaje a partir de un corpus en inglés
#     - Modelo de bigramas
#     - Modelo de trigramas

# + [markdown] id="nNyww-_MsZb6"
# > Un modelo del lenguaje es un modelo estadístico que asigna probabilidades a cadenas dentro de un lenguaje - Jurafsky, 2000
#
# $$ \mu = (\Sigma, A, \Pi)$$
#
# Donde:
# - $\mu$ es el modelo del lenguaje
# - $\Sigma$ es el vocabulario
# - $A$ es el tensor que guarda las probabilidades
# - $\Pi$ guarda las probabilidades iniciales

# + [markdown] id="YQV3jCdc689o"
# - Este modelo busca estimar la probabilidad de una secuencia de tokens
# - Pueden ser palabras, caracteres o tokens
# - Se pueden considerar varios escenarios para la creación de estos modelos
# - Si podemos estimar la probabilidad de una unidad lingüística (palabras, tokens, oracines, etc), podemos usarlar de formas insospechadas

# + [markdown] id="fARPrQsSfA-s"
# ## I saw a cat in a mat

# + [markdown] id="ZuAH9oFwe4Os"
# <img src="https://lena-voita.github.io/resources/lectures/lang_models/general/i_saw_a_cat_prob.gif">

# + [markdown] id="_QdN5z6A5lsx"
# ## Aplicaciones

# + [markdown] id="DA3h5XD_5nhK"
# - Traducción automática
# - Completado de texto
# - Generación de texto

# + [markdown] id="k2kSHeo6dNVi"
# ![](https://lena-voita.github.io/resources/lectures/lang_models/examples/suggest-min.png)
# Tomado de [Lena Voita](https://lena-voita.github.io/nlp_course/language_modeling.html)

# + [markdown] id="grLl2l7s56gx"
# ## De los bigramas a los n-gramas

# + [markdown] id="0sFQj5FyoW0A"
# - Para bigramas tenemos la propiedad de Markov
# - Para $n > 2$ las palabras dependen de mas elementos
#     - Trigramas
#     - 4-gramas
# - En general para un modelo de n-gramas se toman en cuenta $n-1$ elementos

# + [markdown] id="VpeyW0YpdFqD"
# ## Programando nuestros modelos del lenguaje

# + [markdown] id="E4NmkcRVg7UI"
# Utilizaremos un [corpus](https://www.nltk.org/book/ch02.html) en inglés disponible en NLTK

# + colab={"base_uri": "https://localhost:8080/"} id="8LPzWyu5g5ra" outputId="91b4e453-52c5-4204-f7c6-bd382737e7b9"
import nltk
nltk.download('reuters')
nltk.download('punkt')

# + id="nTQ85iSZhQxA"
# Computando números
import numpy as np
# Corpus
from nltk.corpus import reuters
# Para crear ngramas
from nltk import ngrams
# Utilidades para manejar las probabilidades
from collections import Counter, defaultdict

# + id="xf1cj5x7hcxf" outputId="a246ab70-cdc6-42dc-efd6-aac2918de64c" colab={"base_uri": "https://localhost:8080/"}
len(reuters.sents())

# + id="u_7IxYFZgYgi"
import re

def preprocess(sent: list[str]) -> list[str]:
    """Función de preprocesamiento

    Agrega tokens de inicio y fin, normaliza todo a minusculas
    """
    result = [word.lower() for word in sent]
    # Al final de la oración
    result.append("<EOS>")
    result.insert(0, "<BOS>")
    return result


# + colab={"base_uri": "https://localhost:8080/"} id="0xKE9xTd1-1L" outputId="6ab41489-09f2-40f2-8047-6a2188ee748d"
print(reuters.sents()[11])
preprocess(reuters.sents()[11])

# + colab={"base_uri": "https://localhost:8080/"} id="TkmCVL5WQnDs" outputId="a841080e-49a8-48b2-e3f2-337e9c98c4b6"
list(ngrams(reuters.sents()[0], 3))

# + [markdown] id="Dl4yno22h6nT"
# ### Obteniendo modelo de trigramas

# + id="hkkgPOVZh9YI"
trigram_model = defaultdict(lambda: defaultdict(lambda: 0))

# + id="Bf6QyJGehPqE"
N = 3
for sentence in reuters.sents():
    # Obtenemos los ngramas normalizados
    n_grams = ngrams(preprocess(sentence), N)
    # Guardamos los bigramas en nuestro diccionario
    for w1, w2, w3 in n_grams:
        trigram_model[(w1, w2)][w3] += 1

# + colab={"base_uri": "https://localhost:8080/"} id="Qu_-y1a43sGY" outputId="42940f49-ec2c-48c1-adf0-2b34338c5971"
trigram_model["<BOS>", "the"]

# + colab={"base_uri": "https://localhost:8080/"} id="SVWgG_-Yn-Ps" outputId="d74762f1-2110-4759-ba33-36b353521e0c"
for i, entry in enumerate(trigram_model.items()):
    print(entry)
    if i == 3:
        break

# + id="vkWwqLDaliaf"
VOCABULARY = set([word.lower() for sent in reuters.sents() for word in sent])
# # +2 por los tokens <BOS> y <EOS>
VOCABULARY_SIZE = len(VOCABULARY) + 2


# + id="IXIq0KbPkFHE"
def calculate_model_probabilities(model: defaultdict) -> defaultdict:
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for prefix in model:
        # Todas las veces que vemos la key seguido de cualquier cosa
        total = float(sum(model[prefix].values()))
        for next_word in model[prefix]:
            # Laplace smothing
            #result[prefix][next_word] = (model[prefix][next_word] + 1) / (total + VOCABULARY_SIZE)
            # Without smothing
            result[prefix][next_word] = model[prefix][next_word] / total
    return result


# + id="oiWDoH1OnbqG"
trigram_probs = calculate_model_probabilities(trigram_model)

# + colab={"base_uri": "https://localhost:8080/"} id="dFDn06SNpn1E" outputId="28780433-141e-4461-8155-412f3824d87c"
sorted(dict(trigram_probs["this","is"]).items(), key=lambda x:-1*x[1])


# + [markdown] id="7Vu_qYzFp3a_"
# ## Calculando la siguiente palabra más probable

# + id="MTS9W_xZp61d"
def get_likely_words(model_probs: defaultdict, context: str, top_count: int=10) -> list[tuple]:
    """Dado un contexto obtiene las palabras más probables

    Params
    ------
    model_probs: defaultdict
        Probabilidades del modelo
    context: str
        Contexto con el cual calcular las palabras más probables siguientes
    top_count: int
        Cantidad de palabras más probables. Default 10
    """
    history = tuple(context.split())
    return sorted(dict(model_probs[history]).items(), key=lambda prob: -1*prob[1])[:top_count]


# + colab={"base_uri": "https://localhost:8080/"} id="leg6tl_esCDu" outputId="81717a5a-9031-44b4-a153-0d7541b6e481"
get_likely_words(trigram_probs, "<BOS> the", top_count=3)

# + [markdown] id="3j_EmFa0sd4U"
# ### Estrategias de generación

# + id="aWk0CSpisijj"
from random import randint

def get_next_word(words: list) -> str:
    # Strategy here
    return words[0][0]

def get_next_word(words: list) -> str:
    return words[randint(0, len(words)-1)][0]


# + colab={"base_uri": "https://localhost:8080/", "height": 35} id="kDm7pbVQs9Oa" outputId="63be0d4d-4de7-4b6f-a58f-f98bf5cfbbbc"
get_next_word(get_likely_words(trigram_probs, "<BOS> the", 50))

# + [markdown] id="KA3xxV8itJkb"
# ### Generando texto

# + id="ecDfhBcLtLeJ"
MAX_TOKENS = 30
def generate_text(model: defaultdict, history: str, tokens_count: int) -> None:
    next_word = get_next_word(get_likely_words(model, history, top_count=30))
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    generate_text(model, history.split()[1]+ " " + next_word, tokens_count)


# + colab={"base_uri": "https://localhost:8080/"} id="Ot-amn-AxPQC" outputId="78a2ab95-93fb-42d1-fa7f-648d26a8ed44"
sentence = "<BOS> they"
print(sentence, end=" ")
generate_text(trigram_probs, sentence, 0)


# + [markdown] id="TDMEyBeR6YF5"
# ## Calculando la probabilidad de una oración

# + id="DYdXiOrv-35T"
def calculate_sent_prob(model: defaultdict, sentence: str, n: int) -> float:
    n_grams = ngrams(preprocess(sentence), n)
    p = 0.0
    for gram in n_grams:
        if n == 3:
            key = (gram[0], gram[1])
            value = gram[2]
        elif n == 2:
            key = gram[0]
            value = gram[1]
        try:
            p += np.log(model[key][value])
        except:
            p += 0.0
    return p


# + colab={"base_uri": "https://localhost:8080/"} id="-N9mzNpzV581" outputId="a9111d27-d5f9-4a50-8032-1928c87858d7"
sentence = reuters.sents()[10]
print(" ".join(sentence))
calculate_sent_prob(trigram_probs, reuters.sents()[10], n=3)

# + colab={"base_uri": "https://localhost:8080/"} id="hS0xDieGAUng" outputId="1e6e074b-2a9b-4eca-bc3f-40d1350968e8"
sentence = reuters.sents()[100]
print(" ".join(sentence))
calculate_sent_prob(trigram_probs, reuters.sents()[100], n=3)

# + [markdown] id="COr2ki7q7oLs"
# ## Evaluación de modelos

# + colab={"base_uri": "https://localhost:8080/"} id="grH2p6rue3t-" outputId="63038481-14ad-4e58-9a56-599f7f41f3e6"
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(reuters.sents(), test_size=0.3)

print("Train data", len(train_data))
print("tests data", len(test_data))

# + colab={"base_uri": "https://localhost:8080/"} id="y6Mvps9CntJ7" outputId="c7e1eb29-2bc3-4118-f520-1bff09a64d8c"
pp=[]
for sentence in test_data:
  #1. Normalizamos y agregamos símbolos especiales:

  #Log perplexity calculada para cada oracion:
  log_prob=calculate_sent_prob(trigram_probs, sentence, 3)
  perplexity=-(log_prob/len(sentence)-1)
  pp.append(perplexity)


#promedio de las log perplexity:
total_perplexity= sum(pp) / len(pp)
print(total_perplexity)

# + [markdown] id="_B3jZhXvfOhU"
# Para evaluar un modelo se utiliza como medida la (log) perplejidad o *perplexity*

# + [markdown] id="mIpYWhzEBJCo"
# ### Comparando con un modelo de bigramas

# + id="YRqfISpJte2H"
bigram_model = defaultdict(lambda: defaultdict(lambda: 0))

# + id="yaQaaDbPte2Q"
N = 2
for sentence in reuters.sents():
    # Obtenemos los ngramas normalizados
    n_grams = ngrams(preprocess(sentence), N)
    # Guardamos los bigramas en nuestro diccionario
    for w1, w2 in n_grams:
        bigram_model[w1][w2] += 1

# + id="qy3OA06ate2S" colab={"base_uri": "https://localhost:8080/"} outputId="3e9e09d1-393a-437e-e698-c2ad8e24a758"
bigram_model["problems"]

# + colab={"base_uri": "https://localhost:8080/"} outputId="0fb1be6c-9249-4a11-c7f1-f8dac138c64a" id="f-s0A9tste2U"
for i, entry in enumerate(bigram_model.items()):
    print(entry)
    if i == 3:
        break

# + id="uZLt4ZxXsmG-"
bigram_probs = calculate_model_probabilities(bigram_model)

# + colab={"base_uri": "https://localhost:8080/"} id="AMHBLBUDuQO9" outputId="6242a734-9649-4ddd-8396-15b4a7b3eaad"
sorted(dict(bigram_probs["<BOS>"]).items(), key=lambda x:-1*x[1])[:10]

# + colab={"base_uri": "https://localhost:8080/"} id="X1u7n_1_urMR" outputId="838f1de0-786f-4baf-d2b7-a34401369cc4"
calculate_sent_prob(bigram_probs, reuters.sents()[100], 2)

# + colab={"base_uri": "https://localhost:8080/"} id="rO3bR8RTvE8g" outputId="ed000773-f5cd-4dfb-9538-bab81d682c88"
pp=[]
for sentence in test_data:
  #1. Normalizamos y agregamos símbolos especiales:

  #Log perplexity calculada para cada oracion:
  log_prob=calculate_sent_prob(bigram_probs, sentence, 2)
  perplexity=-(log_prob/len(sentence)-1)
  pp.append(perplexity)


#promedio de las log perplexity:
total_perplexity= sum(pp) / len(pp)
print(total_perplexity)

# + [markdown] id="AO9uvsff3wUe"
# ## Práctica 6: Evaluación de modelos de lenguaje
#
# **Fecha de entrega: 21 de abril de 2024**

# + [markdown] id="sBdjG6yY4FoA"
# - Crear un par de modelos del lenguaje usando un **corpus en español**
#     - Corpus: El Quijote
#         - URL: https://www.gutenberg.org/ebooks/2000
#     - Modelo de n-gramas con `n = [2, 3]`
#     - Hold out con `test = 30%` y `train = 70%`
# - Evaluar los modelos y reportar la perplejidad de cada modelo
#   - Comparar los resultados entre los diferentes modelos del lenguaje (bigramas, trigramas)
#   - ¿Cual fue el modelo mejor evaluado? ¿Porqué?

# + [markdown] id="dTAH6joW8IXW"
# # Referencias
#
# - Mucho del código mostrado fue tomado del trabajo de la Dr. Ximena Guitierrez-Vasques
# - https://lena-voita.github.io/nlp_course/language_modeling.html

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

# Computando números
import numpy as np
# Para crear ngramas
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
# Utilidades para manejar las probabilidades
from collections import Counter, defaultdict

with open('2000-0.txt', 'r', encoding='utf-8-sig') as file:
    text = file.read()
    text = text.replace('\n', ' ')
    text = text.replace(',', '')
sentences = sent_tokenize(text)
sentences = [word_tokenize(sent) for sent in sentences]

# +
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(sentences, test_size=0.3)

print("Train data", len(train_data))
print("tests data", len(test_data))

# +
import re

def preprocess(sent: list[str]) -> list[str]:
    """Función de preprocesamiento

    Agrega tokens de inicio y fin, normaliza todo a minusculas
    """
    result = [word.lower() for word in sent]
    # Al final de la oración
    result[-1] = "<EOS>"
    result.insert(0, "<BOS>")
    return result


# -

trigram_model = defaultdict(lambda: defaultdict(lambda: 0))

N = 3
for sentence in train_data:
    # Obtenemos los ngramas normalizados
    n_grams = ngrams(preprocess(sentence), N)
    # Guardamos los bigramas en nuestro diccionario
    for w1, w2, w3 in n_grams:
        trigram_model[(w1, w2)][w3] += 1

VOCABULARY = set([word.lower() for sent in train_data for word in sent])
# # +2 por los tokens <BOS> y <EOS>
VOCABULARY_SIZE = len(VOCABULARY) + 2


def calculate_model_probabilities(model: defaultdict) -> defaultdict:
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for prefix in model:
        # Todas las veces que vemos la key seguido de cualquier cosa
        total = float(sum(model[prefix].values()))
        for next_word in model[prefix]:
            #Laplace smothing
            #result[prefix][next_word] = (model[prefix][next_word] + .1) / (total + VOCABULARY_SIZE*.1)
            # Without smothing
            result[prefix][next_word] = model[prefix][next_word] / total
    return result


trigram_probs = calculate_model_probabilities(trigram_model)


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


get_likely_words(trigram_probs, "ha de", top_count=3)

# +
from random import randint

def get_next_word(words: list) -> str:
    # Strategy here
    return words[0][0]

def get_next_word(words: list) -> str:
    return words[randint(0, len(words)-1)][0]


# -

MAX_TOKENS = 30
def generate_text(model: defaultdict, history: str, tokens_count: int) -> None:
    next_word = get_next_word(get_likely_words(model, history, top_count=30))
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    generate_text(model, history.split()[1]+ " " + next_word, tokens_count)


sentence = "dulcinea del"
print(sentence, end=" ")
generate_text(trigram_probs, sentence, 0)


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
            p += np.log(model[key][value]+1*10**-20)
        except:
            p += 0.0
    return p


bigram_model = defaultdict(lambda: defaultdict(lambda: 0))

N = 2
for sentence in train_data:
    # Obtenemos los ngramas normalizados
    n_grams = ngrams(preprocess(sentence), N)
    # Guardamos los bigramas en nuestro diccionario
    for w1, w2 in n_grams:
        bigram_model[w1][w2] += 1

bigram_probs = calculate_model_probabilities(bigram_model)

# +
#Evaluación de modelos en datos de entrenamiento

# +
pp=[]

for sentence in train_data:
  #1. Normalizamos y agregamos símbolos especiales:

  #Log perplexity calculada para cada oracion:
  log_prob=calculate_sent_prob(bigram_probs, sentence, 2)
  perplexity=-(log_prob/len(sentence)-1)
  pp.append(perplexity)


#promedio de las log perplexity:
total_perplexity= sum(pp) / len(pp)
print(total_perplexity)

# +
pp=[]

for sentence in train_data:
  #1. Normalizamos y agregamos símbolos especiales:

  #Log perplexity calculada para cada oracion:
  log_prob=calculate_sent_prob(trigram_probs, sentence, 3)
  perplexity=-(log_prob/len(sentence)-1)
  pp.append(perplexity)


#promedio de las log perplexity:
total_perplexity= sum(pp) / len(pp)
print(total_perplexity)

# +
#Evaluación en datos de prueba

# +
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

# +
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
# -

# ## ¿Cual fue el modelo mejor evaluado? ¿Porqué?
# ### El mejor evaluado en los datos de entrenamiento fue el modelo de trigramas, sin embargo,  el modelo de bigramas fue mejor evaluado en los datos de prueba. Mi hipótesis es que aunque un modelo de trigramas como vimos en clase reduce la perplejidad, los datos de entrenamiento no fueron suficientes para poder "generalizar" las probabilidades de los trigramas. Es muy probable encontrarse palabras nuevas en los datos de entrenamiento de un corpus tan especial y pequeño.

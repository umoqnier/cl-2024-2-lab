import nltk
import numpy as np
from nltk import ngrams
from collections import Counter, defaultdict
import re
nltk.download('punkt')

def lee_corpus(file_name: str) -> str:
    with open(file_name, "r") as f:
        return f.read()
    
quijote = lee_corpus("El_Quijote.txt")
sents = nltk.sent_tokenize(quijote)
corpus = [nltk.word_tokenize(sent) for sent in sents]


def preprocess(sent: list[str]) -> list[str]:
    """Función de preprocesamiento

    Agrega tokens de inicio y fin, normaliza todo a minusculas
    """
    result = [word.lower() for word in sent]
    result.append("<EOS>")
    result.insert(0, "<BOS>")
    return result

def generate_trigram_model(corpus, N):
    ngram_model = defaultdict(lambda: defaultdict(lambda: 0))
    for sentence in corpus:
        n_grams = ngrams(preprocess(sentence), N)
        for ngram in n_grams:
            ngram_model[ngram[:-1]][ngram[-1]] += 1
    return ngram_model

bigram_model = defaultdict(lambda: defaultdict(lambda: 0))
N = 2
for sentence in corpus:
    # Obtenemos los ngramas normalizados
    n_grams = ngrams(preprocess(sentence), N)
    # Guardamos los bigramas en nuestro diccionario
    for w1, w2 in n_grams:
        bigram_model[w1][w2] += 1

trigram_model = generate_trigram_model(corpus, 3)

VOCABULARY = set([word.lower() for sent in corpus for word in sent])
# +2 por los tokens <BOS> y <EOS>
VOCABULARY_SIZE = len(VOCABULARY) + 2

def calculate_model_probabilities(model: defaultdict) -> defaultdict:
    result = defaultdict(lambda: defaultdict(lambda: 0))
    for prefix in model:
        # Todas las veces que vemos la key seguido de cualquier cosa
        total = float(sum(model[prefix].values()))
        for next_word in model[prefix]:
            result[prefix][next_word] = model[prefix][next_word] / total
    return result

trigram_probs = calculate_model_probabilities(trigram_model)
bigram_probs = calculate_model_probabilities(bigram_model)

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
    if len(context.split()) == 1:
        history = context
    else:
        history = tuple(context.split())
    return sorted(dict(model_probs[history]).items(), key=lambda prob: -1*prob[1])[:top_count]

from random import randint

def get_next_word(words: list) -> str:
    return words[randint(0, len(words)-1)][0]

MAX_TOKENS = 30
def generate_text(model: defaultdict, history: str, tokens_count: int) -> None:
    next_word = get_next_word(get_likely_words(model, history, top_count=30))
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    if len(history.split()) == 1:
        generate_text(model, next_word, tokens_count)
    else:
        generate_text(model, history.split()[1]+ " " + next_word, tokens_count)

print("\nProbando los modelos generados:")
print("Modelo de trigramas:")
print("en esta", end=" ")
generate_text(trigram_probs, "en esta", 0)
print("\nModelo de bigramas:")
print("esta", end=" ")
generate_text(bigram_probs, "esta", 0)


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

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(corpus, test_size=0.3)

def calculate_perplexities(test_data, model, n):
    pp = []
    for sentence in test_data:
        # Log perplexity calculada para cada oracion:
        log_prob = calculate_sent_prob(model, sentence, n)
        perplexity = -(log_prob / len(sentence) - 1)
        pp.append(perplexity)
    # Promedio de las perplexities
    return sum(pp) / len(pp)

pp_trigram = calculate_perplexities(test_data, trigram_probs, 3)
pp_bigram = calculate_perplexities(test_data, bigram_probs, 2)

print("\n\nPerplejidad del modelo de trigramas:", pp_trigram)
print("Perplejidad del modelo de bigramas:", pp_bigram)

'''
Preguntas
Comparación de resultados: El modelo de trigramas tiene una perplejidad de aproximadamente 2.7, mientras que el de bigramas es de aproximadamente 4.9.

¿Cual fue el modelo mejor evaluado? ¿Porqué?
El modelo mejor evaluado fue el de trigramas, pues tiene menor perplejidad. Esto se debe a que el modelo de trigramas considera más contexto que el de bigramas, lo que le permite hacer mejores predicciones.
'''

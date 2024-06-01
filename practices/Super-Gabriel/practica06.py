import requests
import nltk
from sklearn.model_selection import train_test_split


# descargando texto, la liga del notebook no funcionaba, encontré esta.
quijote = requests.get('https://fegalaz.usc.es/~gamallo/aulas/lingcomputacional/corpus/quijote-es.txt').text

# haciendo un preprocesamiento y tokenizando
tokens = nltk.word_tokenize(quijote)
text = list(map(str.lower, tokens))

# haciendo la división del conjunto de entrenamiento y prueba
train, test = train_test_split(text, test_size=0.3, random_state=42)

import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


train_data_bigram, padded_bigrams = padded_everygram_pipeline(2, train)
train_data_trigram, padded_trigrams = padded_everygram_pipeline(3, train)

from nltk.lm import Laplace

bigram_model = Laplace(2)
bigram_model.fit(train_data_bigram, padded_bigrams)

trigram_model = Laplace(3)
trigram_model.fit(train_data_trigram, padded_trigrams)

bigram_perplexity = bigram_model.perplexity(test)
trigram_perplexity = trigram_model.perplexity(test)

print("Perplejidad del modelo de bigramas:", bigram_perplexity)
print("Perplejidad del modelo de trigramas:", trigram_perplexity)


"""

en este caso la perplejidad de los modelos no variaron mucho pero ligeramente tuvo menor
perplejidad el modelo de bigramas lo cual me extraña, pero puede ser por el tamaño y la
naturaleza del corpus por lo cual se obtuvieron esos resultados.

"""

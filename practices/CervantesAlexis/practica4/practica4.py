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

import elotl.corpus
import elotl.nahuatl.orthography
axolotl = elotl.corpus.load("axolotl")

import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import brown
nltk.download('brown')

CORPORA_PATH = "corpora/tokenization/"

axolotl_words = [word for row in axolotl for word in row[1].lower().split()]

train_rows_count = len(axolotl_words) - round(len(axolotl_words)*.30)

axolotl_train_words = axolotl_words[:train_rows_count]
axolotl_test_words = axolotl_words[train_rows_count:]

# +
import requests

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)


# -

write_plain_text_corpus(" ".join(axolotl_train_words), CORPORA_PATH + "axolotl_plain")

# !subword-nmt learn-bpe -s 100 < corpora/tokenization/axolotl_plain.txt > models/tokenization/axolotl.model

axolotl_test_types = Counter(axolotl_test_words)

write_plain_text_corpus(" ".join(axolotl_test_words), CORPORA_PATH + "axolotl_plain_test")

# !subword-nmt apply-bpe -c models/tokenization/axolotl.model < corpora/tokenization/axolotl_plain_test.txt > corpora/tokenization/axolotl_tokenized.txt

with open(CORPORA_PATH + "axolotl_tokenized.txt") as f:
    axolotl_test_tokenized = f.read().split()

len(axolotl_test_tokenized)

print(axolotl_test_tokenized[:10])

axolotl_test_tokenized_types = Counter(axolotl_test_tokenized)

axolotl_test_tokenized_types.most_common(20)

print("Axolotl Information")
print("Tokens:", len(axolotl_test_words))
print("Types (word-base):", len(axolotl_test_types))
print("Types (native BPE):", len(axolotl_test_tokenized_types))
print("TTR (word-base):", len(axolotl_test_types)/len(axolotl_test_words))
print("TTR (BPE):", len(axolotl_test_tokenized_types)/len(axolotl_test_tokenized))

# +
from collections import Counter

brown_corpus = [word for word in brown.words() if re.match("\w", word)]
print(brown_corpus[0])
print("Tokens:", len(brown_corpus))
print("Tipos:", len(Counter(brown_corpus)))
# -

train_rows_brown = len(brown_corpus) - round(len(brown_corpus)*.30)

brown_train_words = brown_corpus[:train_rows_brown]
brown_test_words = brown_corpus[train_rows_brown:]

write_plain_text_corpus(" ".join(brown_train_words), CORPORA_PATH + "brown_plain")

# !subword-nmt learn-bpe -s 100 < corpora/tokenization/brown_plain.txt > models/tokenization/brown.model

brown_test_types = Counter(brown_test_words)

write_plain_text_corpus(" ".join(brown_test_words), CORPORA_PATH + "brown_plain_test")

# !subword-nmt apply-bpe -c models/tokenization/brown.model < corpora/tokenization/brown_plain_test.txt > corpora/tokenization/brown_tokenized.txt

with open(CORPORA_PATH + "brown_tokenized.txt") as f:
    brown_test_tokenized = f.read().split()

brown_test_tokenized_types = Counter(brown_test_tokenized)

# +
import math

def calculate_entropy(corpus: list[str]) -> float:
    words_counts = Counter(corpus)
    total_words = len(corpus)
    probabilities = {word: count / total_words for word, count in words_counts.items()}
    entropy = -sum(p * math.log2(p) for p in probabilities.values())
    return entropy


# -

calculate_entropy(brown_test_tokenized)

calculate_entropy(brown_test_words)

calculate_entropy(axolotl_test_tokenized)

calculate_entropy(axolotl_test_words)

# # Preguntas:
#  ## ¿Aumentó o disminuyó la entropía en los corpus?
#   ### disminuyó en ambos de 10 a 7 en el caso de brown y de 11 a 7 en el caso de axolotl
#  ## ¿Qué significa que aumente o disminuya la entropía?
#   ### Una mayor entropía nos indica una menor predecibilidad debido a una cantidad mayor de tipos con relación al número de tokens y esto a su vez puede ser por el tipo de corpus que tenemos o la morfología de la lengua ya que en lenguas con morfología compelja, varias palabras pueden relacionarse con una misma base pero con formas diferentes lo que disminuye su frecuencia individual. A su vez si queremos entrenar un modelo estadístico, esto hace que los datos de cada tipo sea más representativo y pueda generalizar mejor.
#  ## ¿Cómo influye la tokenización en la entropía de un texto?
#   ### Como está basado en reglas estadísticas la tokenización tiene nos permite contar patrones que el texto contiene y usarlos para incrementar la frecuencia de unidades con significado como se haría con un sistema basado en reglas pero con la ventaja de que las reglas estadísticas pueden mejorar los resultados o que no es necesario un análisis tan intensivo o extensivo de la lengua. Así disminuimos la cantidad de tipos y aumentamos la predecibilidad. 

# # Extra:

nahuatl_normalizer = elotl.nahuatl.orthography.Normalizer("sep")

axolotl_norm_train_words = [nahuatl_normalizer.normalize(word) for word in axolotl_train_words]

axolotl_norm_test_words = [nahuatl_normalizer.normalize(word) for word in axolotl_test_words]

write_plain_text_corpus(" ".join(axolotl_norm_test_words), CORPORA_PATH + "axolotl_norm_plain_test")

write_plain_text_corpus(" ".join(axolotl_norm_test_words), CORPORA_PATH + "axolotl_norm_plain")

# !subword-nmt learn-bpe -s 100 < corpora/tokenization/axolotl_norm_plain.txt > models/tokenization/axolotl_norm.model

# !subword-nmt apply-bpe -c models/tokenization/axolotl_norm.model < corpora/tokenization/axolotl_norm_plain_test.txt > corpora/tokenization/axolotl_norm_tokenized.txt

axolotl_norm_test_tokenized_types = Counter(axolotl_test_tokenized)

axolotl_norm_test_types = Counter(axolotl_norm_test_words)

with open(CORPORA_PATH + "axolotl_tokenized.txt") as f:
    axolotl_norm_test_tokenized = f.read().split()

print("Axolotl Information")
print("Tokens:", len(axolotl_norm_test_words))
print("Types (word-base):", len(axolotl_norm_test_types))
print("Types (native BPE):", len(axolotl_norm_test_tokenized_types))
print("TTR (word-base):", len(axolotl_norm_test_types)/len(axolotl_norm_test_words))
print("TTR (BPE):", len(axolotl_norm_test_tokenized_types)/len(axolotl_norm_test_tokenized))

print("Axolotl Information")
print("Tokens:", len(axolotl_test_words))
print("Types (word-base):", len(axolotl_test_types))
print("Types (native BPE):", len(axolotl_test_tokenized_types))
print("TTR (word-base):", len(axolotl_test_types)/len(axolotl_test_words))
print("TTR (BPE):", len(axolotl_test_tokenized_types)/len(axolotl_test_tokenized))

calculate_entropy(axolotl_norm_test_words)

calculate_entropy(axolotl_norm_test_tokenized)



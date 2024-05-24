# ---
# Ejercicio 1
# ---

from collections import Counter
import re
import math

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)

def plain_text_to_list(file_name:str) -> list:
    data = []
    with open(f"{file_name}.txt", "r") as file:
        data = file.read().split(" ")
    return(data)

def calculate_entropy(corpus: list[str]) -> float:
    words_counts = Counter(corpus)
    total_words = len(corpus)
    probabilities = {word: count / total_words for word, count in words_counts.items()}
    entropy = -sum(p * math.log2(p) for p in probabilities.values())
    return entropy

# Brown

import nltk
from nltk.corpus import brown
nltk.download('brown')

brown_words = [word for word in brown.words() if re.match("\w", word)]
write_plain_text_corpus(" ".join(brown_words), "./practices/pianissimo-pp/practica4/brown_plain")

# subword-nmt learn-bpe -s 500 < ./brown_plain.txt > tokenization_models/brown.model
# subword-nmt apply-bpe -c tokenization_models/brown.model < ./brown_plain.txt > ./brown_tokenized.txt

brown_words_t = plain_text_to_list("./practices/pianissimo-pp/practica4/brown_tokenized")

# Axolotl

import elotl.corpus
axolotl = elotl.corpus.load("axolotl")

axolotl_words = [word for row in axolotl for word in row[1].lower().split()]
write_plain_text_corpus(" ".join(axolotl_words), "./practices/pianissimo-pp/practica4/axolotl_plain")

# subword-nmt learn-bpe -s 500 < ./axolotl_plain.txt > tokenization_models/axolotl.model
# subword-nmt apply-bpe -c tokenization_models/axolotl.model < ./axolotl_plain.txt > ./axolotl_tokenized.txt

axolotl_words_t = plain_text_to_list("./practices/pianissimo-pp/practica4/axolotl_tokenized")


from tabulate import tabulate
table = [["Word-Level",
        calculate_entropy(brown_words),
        calculate_entropy(axolotl_words)],
        ["BPE",
        calculate_entropy(brown_words_t),
        calculate_entropy(axolotl_words_t)]]

# ---
# Ejercicio 2
# ---

print("\nComparativa (Entropía)\n")
print(tabulate(table, headers=["Tokenizado", "Brown", "Axolotl"]))

# ---
# Ejercicio 3
# ---

# La entropía disminuyó en ambos casos
# A una entropía alta corresponde una mayor cantidad de información por unidad
# Luego del proceso de tokenización, los tokens aumentan, lo cual implica una mayor variabilidad en los datos y por lo tanto la entropía aumenta

# ---
# Ejercicio Extra
# ---

import elotl.nahuatl.orthography
nahuatl_normalizer = elotl.nahuatl.orthography.Normalizer("inali")
axolotl_words_n = [nahuatl_normalizer.normalize(word) for word in axolotl_words]

write_plain_text_corpus(" ".join(axolotl_words_n), "./practices/pianissimo-pp/practica4/axolotl_n_plain")

# subword-nmt learn-bpe -s 500 < ./axolotl_n_plain.txt > tokenization_models/axolotl_n.model
# subword-nmt apply-bpe -c tokenization_models/axolotl_n.model < ./axolotl_n_plain.txt > ./axolotl_n_tokenized.txt

axolotl_words_n_t = plain_text_to_list("./practices/pianissimo-pp/practica4/axolotl_n_tokenized")

tipos_wl = len(Counter(axolotl_words_t))
tipos_n = len(Counter(axolotl_words_n_t))
tokens_wl = len(axolotl_words_t)
tokens_n = len(axolotl_words_n_t)

table = [["Entropía",
        calculate_entropy(axolotl_words_t),
        calculate_entropy(axolotl_words_n_t)],
        ["Tipos",
        tipos_wl,
        tipos_n],
        ["Tokens",
        tokens_wl,
        tokens_n],
        ["TTR",
        tipos_wl/tokens_wl,
        tipos_n/tokens_n]]

print("\nComparativa (Extra)\n")
print(tabulate(table, headers=["Categoría", "Axolotl", "Axolotl Normalizado (INALI)"], numalign="center"))
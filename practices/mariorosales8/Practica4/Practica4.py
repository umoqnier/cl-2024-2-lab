import re
import nltk
from collections import Counter
import elotl.corpus
import math

from nltk.corpus import brown
nltk.download('brown')

# Corpus tokenizados a nivel palabra
brown_words = [word for word in brown.words() if re.match("\w", word)]
axolotl = elotl.corpus.load("axolotl")
axolotl_words = [word for row in axolotl for word in row[1].lower().split()]

# Corpus tokenizados con BPE
def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)
#write_plain_text_corpus(" ".join(brown_words), "brown_plain")
#write_plain_text_corpus(" ".join(axolotl_words), "axolotl_plain")
        
def read_plain_text_corpus(file_name: str) -> str:
    with open(file_name, "r") as f:
        return f.read()
brown_bpe = read_plain_text_corpus("brown.bpe").split()
axolotl_bpe = read_plain_text_corpus("axolotl.bpe").split()

# Calcular entropía
def calculate_entropy(corpus: list[str]) -> float:
    words_counts = Counter(corpus)
    total_words = len(corpus)
    probabilities = {word: count / total_words for word, count in words_counts.items()}
    entropy = -sum(p * math.log2(p) for p in probabilities.values())
    return entropy

print()

print("Entropía del corpus brown:")
print("Word level:", calculate_entropy(brown_words))
print("BPE level:", calculate_entropy(brown_bpe))

print()

print("Entropía del corpus axolotl:")
print("Word level:", calculate_entropy(axolotl_words))
print("BPE level:", calculate_entropy(axolotl_bpe))
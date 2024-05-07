# %% [markdown]
# ### Práctica 4. Tokenization

# %%
"""
Install the necessary packages
"""

!pip install elotl subword-nmt nltk

# %%
"""
Make the necessary imports
"""

import nltk
import re
import math
import elotl.corpus

from nltk.corpus import brown
from collections import Counter

nltk.download('brown')

axolotl_corpus = elotl.corpus.load("axolotl")
brown_complete = [word for word in brown.words() if re.match("\w", word)]

# %%
def calculate_entropy(corpus: list[str]) -> float:
    # Counter object to hold the count of each word in the corpus
    word_counts = Counter(corpus)

    # Total number of words in the corpus
    total_word_count = len(corpus)

    # Dictionary to hold the probability of each word
    # The probability is calculated as the count of the word divided by the total number of words
    word_probabilities = {word: count / total_word_count for word, count in word_counts.items()}

    # Calculate the entropy of the corpus
    # The entropy is the sum of the product of the probability of each word and the log base 2 of the probability
    entropy = -sum(probability * math.log2(probability) for probability in word_probabilities.values())

    return entropy

# %%
"""
Entropía con tokenización word-level
"""

brown_corpus = brown_complete[:100000]
h_brown_wl = calculate_entropy(brown_corpus)
print("Calculating entropy for the Brown corpus with word-level tokenization...")
print("Entropy of the Brown corpus with word-level tokenization is: ", h_brown_wl)

axolotl_words = [word for row in axolotl_corpus for word in row[1].lower().split()]
h_axolotl_wl = calculate_entropy(axolotl_words)
print("Calculating entropy for the Axolotl corpus with word-level tokenization...")
print("Entropy of the Axolotl corpus with word-level tokenization is: ", h_axolotl_wl)

# %%
"""
Entropía con tokenización BPE
"""

DATA_PATH = "./"

def write_text_to_file(raw_text: str, file_name: str) -> None:
    """
    This function writes a given text to a file.

    Parameters:
    raw_text (str): The text to be written to the file.
    file_name (str): The name of the file.

    Returns:
    None
    """
    # Open the file in write mode. If the file does not exist, it will be created.
    with open(f"{file_name}.txt", "w", encoding='utf-8') as file:
        # Write the raw text to the file
        file.write(raw_text)

# Calculate the number of rows to use for training from the Axolotl corpus
axolotl_corpus_train_rows_count = len(axolotl_words) - round(len(axolotl_words)*.30)

# Split the Axolotl corpus into training and testing sets
axolotl_corpus_train = axolotl_words[:axolotl_corpus_train_rows_count]
axolotl_corpus_test = axolotl_words[axolotl_corpus_train_rows_count:]

# Write the training set to a file
write_text_to_file(" ".join(axolotl_corpus_train), DATA_PATH + "axolotl_plain")
print("Training set written to file.")

# Learn the BPE model from the training set
!subword-nmt learn-bpe -s 500 < axolotl_plain.txt > axolotl.model
print("BPE model learned from training set.")

# Write the testing set to a file
write_text_to_file(" ".join(axolotl_corpus_test), DATA_PATH + "axolotl_plain_test")
print("Testing set written to file.")

# Apply the BPE model to the testing set
!subword-nmt apply-bpe -c axolotl.model < axolotl_plain_test.txt > axolotl_tokenized.txt
print("BPE model applied to testing set.")


# %%
# Open the file containing the tokenized Axolotl corpus
with open(DATA_PATH + "axolotl_tokenized.txt", encoding='utf-8') as file:
    # Read the file and split the text into tokens
    axolotl_corpus_tokens = file.read().split()

# Count the occurrences of each token in the Axolotl corpus
axolotl_corpus_token_counts = Counter(axolotl_corpus_tokens)

# Print the 20 most common tokens in the Axolotl corpus
print("The 20 most common tokens in the Axolotl corpus are:")
for token, count in axolotl_corpus_token_counts.most_common(20):
    print(f"{token}: {count}")

# Calculate the entropy of the Axolotl corpus
axolotl_corpus_entropy = calculate_entropy(axolotl_corpus_tokens)

# Print the entropy of the Axolotl corpus
print("The entropy of the Axolotl corpus with BPE tokenization is: ", axolotl_corpus_entropy)

# Calculate the number of rows to use for training from the Brown corpus
brown_corpus_train_rows_count = len(brown_corpus) - round(len(brown_corpus)*.30)

# Split the Brown corpus into training and testing sets
brown_corpus_train = brown_corpus[:brown_corpus_train_rows_count]
brown_corpus_test = brown_corpus[brown_corpus_train_rows_count:]

# Write the training set to a file
write_text_to_file(" ".join(brown_corpus_train), DATA_PATH + "brown_plain")

# Learn the BPE model from the training set
!subword-nmt learn-bpe -s 500 < brown_plain.txt > brown.model

# Write the testing set to a file
write_text_to_file(" ".join(brown_corpus_test), DATA_PATH + "brown_plain_test")

# Apply the BPE model to the testing set
!subword-nmt apply-bpe -c brown.model < brown_plain_test.txt > brown_tokenized.txt

# %%
# Open the file containing the tokenized Brown corpus
with open(DATA_PATH + "brown_tokenized.txt", encoding='utf-8') as file:
    # Read the file and split the text into tokens
    brown_corpus_tokens = file.read().split()

# Count the occurrences of each token in the Brown corpus
brown_corpus_token_counts = Counter(brown_corpus_tokens)

# Print the 20 most common tokens in the Brown corpus
print("The 20 most common tokens in the Brown corpus are:")
for token, count in brown_corpus_token_counts.most_common(20):
    print(f"{token}: {count}")

# Calculate the entropy of the Brown corpus
brown_corpus_entropy = calculate_entropy(brown_corpus_tokens)

# Print the entropy of the Brown corpus
print("The entropy of the Brown corpus with BPE tokenization is: ", brown_corpus_entropy)

# %%
print("Entropy values for different corpora and tokenization methods:\n")

print("Corpus Brown:")
print("With word-level tokenization:")
print(h_brown_wl)

print("\nWith Byte Pair Encoding (BPE):")
print(brown_corpus_entropy)
print()

print("Corpus Axolotl:")
print("With word-level tokenization:")
print(h_axolotl_wl)

print("\nWith Byte Pair Encoding (BPE):")
print(axolotl_corpus_entropy)

# %% [markdown]
# ### Preguntas
# 
# **¿Aumentó o disminuyó la entropía para los corpus?**  
# En los dos casos, i.e. en ambos corpus, observamos una disminución significativa en la entropía al aplicar la tokenización con BPE (Byte Pair Encoding), en comparación con la tokenización a nivel de palabra. Esta reducción fue particularmente notable en el corpus de axolotl, donde la entropía disminuyó de 11.84 a 8.35. Esto indica que el proceso de tokenización con BPE consigue simplificar la estructura del texto al reducir la variedad de tokens necesarios para representarlo.
# 
# **¿Qué significa que la entropía aumente o disminuya en un texto?**  
# La entropía en un texto se refiere a la medida de incertidumbre o impredecibilidad asociada con el lenguaje utilizado. Un aumento en la entropía indica que el texto tiene un vocabulario más amplio y estructuras más complejas, lo que aumenta la impredecibilidad. Esto puede ser deseable desde un punto de vista literario o lingüístico, ya que refleja riqueza y diversidad en el uso del lenguaje. Sin embargo, para la computación y el procesamiento de lenguaje natural, un alto nivel de entropía puede representar un desafío, ya que la variedad y la complejidad del lenguaje complican la interpretación y el análisis automáticos del texto.
# 
# **¿Cómo influye la tokenización en la entropía de un texto?**  
# La tokenización es un proceso crucial en el análisis de texto que consiste en dividir el texto en unidades más pequeñas, conocidas como tokens. Este proceso puede influir considerablemente en la entropía de un texto. Al aplicar métodos de tokenización como el BPE, se simplifica el vocabulario del texto y se estandarizan las formas de las palabras, lo que generalmente resulta en una reducción de la entropía. Esta disminución facilita la tarea de procesamiento de textos, ya que un menor nivel de entropía implica menos impredecibilidad y una estructura más uniforme, lo que es beneficioso para algoritmos de procesamiento de lenguaje natural y otras aplicaciones informáticas.



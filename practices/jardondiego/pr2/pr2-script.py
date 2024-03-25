# %% [markdown]
# ## Práctica 2
# 
# ### Objetivo
# 
# Implementar modelo etiquetador POS para el idioma otomí.  
# A partir de una oración en otomí, obtener una secuencia de etiquetas.
# Utilizar un modelo CRF.
# 
# - Definir feature functions
# - Entrenar modelo

# %%
"""
configurar dependencias
"""

%pip install scikit-learn
%pip install sklearn_crfsuite

# %%
import json

"""
preprocesar conjunto de datos
partir los datos en dos conjuntos
uno para entrenar y otro para probar
"""


def parse_raw_line(raw_string: str):
    """
    parse a line of the file
    """
    sample_as_list = json.loads(raw_string)
    parsed_elements = []

    for i in range(len(sample_as_list)):
        element = sample_as_list[i]
        *components, tag = element
        word = "".join([ component[0] for component in components ])
        parsed_elements.append((word, tag))

    return parsed_elements

def parse_list(phrase):
    """
    parse a pharase as a list of elements
    [ *components, tag ]

    where components is a list of [ word, tag ]
    and tag is a string
    """
    parsed_elements = []

    for i in range(len(phrase)):
        element = phrase[i]
        *components, tag = element
        word = "".join([ component[0] for component in components ])
        parsed_elements.append((word, tag))

    return parsed_elements

def parse_file_into_dataset():
    """
    parse a file into a dataset
    of the form

    [ element ]
    where element is a list of [ word, tag ]
    """
    FILENAME = "corpus_otomi.txt"
    file = open(FILENAME, "r")
    data = file.read()
    file.close()

    lines = data.split("\n")
    lines = [line.strip() for line in lines]

    # parsed_data = [parse_raw_line(line) for line in lines]
    parsed_data = []
    for line in lines:
        error_count = 0
        try:
            parsed_data.append(parse_raw_line(line))
        except:
            error_count += 1

    print("had {} errors".format(error_count))

    return parsed_data


def detect_encoding_issues(s):
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        # The string contains non-ASCII characters, which could be fine.
        # Further checks for common misencoding patterns could go here.
        if "Ã" in s or "�" in s:
            return True  # Found common misencoding indicators.
        # Check for high ordinal values that might indicate misencoding.
        if any(ord(c) > 127 for c in s):
            return True  # Found characters outside the standard ASCII range.
    return False

def phrase_as_single_string(phrase):
    """
    convert a phrase into a single string
    """
    return " ".join([word for word, tag in phrase])


def get_mock_dataset():
    """
    generate a simple dataset for initial testing
    """

    sample1 = [
        [["bi", "3.cpl"], ["'u̱n", "stem"], ["gí", "1.obj"], "v"],
        [["yi̱", "det.pl"], "det"],
        [["mbu̱hí", "stem"], "obl"],
        [["nge", "stem"], "cnj"],
        [["hín", "stem"], "neg"],
        [["dí", "1.icp"], ["má", "ctrf"], ["né", "stem"], "v"],
        [["gwa", "1.icp.irr"], ["porá", "stem"], "v"],
        [["nge", "stem"], "cnj"],
        [["dí", "1.icp"], ["má", "ctrf"], ["dáhní", "stem"], "v"],
    ]

    sample2 = [
        [["bo", "3.cpl"], ["pihkí", "stem"], "v"],
        [["yi̱", "det.pl"], "det"],
        [["k'iñá", "stem"], "obl"],
    ]

    samples = [sample1, sample2]
    return [parse_list(sample) for sample in samples]

mock_dataset = get_mock_dataset()
dataset = parse_file_into_dataset()

# filter out non-ascii sequences
dataset = [sample for sample in dataset if all(ord(c) < 128 for c in sample[0][0])]
dataset = [ sample for sample in dataset if not detect_encoding_issues(phrase_as_single_string(sample)) ]

print(dataset[0])
print(len(dataset))

# %%
"""
construir conjunto de entrenamiento y prueba
para ello vamos a usar la función train_test_split
de scikit-learn que nos permite dividir un conjunto
de datos en dos conjuntos, uno para entrenar y otro para pruebas
"""

from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.2)
train

# %%
def get_word_features(sentence, word, position):
    """
    construye el conjunto de funciones de características
    """
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],  # Suffix
        "word[-2:]": word[-2:],  # Suffix
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }

    if position > 0:
        word1 = sentence[position - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if position < len(sentence)-1:
        word1 = sentence[position+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def get_phrase_features(phrase):
    """
    obtiene las características de una muestra del conjunto de datos
    """
    return [get_word_features(phrase, word, i) for i, (word, tag) in enumerate(phrase)]

def get_labels(phrase):
    """
    obtiene las etiquetas de una muestra del conjunto de datos
    """
    return [tag for _, tag in phrase]

def get_tokens(phrase):
    """
    obtiene las palabras de una muestra del conjunto de datos
    """
    return [word for word, _ in phrase]

# %%
"""
entrenar el modelo
utilizando linear-chain CRF
"""

# preparar dataset de entrenamiento

X_train = [get_phrase_features(phrase) for phrase in train]
y_train = [get_labels(phrase) for phrase in train]

# preparar dataset de pruebas
X_test = [get_phrase_features(phrase) for phrase in test]
y_test = [get_labels(phrase) for phrase in test]

import sklearn_crfsuite

# crear el modelo
model = sklearn_crfsuite.CRF()

# entrenar el modelo
model.fit(X_train, y_train)

# hacer predicciones a partir del conjunto de pruebas
y_pred = model.predict(X_test)

# %%
"""
evaluación del modelo
utilizar una matriz de confusión
apoyado en la función classification_report de scikit-learn
"""



# %% [markdown]
# # Referencia
# 
# To implement a simple POS tagger using linear-chain Conditional Random Fields (CRFs) with `sklearn-crfsuite` (an extension for scikit-learn designed specifically for CRF), you will need to follow these steps. Note that `sklearn-crfsuite` is a convenient wrapper around the `python-crfsuite` library, which is specifically designed for CRF models and is well-suited for tasks like POS tagging.
# 
# First, make sure you have `sklearn-crfsuite` installed. If not, you can install it using pip:
# 
# ```bash
# pip install sklearn-crfsuite
# ```
# 
# Let's assume you have a dataset for Spanish POS tagging. The dataset should be a list of sentences, where each sentence is a list of `(word, tag)` tuples. For the sake of an example, let's define a very small dataset:
# 
# ```python
# sentences = [
#     [("Todos", "DET"), ("los", "DET"), ("hombres", "NOUN"), ("deben", "VERB"), ("morir", "VERB"), (",", "PUNCT"), ("Jon", "PROPN"), ("Nieve", "PROPN"), (".", "PUNCT")],
#     [("¿Quién", "PRON"), ("es", "VERB"), ("John", "PROPN"), ("Galt", "PROPN"), ("?", "PUNCT")]
# ]
# ```
# 
# Next, define feature extraction functions. Feature extraction is crucial for CRFs, as it determines the information the model can use to make predictions:
# 
# ```python
# def word2features(sent, i):
#     """Extract features for a given word in a sentence."""
#     word = sent[i][0]
#     features = {
#         'bias': 1.0,
#         'word.lower()': word.lower(),
#         'word[-3:]': word[-3:],  # Suffix
#         'word[-2:]': word[-2:],  # Suffix
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.isdigit()': word.isdigit(),
#     }
#     if i > 0:
#         word1 = sent[i-1][0]
#         features.update({
#             '-1:word.lower()': word1.lower(),
#             '-1:word.istitle()': word1.istitle(),
#             '-1:word.isupper()': word1.isupper(),
#         })
#     else:
#         features['BOS'] = True  # Beginning of Sentence
# 
#     if i < len(sent)-1:
#         word1 = sent[i+1][0]
#         features.update({
#             '+1:word.lower()': word1.lower(),
#             '+1:word.istitle()': word1.istitle(),
#             '+1:word.isupper()': word1.isupper(),
#         })
#     else:
#         features['EOS'] = True  # End of Sentence
#     
#     return features
# 
# def sent2features(sent):
#     """Extract features for all words in a sentence."""
#     return [word2features(sent, i) for i in range(len(sent))]
# 
# def sent2labels(sent):
#     """Extract labels for all words in a sentence."""
#     return [label for token, label in sent]
# 
# def sent2tokens(sent):
#     """Extract tokens for all words in a sentence."""
#     return [token for token, label in sent]
# ```
# 
# Now, prepare the dataset for training:
# 
# ```python
# X_train = [sent2features(s) for s in sentences]
# y_train = [sent2labels(s) for s in sentences]
# ```
# 
# Train the CRF model:
# 
# ```python
# import sklearn_crfsuite
# 
# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     c1=0.1,
#     c2=0.1,
#     max_iterations=100,
#     all_possible_transitions=True
# )
# crf.fit(X_train, y_train)
# ```
# 
# After training, you can use the trained model to predict POS tags for new sentences:
# 
# ```python
# test_sentence = [("Este", "DET"), ("es", "VERB"), ("un", "DET"), ("ejemplo", "NOUN")]
# X_test = [sent2features(test_sentence)]
# y_pred = crf.predict(X_test)
# print("Predicted:", y_pred)
# ```
# 
# This simple example demonstrates the basic process of using `sklearn-crfsuite` for POS tagging with a CRF model. For real-world applications, you would need a much larger dataset and more sophisticated feature engineering to achieve high accuracy.



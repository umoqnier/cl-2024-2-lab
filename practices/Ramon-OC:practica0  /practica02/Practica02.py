# -*- coding: utf-8 -*-
"""2_language_levels_II.ipynb

Original file is located at
    https://colab.research.google.com/drive/1KxWQt-CkhpgBsRl8V1bnXPlDT4ZsBtRR
"""
"""# Implement a POS tagger for the Otomi language"""

import requests
from unidecode import unidecode
import json
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def get_corpus() -> list:
    url = "https://raw.githubusercontent.com/Ramon-OC/cl-2024-2-lab/practica02/practices/Ramon-OC%3Apractica0%20%20/practica02/corpus_otomi"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split('\n')
        corpus = []
        for line in lines:
            if line.strip(): # Removes any leading, and trailing whitespaces
                corpus.append(json.loads(line))
        return corpus
    else:
        print("Corpus request not completed ERROR:", response.status_code)
        return []

def get_morphemes(data: list):
    num_chunks = len(data) - 1
    word = "".join(unidecode(chunk[0]) for chunk in data[:num_chunks])
    gloss_vector = [unidecode(chunk[1]) for chunk in data[:num_chunks]]
    tag = unidecode(data[num_chunks])
    return (word, tag, num_chunks, gloss_vector)

def word_to_features(sent: list, i: int):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'prefix_1': word[:1],
        'prefix_2': word[:2],
        'suffix_1': word[-1:],
        'word_len': len(word)
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    return features

# Extract features and labels
def sentence_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

def train_crf(corpus: list) -> tuple:
    sentences = []
    for sentence in corpus:
        joined = list(map(get_morphemes, sentence))
        sentences.append(joined)

    # Prepare data for CRF
    x = [[word_to_features(sent, i) for i in range(len(sent))] for sent in sentences]
    y = [[tag for _, tag, _, _ in s] for s in sentences]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

    # Training model
    # Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=True)

    try:
        crf.fit(x_train, y_train)
    except AttributeError as e:
        print(e)

    return crf, x_test, y_test

otomi_corpus = get_corpus()
crf, x_test, y_test = train_crf(otomi_corpus)
y_pred = crf.predict(x_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

"""# Report accuracy, precision, recall and F1-score"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# It is responsible for defining how accurate the model is
print("Accuracy: ", accuracy_score(y_pred_flat, y_test_flat))
# Relationship between correct positive predictions with the total predictions of the class regardless of whether they were correct or not
print("Precision: ", precision_score(y_pred_flat, y_test_flat, average = "macro", zero_division=1))
# Relationship between correct positive predictions with the total of incorrect predictions of other classes
print("Recall: ", recall_score(y_pred_flat, y_test_flat, average="macro", zero_division=1))
# Weighted average between precision and recall
print("F1: ", f1_score(y_pred_flat, y_test_flat, average="macro"))

"""#Tagged sentence example"""

import pandas as pd

sentence_example = [ feature["word.lower()"] for feature in x_test[10] ]
original =  y_test[10]
prediction = crf.predict_single(x_test[10])
data = {"Sentence": sentence_example, "Original": original, "Prediction": prediction}
table = pd.DataFrame(data)

print(table)
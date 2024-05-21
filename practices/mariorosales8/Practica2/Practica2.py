import nltk
from nltk.corpus import cess_esp
import requests
from sklearn.model_selection import train_test_split
from nltk.tag import hmm
from collections import defaultdict
from inspect import Attribute
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def parse_list(str):
    lista = []
    str = str.strip()[1:-1]
    corchetes = 0
    item = ""
    for s in str:
        if s == "[":
            corchetes += 1
        elif s == "]":
            corchetes -= 1
        if s == "," and corchetes == 0:
            lista.append(item)
            item = ""
        elif s!="\"" and s!=" ":
            item += s
    lista.append(item)
    return lista


def parse_corpus(corpus):
    parsed = []
    for word in corpus:
        palabra = ""
        lista = parse_list(word)
        for i in range(len(lista) - 1):
            palabra += parse_list(lista[i])[0]
        parsed.append((palabra, lista[-1]))
    return parsed

corpora = []

with open('corpus_otomi', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        corpus = parse_list(line)
        corpora.append(parse_corpus(corpus))

train_data, test_data = train_test_split(corpora, test_size=0.3, random_state=42)

# Creando el modelo HMM usando nltk
trainer = hmm.HiddenMarkovModelTrainer()

# Hora de entrenar
hmm_model = trainer.train(train_data)

tagged_test_data = hmm_model.tag_sents([[word for word, _ in sent] for sent in test_data])

# Extrayendo tags verdaderas vs tags predichas
y_true = [tag for sent in test_data for _, tag in sent]
y_pred = [tag for sent in tagged_test_data for _, tag in sent]

def report_accuracy(y_true: list, y_pred: list) -> defaultdict:

    label_accuracy_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for gold_tag, predicted_tag in zip(y_true, y_pred):
        label_accuracy_counts[gold_tag]["total"] += 1
        if gold_tag == predicted_tag:
            label_accuracy_counts[gold_tag]["correct"] += 1
    return label_accuracy_counts

label_accuracy_counts = report_accuracy(y_true, y_pred)

nltk.download('punkt')

def word_to_features(sent: list, i: int):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        #'word.istitle()': word.istitle(),
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
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

# Convierte las palabras que no son ascii
ascii_corpora = []
for sent in corpora:
    ascii_sent = []
    for word, pos in sent:
        if word.isascii():
            ascii_sent.append((word, pos))
        else:
            ascii_sent.append((word.encode('ascii', 'ignore').decode('utf-8'), pos.encode('ascii', 'ignore').decode('utf-8')))
    if len(ascii_sent) > 0:
        ascii_corpora.append(ascii_sent)

# Prepare data for CRF
X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in ascii_corpora]
y = [[pos for _, pos in sent] for sent in ascii_corpora]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=True)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)

y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

print(f"Accuracy: {accuracy_score(y_pred_flat, y_test_flat)}\n")

print(f"Precision: {precision_score(y_pred_flat, y_test_flat, average='macro', zero_division=0)}\n")

print(f"Recall: {recall_score(y_pred_flat, y_test_flat, average='macro', zero_division=0)}\n")

print(f"F1-Score: {f1_score(y_pred_flat, y_test_flat, average='macro', zero_division=0)}\n")

prueba = ""
for word in X_test[120]:
    prueba += word['word.lower()'] + " "

# Tokenizando
tokenized_sentence = nltk.word_tokenize(prueba)

# Haciendo predicciones
predicted_tags = [tag for word, tag in hmm_model.tag(tokenized_sentence)]
# Tags verdaderas
actual_tags = [tag for word, tag in zip(tokenized_sentence, y_test[120])]

print()
print("\nOración de ejemplo:", prueba)
print()
for word, pred, actual in zip(tokenized_sentence, predicted_tags, actual_tags):
    print(f"Palabra: {word}.\nTag predicha: \t{pred}.\nTag real: \t{actual}\n")

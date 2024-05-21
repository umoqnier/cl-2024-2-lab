import warnings
warnings.filterwarnings("ignore")

# ---
# Ejercicio 1
# ---

import json
from unidecode import unidecode

# Leemos el archivo corpus_otomi

data = []
with open("./practices/pianissimo-pp/practica2/corpus_otomi", "r") as file:
    for line in file:
        data.append(json.loads(line))

# Damos el formato requerido a los datos

tags = ["aff", "cnj", "cnj.adv", "cond", "conj", "conj.adv", "cord", "dec", "dem", "det", "dim", "gen", "it", "loc", "lim", "n", "neg", "obl", "p.loc", "prt", "regular/v", "unkwn", "v"]

corpora = []
for list in data:
    sentence = []
    vs = True
    for item in list:
        word = ""
        map_tag = ""
        for subitem in item:
            if isinstance(subitem,str):
                map_tag = unidecode(subitem)
            else:
                word += unidecode(subitem[0])
        if map_tag in tags:
            sentence.append((word,map_tag))
        else:
            vs = False
    if vs:
        corpora.append(sentence)

# Definimos la función característica

def word_to_features(sent: list, i: int):
    word = sent[i][0]
    features = {
        'ap': '\'' in word,
        'prefix_0': word[0],
        'prefix_1': word[:3],
        'prefix_2': word[:4],
        'suffix_0': word[-1],
        'suffix_1': word[-2:],
        'suffix_2': word[-3:],
        'nwp': i/len(sent),
        'word_len': len(word)
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features['prev_word'] = prev_word
    else:
        features['BOS'] = True  # Beginning of sentence
    if i < len(sent)-1:
        next_word = sent[i + 1][0]
        features['next_word'] = next_word
    else:
        features['EOS'] = True  # Ending of sentence
    return features

def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in corpora]
y = [[pos for _, pos in sent] for sent in corpora]

from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split

# Separamos los datos de entrenamiento y los de prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# Entrenamos al modelo

from inspect import Attribute
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=True)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)

# ---
# Ejercicio 2
# ---

y_pred = crf.predict(X_test)

# Extraemos los tags verdaderos y los tags predichos

y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate

print("Desempeño:\n")
row1 = [["Accuracy",accuracy_score(y_pred_flat, y_test_flat)],
        ["Precision",precision_score(y_pred_flat, y_test_flat, average="macro")],
        ["Recall",recall_score(y_pred_flat, y_test_flat, average="macro")],
        ["F1 Score:",f1_score(y_pred_flat, y_test_flat, average="macro")]]
print(tabulate(row1,headers=["Metrics","CRF"],tablefmt="presto"))
print("\n")

# ---
# Ejercicio 3
# ---

import random
ei = random.randint(0, len(y_pred)-1)
es = corpora[X.index(X_test[ei])]
et = y_pred[ei]
row = []
i=0
for word,tag in es:
    row.append([word,tag,et[i]])
    i+=1
print("Ejemplo de oración etiquetada:\n")
print(tabulate(row, headers=["Palabra", "Etiqueta Real", "Etiqueta Predicha"],tablefmt="presto"))
print("\n")

# ---
# Ejercicio Extra 1
# ---

from nltk.tag import hmm

# Separamos los datos de entrenamiento y los de prueba

train_data, test_data = train_test_split(corpora, test_size=0.3, random_state=62)

# Entrenamos al modelo

trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train(train_data)
tagged_test_data = hmm_model.tag_sents([[word for word, _ in sent] for sent in test_data])

# ---
# Ejercicio Extra 2
# ---

# Extraemos los tags verdaderos y los tags predichos

y_test_flat = [tag for sent in test_data for _, tag in sent]
y_pred_flat = [tag for sent in tagged_test_data for _, tag in sent]

row2 = [accuracy_score(y_pred_flat, y_test_flat),
        precision_score(y_pred_flat, y_test_flat, average="macro"),
        recall_score(y_pred_flat, y_test_flat, average="macro"),
        f1_score(y_pred_flat, y_test_flat, average="macro")]
i=0
for item in row1:
    item.append(row2[i])
    item.append(abs(item[1]-item[2]))
    i+=1
print("Comparativa de desempeño: \n")
print(tabulate(row1,headers=["Metrics","CRF","HMM","Difference"],tablefmt="presto"))
print("\n")

# ---
# Ejercicio Extra 3
# ---

# La principal diferencia es la cantidad de datos disponible, misma que se ve reflejada en los resultados, pues a comparación del español, donde los datos son abundantes, las métricas difícilmente se acercan a 1, así como el desconocimiento de la lengua para definir feature functions adecuadas en el caso de los CRF
# En general, como los CRF generalizan a los HMM, su desempeño es mejor, pues nos dan la posibilidad de caracterizar mejor las dependencias entre palabras
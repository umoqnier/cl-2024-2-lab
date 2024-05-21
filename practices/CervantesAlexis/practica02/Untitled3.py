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

with open('./corpus_otomi', 'r') as file: #Importamos el corpus
    texto = file.read()

#Importamos las paqueterías necesarias.
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from inspect import Attribute
from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report

#Preparamos el corpus para usarlos
texto_para_lista = "lista = [" + texto.strip().replace("\n", ",") + "]"

exec(texto_para_lista)

#Damos formato a los datos de entrenamiento y definimos las features.
X = []
c = 0
for sent in lista:
    l = []
    for word in sent:
        dic = {}
        palabra = ''
        for glosa in word:
            if type(glosa) == list:
                palabra += glosa[0]
        dic.update({
            'word' : palabra,
            'suffix3' : palabra[:-3],
            'suffix2' : palabra[:-2],
            'suffix1' : palabra[:-1],
            'mid' : palabra[2:-2],
            'word_len' : len(word),
            'prefix1' : palabra[:1],
            'prefix2' : palabra[:2],
        })
        l.append(dic)
    X.append(l)
for i in X:
    for j in range(len(i)):
        if j > 0:
            prev_word = i[j - 1]['word']
            i[j].update({
                'prev_word': prev_word,
            })
        if j < len(i) - 1:
            next_word = i[j + 1]['word']
            i[j].update({
                'next_word': next_word,
            })

y = []
for sent in lista:
    sent_labels = []
    for word in sent:
        sent_labels.append(word[-1].encode('utf-8')) #Evitamos el error 'ascii' codec can't encode character
    y.append(sent_labels)

#Hacemos el split de los datos para entrenar y evaluar y mantenemos concordancia en el formato de los datos.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
for sent in y_test:
    for i in range(len(sent)):
        sent[i] = sent[i].decode('utf-8')

#Establecemos los parámetros y entrenamos el modelo.
crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=40, all_possible_transitions=True, verbose=True)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)

# +
#Obtenemos las medidas de evaluación requeridas.
y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat, zero_division = 0)
print(report)
# -

#ejemplo, tomamos una frase larga:
c = 0
for i in range(len(X_test)):
    if len(X_test[i]) > c:
        c = i
s = ''
for i in range(len(X_test[c])):
    if y_pred[c][i] == y_test[c][i]:
        s +=  X_test[c][i]['word'] + "(\033[2;32;3m" + y_pred[c][i] + "\033[0;0m) "
    else:
        s +=  X_test[c][i]['word'] + "(\033[2;31;1m" + y_pred[c][i] + "\033[0;0m) "
print('Oración de ejemplo:\nPredicción:\n' + s)
s = ''
for i in range(len(X_test[c])):
    if y_pred[c][i] == y_test[c][i]:
        s +=  X_test[c][i]['word'] + "(\033[2;32;3m" + y_test[c][i] + "\033[0;0m) "
    else:
        s +=  X_test[c][i]['word'] + "(\033[2;31;1m" + y_test[c][i] + "\033[0;0m) "
print('\nEtiquetas verdaderas:\n' + s)

import nltk
from nltk.tag import DefaultTagger, UnigramTagger
from nltk.corpus import treebank
import re

lvl3 = """[["n","psd"], ["dó","1.cpl"], ["phu̱di","stem"], "v"]"""

#proceso de nivel(profundidad) 3 del corpus
def process3(lvl3):
    lvl3 = lvl3[1:-1]
    item = lvl3.split("],")
    itemf = []
    
    #elementos de enmedio
    for i in item[1:-1]:
        p = i[2:]
        p = p.split(",")
        p2 = []
        for j in p:
            p2.append(j.strip('"'))

        itemf.append(p2)
        
    #primer elemento
    first = item[0][1:].split(",")
    p2 = []
    for i in first:
        p2.append(i.strip('"'))
        
    itemf.insert(0,p2)

    #ultimo elemento
    p2 = item[len(item)-1][1:].strip('"')
    itemf.insert(len(itemf),p2)
    
    return itemf

#proceso de nivel(profundidad) 2 del corpus
def process2(lvl2):
    item = lvl2.split(', [[')

    #elementos de enmedio
    itemf = []
    for i in item[1:-1]:
        i = "[["+i
        itemf.append(process3(i))
        #processar
        
    #elementos del inicio
    itemFirst = item[0][1:]
    itemf.insert(0,process3(itemFirst))

    #ultimo elemento
    itemLast = "[["+item[len(item)-1][:-1]
    itemf.insert(len(itemf),process3(itemLast))

    return itemf


corpus = []
with open('corpus_otomi.txt', 'r') as file:
    for line in file:
        corpus.append(process2(line))


# Función para convertir el formato del corpus al formato esperado por unigram_tagger
def flatten_corpus(corpus):
    flattened_corpus = []
    for sentence in corpus:
        tokens = [word[0] for word in sentence]
        flattened_corpus.append(tokens)
    return flattened_corpus

# Corpus

# Divide el corpus en conjunto de entrenamiento y conjunto de prueba
train_size = int(len(corpus) * 0.8)
train_corpus = corpus[:train_size]
test_corpus = corpus[train_size:]

# Convertir el corpus a un formato plano
flattened_train_corpus = flatten_corpus(train_corpus)
flattened_test_corpus = flatten_corpus(test_corpus)

# Entrenamiento del etiquetador
default_tagger = DefaultTagger('NN')  # Etiqueta predeterminada para palabras desconocidas
unigram_tagger = UnigramTagger(flattened_train_corpus, backoff=default_tagger)


# Oración de ejemplo
sentence = [
    ["má", "hín", "dó", "né", "bu̱"],
    ["ya", "dó", "hín", "má", "gó"],
    ["dó", "tsó", "gó"],
]
for x in sentence:
    etiqueta = unigram_tagger.tag(x)
    print("oración de ejemplo:", etiqueta)





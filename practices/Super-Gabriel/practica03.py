import nltk
from nltk.corpus import stopwords

""" ------------------------ Parte 1 --------------------------- """

#descargando los stopwords de nltk
nltk.download('stopwords')

# guradando solo los primeros 100 elementos
nltk_stopwords = stopwords.words('spanish')[:100]

#importando el corpus CREA
import csv

# Lista para almacenar los datos del CSV
crea_freq_words = []
iterador = 0;

with open('../../notebooks/corpora/zipf/crea_full.csv', newline='', encoding='latin-1') as archivo_csv:
    lector_csv = csv.reader(archivo_csv)
    
    for fila in lector_csv:
        # Agregando la fila como una lista a los datos del CSV
        crea_freq_words.append(fila[0].split('\t')[0])

        # solo los primeros 100 elementos
        iterador += 1
        if iterador >= 100:
            break

# eliminando espacios de las palabras
crea_freq_words = [palabra.replace(" ", "") for palabra in crea_freq_words]

# contando coincidencias en ambas listas
contador = 0
for word in crea_freq_words[1:]:
    if word in nltk_stopwords:
        contador += 1


#imprimiendo número de coincidencias
print(str(contador)+"/100 palabras coinciden del corpus de CREA con los stopwords de nltk")

""" ¿Obtenemos el mismo resultado?

como vemos, las coincidencias de las palabras mas frecuentes del corpus de CREA con las
stopwords son ~60% por lo que es bastante considerable, esto probablemente a que las
palabras más frecuentes tienden a ser determinantes o palabras funcionales y curiosamente
de longitud muy pequeña, con respecto a las stopwords sucede lo mismo, palabras muy pequeñas
o funcionales no suelen aportar mucha información como son las stopwords a diferencia de
palabras mas complejas que se usan en contextos más especificos.

"""


""" ------------------------ Parte 2 --------------------------- """

"""
el lenguaje propuesto será las cadenas de 0s y 1s con máximo longitud 7
nombre propuesto: bin-7
oracion de ejemplo: 01 000 0101 0000000
"""
import random

#función para generar palabras del lenguaje aleatoriamente
def get_word():
    #determinando la longitud de la palabra
    length = random.randint(1,7)
    word = ""
    
    #construyendo la palabra
    for num in range(length):
        word += str(random.randint(0,1))

    return word;


# funcion para generar un corpus artificial en una lista
def get_corpus(r):
    corpus = []
    #construyendo el corpus
    for word in range(r):
        corpus.append(get_word())

    return corpus

        
# generando una lista de 1000 palabras que representarán un corpus artificial
corpus = get_corpus(1000)
# lista de tipos
corpus_types = list(set(corpus))

freq_list = []
for elem in corpus_types:
    freq_list.append(corpus.count(elem))

# lista con la palabra y su frecuencia en el corpus
vector_list = list(zip(corpus_types,freq_list))

# mostrar corpus artificial
# print(corpus)

# mostrar lista de tipos
# print(corpus_types)

# lista ordenada de mayor a menor de acuerdo a su frecuencia
final_list = sorted(vector_list, key=lambda x: x[1], reverse=True)

print(final_list)

"""como vemos se vé un decremento al inicio mayor con respecto a las menos frecuentes, por lo que
si cumple con zipf, esto tal vez debido a que es más probable encontrarse cadenas de 0s y 1s más
pequeñas que las grandes."""

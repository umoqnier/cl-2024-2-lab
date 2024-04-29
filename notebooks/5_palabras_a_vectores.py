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

# + [markdown] id="BBx7ZEhlg7ZF"
# # Práctica 5: Palabras a Vectores

# + [markdown] id="zrSf6OaCdBTG"
# <img src="http://ruder.io/content/images/2016/04/word_embeddings_colah.png" width="300" heitgh="300">

# + [markdown] id="ZoMC0KktNlBd"
# ## Objetivo

# + [markdown] id="sTM0AzHGNmu3"
# - Explorar representaciones vectoriales de palabras
#     - Basados en Frecuencias
#     - Usando TF-IDF
#     - Word2Vec
# - Medir la similitud entre dos vectores
# - Manipulación de representaciones vectoriales previamente entrenadas
# - Entrenar representaciones usando `gensim`

# + [markdown] id="oYHDp-WZOj4Z"
# ## ¿Cómo representamos las palabra?

# + [markdown] id="qmfWiGDDPBlX"
# Las palabras tienen un significado y nos da una imagen mental
#
#
# + [markdown] id="oumlomiXS7pS"
# ###Interpretación lingüística
#
# - Significante -> Significado
#
# Interpretamos el signo lingüístico
#
# <img src="https://pymstatic.com/22726/conversions/diferencias-significado-significante-social.jpg" width=500>

# + [markdown] id="HV5-sOecS9vE"
# ## Haciendo útil al significado para las computadoras (solo inglés)
#
# Una forma de almacenar las relaciones semánticas de las palabras fue usando WordNets. Una suerte de diccionario + tesauro[1] para automatizar tareas de análisis de textos
#
# [1]: https://es.wikipedia.org/wiki/Tesauro

# + colab={"base_uri": "https://localhost:8080/"} id="D4AkIkFKWDCJ" outputId="f6fc81d3-696c-4fd0-931c-e8a4f99e9087"
import nltk

nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

# + colab={"base_uri": "https://localhost:8080/"} id="a0rHEKX8Vg2e" outputId="af86a88e-d458-4e2c-94b8-b5f371656708"
from nltk.corpus import wordnet as wn

bear = wn.synset("bear.n.01")
hyper = lambda s: s.hypernyms()
list(bear.closure(hyper))

# + colab={"base_uri": "https://localhost:8080/"} id="pObOb5BbWNf_" outputId="f2b2fb45-5541-4a50-c874-8a892814bfc9"
wn.synonyms("auto", lang="spa")

# + [markdown] id="IBxY2yCsTKtA"
# ### Desventajas
#
# - Como las cosas basadas en reglas es díficil de mantener en el tiempo
#   - Nuevas palabras
#   - Subjetivo
#   - Requiere muchisimo trabajo manual especializado

# + [markdown] id="tjukJoODdbez"
#  ## Representaciones vectoriales
#

# + [markdown] id="afUdI9q46hJN"
# - Sabemos que las computadoras son buenas computando numero no textos crudo
#   - Especialemente metodos de ML que son los que utilizamos para resolver tareas
#   - Clasificacion de documentos
#   - Rankear documentos por reelevancia dada una query
# - Buscamos una forma de mapear textos -> **espacio vectorial**
#  - Un enfoque muy utilizado es la Bolsa de Palabras (Bag of Words)
#    - Matriz de documentos-terminos
#    - Cada fila es un vector con $N$ features donde las features serán el vocabulario del corpus

# + [markdown] id="5JBqJNfV6iox"
# <img src="https://sep.com/wp-content/uploads/2020/12/2020-07-bagofwords.jpg">

# + id="HJ2xVdsqeUEe"
import gensim

# + id="9kEUUJzBhqyH"
doc_1 = "Augusta Ada King, condesa de Lovelace (Londres, 10 de diciembre de 1815-íd., 27 de noviembre de 1852), registrada al nacer como Augusta Ada Byron y conocida habitualmente como Ada Lovelace, fue una matemática y escritora británica, célebre sobre todo por su trabajo acerca de la computadora mecánica de uso general de Charles Babbage, la denominada máquina analítica. Fue la primera en reconocer que la máquina tenía aplicaciones más allá del cálculo puro y en haber publicado lo que se reconoce hoy como el primer algoritmo destinado a ser procesado por una máquina, por lo que se le considera como la primera programadora de ordenadores."
doc_2 = "Brassica oleracea var. italica, el brócoli,1​ brécol2​ o bróquil3​ del italiano broccoli (brote), es una planta de la familia de las brasicáceas. Existen otras variedades de la misma especie, tales como: repollo (B. o. capitata), la coliflor (B. o. botrytis), el colinabo (B. o. gongylodes) y la col de Bruselas (B. o. gemmifera). El llamado brócoli chino o kai-lan (B. o. alboglabra) es también una variedad de Brassica oleracea."
doc_3 = "La bicicleta de piñón fijo, fixie o fixed es una bicicleta monomarcha, que no tiene piñón libre, lo que significa que no tiene punto muerto; es decir, los pedales están siempre en movimiento cuando la bicicleta está en marcha. Esto significa que no se puede dejar de pedalear, ya que, mientras la rueda trasera gire, la cadena y los pedales girarán siempre solidariamente. Por este motivo, se puede frenar haciendo una fuerza inversa al sentido de la marcha, y también ir marcha atrás."

# + id="4GMdxLmueGoZ"
documents = [doc_1, doc_2, doc_3]

# + id="121CNkZ9eNBc"
from gensim.utils import simple_preprocess

def sent_to_words(sentences: list[str]) -> list[list[str]]:
    """Function convert sentences to words

    Use the tokenizer provided by gensim using
    `simple_process()` which remove punctuation and converte
    to lowercase (`deacc=True`)
    """
    return [simple_preprocess(sent, deacc=True) for sent in sentences]



# + id="fq6yZ-KlfM9s" colab={"base_uri": "https://localhost:8080/"} outputId="965cbe49-21dc-4804-fae2-ec210916277e"
docs_tokenized = sent_to_words(documents)
docs_tokenized[2][:10]

# + id="6I37u6r5fYpc"
from gensim.corpora import Dictionary

gensim_dic = Dictionary()
bag_of_words_corpus = [gensim_dic.doc2bow(doc, allow_update=True) for doc in docs_tokenized]

# + id="35dnVyLk2fFq" colab={"base_uri": "https://localhost:8080/", "height": 186} outputId="7396aec6-7604-4442-ef5b-b629ba43f47c"
type(gensim_dic)

# + id="duD7ekPjyurT" colab={"base_uri": "https://localhost:8080/"} outputId="ef2445e3-a06a-4f38-b0af-2fc4afce4931"
for k, v in gensim_dic.iteritems():
    print(k, v)

# + id="qgWTeVYJyMOD" colab={"base_uri": "https://localhost:8080/"} outputId="38d594f2-6bd8-4a7c-c749-e66f878ab29b"
print(len(bag_of_words_corpus))
bag_of_words_corpus[0]


# + id="k19nlQhR_jLk"
def bag_to_dict(bag_of_words: list, gensim_dic: Dictionary, titles: list[str]) -> list:
    data = {}
    for doc, title in zip(bag_of_words, titles):
        data[title] = dict([(gensim_dic[id], freq) for id, freq in doc])
    return data


# + id="rqfk-85UyPb9"
data = bag_to_dict(bag_of_words_corpus, gensim_dic, titles=["ADA", "BROCOLI", "FIXED"])

# + id="Oj2Lh0gJlA_W" colab={"base_uri": "https://localhost:8080/"} outputId="c049a2b9-0c27-4e35-f756-96ece53ac08d"
data

# + id="8XF444cA2U5r"
import pandas as pd

doc_matrix_simple = pd.DataFrame(data).fillna(0).astype(int).T

# + id="kInN4KlWleRR" colab={"base_uri": "https://localhost:8080/", "height": 193} outputId="7d754252-cd9f-454b-9422-de6763d8557f"
doc_matrix_simple

# + [markdown] id="bVPeYSW_QUeW"
# - Tenemos una matrix de terminos-frecuencias ($tf$). Es decir cuantas veces un termino aparece en cierto documento.
# - Una variante de esta es una **BoW** binaria. ¿Cómo se vería?
#

# + [markdown] id="mjQxKiYSriBb"
# ## ¿Ven algun problema?

# + [markdown] id="uX5SRzqKrj-v"
# - Palabras muy frecuentes que no aportan signifiancia
# - Los pesos de las palabras son tratados de forma equitativa
#     - Palabras muy frecuentes opacan las menos frecuentes y con mayor significado (semántico) en nuestros documentos
# - Las palabras frecuentes no nos ayudarian a discriminar por ejemplo entre documentos

# + [markdown] id="HSUz5DC2tpsG"
# ## *Term frequency-Inverse Document Frequency* (TF-IDF) al rescate
#
# <center><img src="https://telencuestas.imgix.net/blog/como-ponderar-una-muestra.jpg?auto=format&fit=max&w=3840" height=250></center>

# + [markdown] id="vv323MYjtsaq"
# - Metodo de ponderación creado para algoritmos de Information Retrieval
# - Bueno para clasificación de documentos y clustering
# - Se calcula con la multiplicacion $tf_{d,t} \cdot idf_t$
#
# Donde:
#   - $tf_{d,t}$ es la frecuencia del termino en un documento $d$
#   - $idf_t$ es la frecuencia inversa del termino en toda la colección de documentos. Se calcula de la siguiente forma:
#
# $$idf_t = log_2\frac{N}{df_t}$$
#
# Entonces:
#
# $$tf\_idf(d,t) = tf_{d,t} ⋅ \log_2\frac{N}{df_t}$$

# + [markdown] id="eeHYU9zwwlPD"
# ### Codificando TF-IDF con gensim

# + id="UNlrSu6Fwn7_"
from gensim.models import TfidfModel

tfidf = TfidfModel(bag_of_words_corpus, smartirs="ntc")

# + id="DUbkhdDHxb4J" colab={"base_uri": "https://localhost:8080/"} outputId="7c77435b-8493-45de-d75f-0bc33bd422c9"
bag_of_words_corpus[0]

# + id="BiIzl4G0wi6O" colab={"base_uri": "https://localhost:8080/"} outputId="67573c70-be78-413e-d2cc-9d0465a5983e"
tfidf[bag_of_words_corpus[0]]


# + id="5BTp_hOvBYvK"
def bag_to_dict(bag_of_words: list, gensim_dic: Dictionary, titles: list[str]) -> list:
    data = {}
    tfidf = TfidfModel(bag_of_words, smartirs="ntc")
    for doc, title in zip(tfidf[bag_of_words], titles):
        data[title] = dict([(gensim_dic[id], freq) for id, freq in doc])
    return data


# + id="bNwm-tvpx-F9" colab={"base_uri": "https://localhost:8080/"} outputId="000a0f47-60f8-4f75-f0d6-baeb06801bc4"
data = bag_to_dict(tfidf[bag_of_words_corpus], gensim_dic, titles=["ADA", "BROCOLI", "FIXED"])

# + id="iLui4A_SyKl9" colab={"base_uri": "https://localhost:8080/"} outputId="5ffd50bc-c7d9-45c9-d5a5-b6f1e1244a3a"
data

# + id="BsefTIbWyM6w"
doc_matrix_tfidf = pd.DataFrame(data).fillna(0).T

# + id="zEe3-FmOyYAO" colab={"base_uri": "https://localhost:8080/", "height": 193} outputId="45ed822b-9e18-4c82-867b-fa8c24c14c05"
doc_matrix_tfidf

# + [markdown] id="_NqR5YUF8l0C"
# ### Calculando similitud entre vectores

# + [markdown] id="Eu0NF2oG-G9S"
# <center><img src="https://cdn.acidcow.com/pics/20130320/people_who_came_face_to_face_with_their_doppelganger_19.jpg" width=500></center>

# + [markdown] id="fsRiWh4E8oNf"
# La forma estandar de obtener la similitud entre vectores para **BoW** es con la distancia coseno entre ellos
#
# $$cos(\overrightarrow{v},\overrightarrow{w}) = \frac{\overrightarrow{v} \cdot\overrightarrow{w}}{|\overrightarrow{v}||\overrightarrow{w}|}$$
#
# Aunque hay muchas más formas de [calcular la distancia](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) entre vectores

# + [markdown] id="4Fkh3JogYjB2"
# ### Calculando distancia entre vectores

# + id="UR3j1Cyg0d8e" colab={"base_uri": "https://localhost:8080/"} outputId="976c881b-0ad3-4807-aacc-ed66f4ad8c59"
from sklearn.metrics.pairwise import cosine_similarity
doc_1 = doc_matrix_tfidf.loc["BROCOLI"].values.reshape(1, -1)
doc_2 = doc_matrix_tfidf.loc["FIXED"].values.reshape(1, -1)
cosine_similarity(doc_1, doc_2)


# + id="Nknx6Rcv1drd"
def update_bow(doc: str, bag_of_words: list, gensim_dic: Dictionary) -> pd.DataFrame:
    words = simple_preprocess(doc, deacc=True)
    bag_of_words.append(gensim_dic.doc2bow(words, allow_update=True))
    return bag_of_words


# + id="wzu2nsBt3zcm"
#sample_doc = "Las bicicletas fixie, también denominadas bicicletas de piñón fijo, son bicis de una sola marcha, de piñón fijo, y sin punto muerto, por lo que se debe avanzar, frenar y dar marcha atrás con el uso de los pedales. La rueda de atrás gira cuando giran los pedales. Si pedaleas hacia delante, avanzas; si paras los pedales, frenas y si pedaleas hacia atrás, irás marcha atrás. Esto requiere de un entrenamiento añadido que la bicicleta con piñón libre no lo necesita. No obstante, las bicicletas fixie tienen muchísimas ventajas."
sample_doc = "El brócoli o brécol es una planta de la familia de las brasicáceas, como otras hortalizas que conocemos como coles. Está por tanto emparentado con verduras como la coliflor, el repollo y las diferentes coles lisas o rizadas, incluyendo el kale o las coles de Bruselas."

# + id="DwD2qDl03_Bb" colab={"base_uri": "https://localhost:8080/"} outputId="66b5ac05-a061-4db2-dd9a-1ef5ec03981c"
new_bag = update_bow(sample_doc, bag_of_words_corpus.copy(), gensim_dic)
len(new_bag)

# + id="UXoTGo0CDR9L" colab={"base_uri": "https://localhost:8080/"} outputId="1260bae9-56b2-4ac8-cb21-68d39b91f401"
for k, v in gensim_dic.iteritems():
    print(k, v)

# + id="WqhYSPxFCcCZ" colab={"base_uri": "https://localhost:8080/"} outputId="ab3700d4-629a-4594-bd53-af3604cf3671"
new_data = bag_to_dict(new_bag, gensim_dic, ["ADA", "BROCOLI", "FIXED", "SAMPLE"])

# + id="cZWWhN1KEXBa" colab={"base_uri": "https://localhost:8080/", "height": 224} outputId="7ed6c4ab-6f90-4202-99e3-41f203632aec"
new_doc_matrix_tfidf = pd.DataFrame(new_data).fillna(0).T
new_doc_matrix_tfidf

# + id="Tyc4vY1E1Nn1" colab={"base_uri": "https://localhost:8080/"} outputId="43193e17-1e1a-435b-f31c-6d74169f625c"
doc_sample_values = new_doc_matrix_tfidf.loc["SAMPLE"].values.reshape(1, -1)

doc_titles = ["ADA", "BROCOLI", "FIXED"]
for i, doc_title in enumerate(doc_titles):
    current_doc_values = new_doc_matrix_tfidf.loc[doc_title].values.reshape(1, -1)
    print(f"Similarity beetwen SAMPLE/{doc_title}= {cosine_similarity(current_doc_values, doc_sample_values)}")

# + [markdown] id="MFMSl17Z4glM"
# ### Desventajas de las BoW

# + [markdown] id="6riRrRAR4i5V"
# - Vectores de enorme dimensionalidad
# - La relación semántica entre palabras no es modelada
# - El orden de las palabras es ignorado

# + [markdown] id="lCzLekpMGh_1"
# ## Distributional Semantic Models (DSM) al rescate

# + [markdown] id="iVpHMcvKG0cQ"
# <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExamdsMDg2eWk5bjMwdGRqa2Rtcm8waWtlbGNyZzE1Y3g2ZTk2Y3pncCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qtelamtlayGWs/giphy.gif">

# + [markdown] id="OVAJ92hlHWlq"
# - Teoria linguistica que se basa en el hecho que dice que, palabras que ocurren en contextos similares tienen significados similares
# > You shall know a word by the company it keeps - J.R. Firth
#
# - Las palabras son representadas por vectores que capturan el patrón de co-ocurrencia de una palabra con muchas otras en el corpus
# - Cada palabra sera un vector, tal que palabras que aparecen en contextos similares, seran representados por vectores similares
#
# ```json
# ¿Me regalas un 'caxitl' de agua?
# El 'caxitl' de vidrio se rompió
# Sirve la leche en el 'caxitl'
# ```

# + [markdown] id="lcBOgIieIQUN"
# ### ¿Qué significa *caxitl*?

# + [markdown] id="E-8IGzyWIV27"
# <img src="https://www.sdpnoticias.com/resizer/Ri0ntQm5YWqR-2Mq0XKlBl_ExtI=/640x963/filters:format(jpg):quality(90)/cloudfront-us-east-1.images.arcpublishing.com/sdpnoticias/DGJILEMTYNBPZHO4D7KM63PPAY.jpg" height=300>

# + [markdown] id="NSE0bl9DJ-8m"
# ## Sabores de DSM
#
# - Tenemos dos sabores grandes de DSM:
#   - Count-based DSM
#   - Prediction-based DSM

# + [markdown] id="hDysL147LBHg"
# ### Count-based

# + [markdown] id="bxRtiRKFLL9s"
# - Matriz con el patron de co-ocurrenca con otras palabras
#   - Esto tiene la mejora de modelar las relaciones semanticas entre palabras
#   - Aun tenemos vectores enormes (¿Cómo podriamos solucionar esto?)
# - Se define una ventana $L$ para el contexto
#

# + [markdown] id="FZLBAiowchfx"
# _K = The dusty road ends nowhere. The dusty track ends there._

# + [markdown] id="8Hhm-VBiLCxP"
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/cooccurenceMatrixExample.png">

# + [markdown] id="oi0Y5LIbNFzh"
# ### Predicted-based

# + [markdown] id="I3j1qC6zNHmi"
# - En 2013 Mikolov et al. publican el paper: [Efficient Estimation of Word Representations in Vector Space](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
# - Muestran una forma eficiente de obtener DSM word-embeddings (Word2Vec):
#     - CBOW
#     - Skipgram
# - De ahí surgieron variaciones
#     - GloVe
#     - fastText
#     - ELMo

# + [markdown] id="mqX0VpqiRE7d"
#  ## Aplicando Pre-trained Word Embbedings

# + id="mXOtfdNSK_Wb"
import gensim.downloader as gensim_api

# + colab={"base_uri": "https://localhost:8080/"} id="CDNn8WLZK-gF" outputId="d64f46dc-0d99-4f1f-de5d-f1771a6c74a6"
gensim_api.info(name_only=True)

# + id="2boe8G0RR7vK" colab={"base_uri": "https://localhost:8080/"} outputId="caa8fdc4-d3d3-4140-cef0-d8a6402c9950"
word_vectors = gensim_api.load("glove-twitter-100")

# + id="h2XujnHASFX3" colab={"base_uri": "https://localhost:8080/"} outputId="828387ff-7735-437f-900c-c8f7ec6cd16d"
print("Information about vectors")
print(f"Tokens={word_vectors.vectors.shape[0]}")
print(f"Dimension de vectores={word_vectors.vectors.shape[1]}")

# + id="SGreQPKEUgza" colab={"base_uri": "https://localhost:8080/"} outputId="69517716-c6ea-4e37-c5eb-575070f9f522"
word_vectors.index_to_key[40:50]

# + id="ZthtlQdtStks" colab={"base_uri": "https://localhost:8080/"} outputId="4f292e11-8185-423d-c286-bcff0e626ca6"
word_vectors["sun"][:10]

# + id="5GGZYrNfSyi1" colab={"base_uri": "https://localhost:8080/"} outputId="70689f90-6a12-4111-fdb7-fcb53055f552"
word_vectors["moon"][:10]

# + id="FjV_qyx8S5DK" colab={"base_uri": "https://localhost:8080/"} outputId="8ed27e65-ae15-4686-c0ed-0858e747defe"
word_vectors.similarity("cat", "dog")

# + id="jYC3q-rJTCmS" colab={"base_uri": "https://localhost:8080/"} outputId="3e20e9a5-977d-422c-8cb3-7b2052a22409"
word_vectors.most_similar("obama", topn=10)

# + id="xHKcUho5TasO" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="3d62f39c-a864-4da5-a995-9e174a804e6d"
word_vectors.doesnt_match("car motor ponny oil mustang".split())

# + id="UjWzxmJlVoFR" colab={"base_uri": "https://localhost:8080/"} outputId="c97dfae3-94be-4689-a429-cffa8c7e69a0"
word_vectors.n_similarity(['mexican', 'market'], ['japanese', 'restaurant'])

# + id="jZSzmIekVscz" colab={"base_uri": "https://localhost:8080/"} outputId="b4744961-e29a-4d01-daf1-0a713e14dba4"
vector = word_vectors['computer']
print(vector.shape)
print(vector[:10])

# + id="soHDSo8hXTja" colab={"base_uri": "https://localhost:8080/"} outputId="d4c22c8a-f6d6-44c7-a125-9f337c8f7b6a"
word_vectors.distance("dog", "cat")

# + id="04Az5nz5Wv4A" colab={"base_uri": "https://localhost:8080/"} outputId="d1b4b2d1-4bf7-4736-e865-d7ed047385ed"
# boy, father, shirt
word_vectors.most_similar(positive=['hospital', 'man'], negative=['woman'])

# + id="yvVxSf8MXeC7" colab={"base_uri": "https://localhost:8080/"} outputId="fb50d8f6-46ce-4f94-98d7-28241f3e4833"
# usa, mexico, australia
word_vectors.most_similar(positive=['london', 'australia'], negative=['england'])


# + id="WTnr1xagYK6j" colab={"base_uri": "https://localhost:8080/"} outputId="1bde0b7d-7b35-4fc9-ac65-292688ba2a61"
# animals (mr gifts)
word_vectors.most_similar(positive=["dog", "oink"], negative=["pig"])

# + [markdown] id="m46cC-ShV-oz"
# ## Visualizando vectores

# + [markdown] id="1pRGBjvGh2eg"
# - Ver vectores en un espacio vectorial mayor a 3 es imposible para las personas
# - Para poder verles necesitamos reducir la dimensión de nuestros vectores
# - Aplicaremos algoritmos de reducción de dimensionalidad

# + id="sTg1OhTlV6dW"
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

# + id="zN1SJ1AFWTq9"
import warnings
warnings.filterwarnings("ignore")

# + id="ZAr_1b5BWB3X"
tsneModel=TSNE(n_components=2,random_state=0)
np.set_printoptions(suppress=True)
model2d=tsneModel.fit_transform(word_vectors[word_vectors.index_to_key[300:600]])

# + id="iYzkR57zWGHZ" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="b867b491-372d-44ed-ede3-8842096ec157"
# %matplotlib inline
plt.figure(figsize=(15,15))
idx = 0
for a in model2d[:300]:
    w = word_vectors.index_to_key[300+idx]
    plt.plot(a[0],a[1],'r.')
    plt.text(a[0],a[1],w)
    idx += 1
plt.show()

# + [markdown] id="Nfm9G8lEoxTB"
# ## Entrenando nuestras propias representaciones vectoriales

# + [markdown] id="iqnf_8mBcvR4"
# ![we](https://miro.medium.com/v2/resize:fit:2000/1*SYiW1MUZul1NvL1kc1RxwQ.png)

# + [markdown] id="JmAB59uGj9NA"
# ### Obteniendo datos

# + [markdown] id="a-qbC4aPj_U7"
# ![](https://data-and-the-world.onrender.com/posts/read-wikipedia-dump/dump_file_list.png)

# + [markdown] id="akr7RvSTHaVm"
# Trabajaremos con una parte de la wikipedia en español. Usaremos la herramienta [wikiextractor](https://github.com/attardi/wikiextractor) y obtendremos los datos de la página: https://dumps.wikimedia.org/eswiki/

# + id="VglXmTCwJps3"
# !pip install wikiextractor

# + id="SO3hGZToYP1E"
import urllib.request
from tqdm import tqdm

#CORPORA_DIR = "corpora/word2vec/"
CORPORA_DIR = "drive/MyDrive/corpora/word2vec/"

# + colab={"base_uri": "https://localhost:8080/"} id="jgv5U_fObiog" outputId="ddefc2d7-154d-41d6-9758-c9437801ab25"
from google.colab import drive
drive.mount('/content/drive')

# + id="obV68MAhBVzS"
# url = "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles1.xml-p1p159400.bz2"
#url = "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles2.xml-p159401p693323.bz2"
url = "https://dumps.wikimedia.org/eswiki/latest/eswiki-latest-pages-articles3.xml-p693324p1897740.bz2"
filename = CORPORA_DIR + "eswiki-articles3.bz2"

with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as t:
    urllib.request.urlretrieve(url, filename, reporthook=lambda block_num, block_size, total_size: t.update(block_size))

# + colab={"base_uri": "https://localhost:8080/"} id="VWb_gHan12Z2" outputId="9fd40cbe-8476-466d-d1d8-ee2acd995180"
import multiprocessing
multiprocessing.cpu_count()

# + colab={"base_uri": "https://localhost:8080/"} id="6D3qT1G4x-l0" outputId="dfaa46b2-12f8-4193-d78e-49a7ca3c71a2"
# !cat /proc/cpuinfo

# + colab={"base_uri": "https://localhost:8080/"} id="diLbver0MCaa" outputId="a94a76b1-9adc-472a-f21f-9254c95fd1d7"
# %%time
# !python -m wikiextractor.WikiExtractor "corpora/word2vec/eswiki-articles-1.bz2" --processes 24 -o "corpora/word2vec/eswiki-dump-1"

# + id="8IVWfOzuCgvA"
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class WikiSentencesExtractor(object):

    def __init__(self, directory, max_lines):
        self.directory = directory
        self.max_lines = max_lines
        self.total_sentences = 0

    @staticmethod
    def preprocess(text: str) -> list:
        if len(text) <= 3 or text.startswith("<"):
            return []
        text = text.lower()
        text = re.sub(f'[^\w\s]', '', text)

        words = word_tokenize(text, language="spanish")

        stop_words = set(stopwords.words("spanish"))
        words = [token for token in words if token not in stop_words]
        words = [token for token in words if token.isalpha() and len(token) > 2]
        return words

    def get_sentences(self):
        for subdir_letter in os.listdir(self.directory):
            file_path = os.path.join(self.directory, subdir_letter)
            for file_name in os.listdir(file_path):
                with open(os.path.join(file_path, file_name)) as file:
                    for line in file:
                        if self.max_lines == self.total_sentences:
                            return
                        words = self.preprocess(line)
                        if not words:
                            continue
                        yield words
                        self.total_sentences += 1

    def __iter__(self):
        return self.get_sentences()

    def __len__(self):
        return self.total_sentences


# + colab={"base_uri": "https://localhost:8080/"} id="IkTTztly4cla" outputId="70503ce8-ca04-4a73-ef79-54282bbd0c55"
directory = CORPORA_DIR + "eswiki-dump"
os.listdir(directory)

# + colab={"base_uri": "https://localhost:8080/"} id="9w4RLfKHX9DB" outputId="f148c171-284d-424e-d394-344e21152ba0"
# %%time
sentences = WikiSentencesExtractor(directory, 3)

# + colab={"base_uri": "https://localhost:8080/"} id="CcPhdqmyF89-" outputId="bcde6109-23b5-4e8d-d636-e65bcfe92228"
for sentence in sentences:
    print(sentence)

# + id="-uCzHEBKEQse"
from gensim.models import word2vec, FastText

# + id="3DKvpkpKYP1F"
#MODELS_DIR = "models/word2vec/"
MODELS_DIR = "drive/MyDrive/models/word2vec/"

# + colab={"base_uri": "https://localhost:8080/"} id="q0S7bc7I5QBJ" outputId="a0dfe1be-acb1-470b-f7b9-1c06d7b439e2"
# %%time
model_name = MODELS_DIR + "eswiki-test.model"
try:
    print(f"Searching for model {model_name}")
    model = word2vec.Word2Vec.load(model_name)
    print("Model found!!!")
except Exception as e:
    print(f"Modelo {model_name} not found. Train a new one")
    model = word2vec.Word2Vec(
        list(WikiSentencesExtractor(directory, max_lines=100000)),
        vector_size=100,
        window=5,
        workers=multiprocessing.cpu_count()
        )
    model.save(model_name)
    print(f"Finish train for model {model_name}")

# + id="IeE5R0bWYP1F"
# Probando mi modelo

# + id="jFarUmsQ3w-o"
from enum import Enum

class Algorithms(Enum):
    CBOW = "CBOW"
    SKIP_GRAM = "SKIP_GRAM"
    FAST_TEXT = "FAST_TEXT"


# + id="I3mQmAQuBVzf"
def load_model(model_path: str):
    try:
        return word2vec.Word2Vec.load(model_path)
    except:
        print(f"[WARN] Model not found in path {model_path}")
        return None


# + id="nKNeVb-BAx8k"
def train_model(sentences, model_name: str, vector_size: int, window=5, workers=2, algorithm = Algorithms.CBOW):
    model_name_params = f"{model_name}-vs{vector_size}-w{window}-{algorithm.value}.model"
    model_path = MODELS_DIR + model_name_params
    if load_model(model_path) is not None:
        print(f"Already exists the model {model_path}")
        return load_model(model_path)
    print(f"TRAINING: {model_path}")
    if algorithm in [Algorithms.CBOW, Algorithms.SKIP_GRAM]:
        algorithm_number = 1 if algorithm == Algorithms.SKIP_GRAM else 0
        model = word2vec.Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            workers=workers,
            sg = algorithm_number,
            seed=42,
            )
    elif algorithm == Algorithms.FAST_TEXT:
        model = FastText(sentences=sentences, vector_size=vector_size, window=window, workers=workers, seed=42, epochs=100)
    else:
        print("[ERROR] algorithm not implemented yet :p")
        return
    model.save(model_path)
    return model


# + id="eYU-ydsAuu8o"
def report_stats(model) -> None:
    """Print report of a model"""
    print("Number of words in the corpus used for training the model: ", model.corpus_count)
    print("Number of words in the model: ", len(model.wv.index_to_key))
    print("Time [s], required for training the model: ", model.total_train_time)
    print("Count of trainings performed to generate this model: ", model.train_count)
    print("Length of the word2vec vectors: ", model.vector_size)
    print("Applied context length for generating the model: ", model.window)


# + [markdown] id="PaXymgOp3uj_"
# ### CBOW

# + colab={"base_uri": "https://localhost:8080/"} id="OnED0jkeuzm-" outputId="3499afde-cda9-469c-8fb1-2976a48f00ba"
# %%time
cbow_100 = train_model(
    WikiSentencesExtractor(directory, -1),
    "eswiki-medium",
    vector_size=100,
    window=5,
    workers=24,
    algorithm=Algorithms.CBOW
    )

# + colab={"base_uri": "https://localhost:8080/"} id="ldk-St_xT2nA" outputId="86a5b4db-dbfe-4bde-d001-592f38886396"
report_stats(cbow_100)

# + [markdown] id="Ue_kS9wY3ylr"
# ### Skip gram

# + colab={"base_uri": "https://localhost:8080/"} id="YIu3E8S243eh" outputId="16c3c1dd-003e-49d8-a7a2-f51c53658a49"
# %%time
skip_gram_100 = train_model(WikiSentencesExtractor(directory, -1), "eswiki-medium", 100, 5, workers=24, algorithm=Algorithms.SKIP_GRAM)

# + id="ilrOqGTvUC_t"
report_stats(skip_gram_100)

# + [markdown] id="sOHEgSce31eb"
# ### fastText

# + [markdown] id="6NpECERd4fyz"
# fastText toma en cuenta la estructura morfologica de las palabras. Esta estructura no es tomada en cuenta en los modelos tradicionales de Word2Vec.
#
# Para hacerlo con fastText se toma la palabra como un agregado de sub-tokens que generalmente y por simplicidad se calculan como los n-gramas de la palabra.
#
# Sauce- https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html#fasttext-model

# + colab={"base_uri": "https://localhost:8080/"} id="iiFsl-i35vOw" outputId="1e4f04fe-3da9-421f-a68e-83ca39f55fc8"
# %%time
fastext_300 = train_model(
    WikiSentencesExtractor(directory, -1),
    "eswiki-medium",
    300,
    5,
    workers=24,
    algorithm=Algorithms.FAST_TEXT
    )

# + id="FInGcq17UGgc"
report_stats(fastext_300)

# + [markdown] id="B4h13MzYqTn6"
# ## Operaciones con los vectores entrenados
#
# Veremos operaciones comunes sobre vectores. Estos resultados dependeran del modelo que hayamos cargado en memoria

# + id="1zs1UIGp2rTD"
models = {
    Algorithms.CBOW: cbow_100,
    Algorithms.SKIP_GRAM: skip_gram_100,
    Algorithms.FAST_TEXT: fastext_300
}

# + id="eE0r8Z7fPs6p"
model = models[Algorithms.FAST_TEXT]

# + colab={"base_uri": "https://localhost:8080/"} id="CJwGqBVNPvhH" outputId="0066d8b8-905f-4a75-f7e5-89c204d3f68e"
for index, word in enumerate(model.wv.index_to_key):
    if index == 100:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

# + colab={"base_uri": "https://localhost:8080/"} id="USTedcpAXLjZ" outputId="70c50090-7fac-485f-a120-0411133018eb"
gato_vec = model.wv["gato"]
print(gato_vec[:10])
print(len(gato_vec))

# + id="7iwhkXvJQIEP"
try:
    agustisidad_vec = model.wv["agusticidad"]
except KeyError:
    print("OOV founded!")


# + colab={"base_uri": "https://localhost:8080/"} id="oC_ckjyg6Vgw" outputId="8932cea5-31c2-4005-bcef-82bcab5803e0"
agustisidad_vec[:10]
len(agustisidad_vec)

# + colab={"base_uri": "https://localhost:8080/"} id="fwruZhD_hNL6" outputId="298417eb-4f57-46f4-bbda-39deef37e3d2"
model.wv.most_similar("agusticidad", topn=5)

# + [markdown] id="n170PD_8RGnT"
# Podemos ver como la similitud entre palabras decrece

# + colab={"base_uri": "https://localhost:8080/"} id="i9wBETdgQaNS" outputId="0bc8604e-8a87-4878-880d-4ae0a6170a59"
word_pairs = [
    ("automóvil", "camion"),
    ("automóvil", "bicicleta"),
    ("automóvil", "cereal"),
    ("automóvil", "conde"),
]

for w1, w2 in word_pairs:
    print(f"{w1} - {w2} {model.wv.similarity(w1, w2)}")

# + colab={"base_uri": "https://localhost:8080/"} id="wukj9jTlSOCc" outputId="e60d4e5d-4030-4f15-efe0-eef917608d03"
# rey es a hombre como ___ a mujer
# londres es a inglaterra como ____ a vino
model.wv.most_similar(positive=['vida', 'enfermedad'], negative=['salud'])

# + colab={"base_uri": "https://localhost:8080/", "height": 34} id="WHZ25Le-6hSN" outputId="a2074156-ae9a-4900-f094-fe862db618bf"
model.wv.doesnt_match(["disco", "música", "mantequilla", "cantante"])

# + colab={"base_uri": "https://localhost:8080/"} id="cRkFqZXM9jBQ" outputId="022135f1-93fd-4a23-98e3-c37be740f797"
model.wv.similarity("noche", "noches")

# + colab={"base_uri": "https://localhost:8080/"} id="5nUpUc8t4LDl" outputId="4b7955b3-cb43-4f02-94c8-bf79c19d685b"
model.wv.most_similar("nochecitas", topn=10)

# + [markdown] id="j_NZcB-Z5bmO"
# ### Ejemplos destacables
#
# - [Tutorial Word Embedding](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/index.html)
# - [Demo](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/index.html)

# + [markdown] id="kjt7nbZuYP1I"
# # Práctica 5: Reducción de la dimensionalidad
#
# **Fecha de entrega: 13 de abril 2024 11:59pm**

# + [markdown] id="3_KJH0i4YP1I"
# Hay varios métodos que podemos aplicar para reduccir la dimensionalidad de nuestros vectores y asi poder visualizar en un espacio de menor dimensionalidad como estan siendo representados los vectores.
#
# - PCA
# - T-SNE
# - SVD
#
# - Entrenar un modelo word2vec
#   - Utilizar como corpus la wikipedia como en la practica
#   - Adaptar el tamaño de ventana y corpus a sus recursos de computo
#   - Ej: Entrenar en colab con ventana de 5 y unas 100k sentencias toma ~1hr
# - Aplicar los 3 algoritmos de reduccion de dimensionalidad
#     - Reducir a 2d
#     - Plotear 1000 vectores de las palabras más frecuentes
# - Analizar y comparar las topologías que se generan con cada algoritmo
#   - ¿Se guardan las relaciones semánticas? si o no y ¿porqué?
#   - ¿Qué método de reducción de dimensaionalidad consideras que es mejor?

# + [markdown] id="_4NuMOSJFzhI"
# # Referencias
#
# - [Verctor Representations - Dr. Johannes Maucher](https://hannibunny.github.io/nlpbook/05representations/05representationsintro.html)
# - [Word Embeddings - Lenia Voita](https://lena-voita.github.io/nlp_course/word_embeddings.html#pre_neural)
# - Partes del código utilizado para este notebook fueron tomados de trabajos de la [Dr. Ximena Gutierrez-Vasques](https://github.com/ximenina/) y el [Dr. Victor Mijangos](https://github.com/VMijangos/LinguisticaComputacional/blob/main/Notebooks/19%20Word2Vec.ipynb)
# - [Corpus streaming on gensim](https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-one-document-at-a-time)
# - [Gensim docs](https://radimrehurek.com/gensim/auto_examples/index.html)

# + id="U5w3pQEg6Q_F"


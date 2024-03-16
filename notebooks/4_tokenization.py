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

# + [markdown] editable=true id="UUPaNJXa1Om4" slideshow={"slide_type": "slide"}
# # 4. Tokenization

# + [markdown] editable=true id="rbo4KM2tJEuZ" slideshow={"slide_type": "subslide"}
# ## Objetivo
#
# - Læs alumnæs entenderán la importancia de la tokenización en un pipeline de NLP
# - Como varía un corpus sin tokenizar y uno tokenizado
# - Explorar métodos de *Subword tokenization* como: *BPE*, *WordPiece* y *Unigram*

# + [markdown] editable=true id="jqslq1xoJcOB" slideshow={"slide_type": "subslide"}
# ## Tokenization

# + [markdown] editable=true id="0lArM5QU1bJ1" slideshow={"slide_type": "fragment"}
# - Buscamos tener unidades de información para representar una lengua
#     - Transformar nuestro texto crudo en datos que pueda procesar nuestro modelo
#     - Similar a los pixeles para imagenes o frames para audio
# - La unidad más intuitiva son las palabras alfa-numericas separadas por
# espacios (tokens)
# - Segmentación de texto en *tokens* de ahí el nombre *tokenization*
#     - Es una parte fundamental de un *pipeline* de *NLP*
#     - Pre-procesamiento

# + [markdown] editable=true id="EljLPY4G8b6m" slideshow={"slide_type": "subslide"}
# ## Word-based tokenization

# + [markdown] editable=true id="7KHsH6cheGcK" slideshow={"slide_type": "fragment"}
# - Fácil de implementar (`.split()`)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1696883315730, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="2KdF0vK35_1w" outputId="24e9b5fd-e220-4dae-9fbe-358a4796e38c" slideshow={"slide_type": "fragment"}
"Mira mamá estoy en la tele".split()

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Se pueden considerar los signos de puntuación agregando reglas simples

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 167, "status": "ok", "timestamp": 1696889494751, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="hdiNlhUn9pob" outputId="c8281b34-9df8-437c-b9c5-aab3550ac267" slideshow={"slide_type": "fragment"}
import re
text = "Let's get started son!!!"
re.findall(r"['!]|\w+", text)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="rE1SIAfW6XPI" slideshow={"slide_type": "subslide"}
# ### Problem?
#
# <img src="http://images.wikia.com/battlebears/images/2/2c/Troll_Problem.jpg" with="250" height="250">

# + [markdown] editable=true id="EUDLwgXZ-OxQ" slideshow={"slide_type": "fragment"}
# - Vocabularios gigantescos difíciles de procesar
# - Generalmente, entre más grande es el vocabulario más pesado será nuestro modelo
#
# **Ejemplo:**
# - Si queremos representaciones vectoriales de nuestros tokens obtendríamos vectores distintos para palabras similares
#     - niño = `v1(39, 34, 5,...)`
#     - niños = `v2(9, 4, 0,...)`
#     - niña = `v3(2, 1, 1,...)`
#     - ...
# - Tendríamos tokens con bajísima frecuencia
#     - merequetengue = `vn(0,0,1,...)`

# + [markdown] editable=true id="OYTxQzaICZdg" slideshow={"slide_type": "subslide"}
# ### Una Solución: Stemming/Lemmatization (AKA la vieja confiable)
#
# <center><img src="img/vieja_confiable.jpg" width=500 height=500></center>

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 286, "status": "ok", "timestamp": 1696889704924, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="g7tblrHXgb-g" outputId="cced4a12-1a99-46d3-aff6-ad6eb76a93c4" slideshow={"slide_type": "subslide"}
import nltk
from nltk.corpus import brown
nltk.download('brown')

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 4100, "status": "ok", "timestamp": 1696889722087, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="IAzu1OYaf4Fe" outputId="59ce41ec-f837-4ae8-eb81-0d8141940a7e" slideshow={"slide_type": "fragment"}
from collections import Counter

brown_corpus = [word for word in brown.words() if re.match("\w", word)]
print(brown_corpus[0])
print("Tokens:", len(brown_corpus))
print("Tipos:", len(Counter(brown_corpus)))

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# <center><img src="https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fimg1.wikia.nocookie.net%2F__cb20140504152558%2Fspongebob%2Fimages%2Fe%2Fe3%2FThe_spongebob.jpg&f=1&nofb=1&ipt=28368023b54a7c84c9100025981b1042d0f4ca3ceaac53be42094cc1c3794348&ipo=images" height=300 width=300></center>

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 195, "status": "ok", "timestamp": 1696889776155, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="LlZRbuQcjRFU" outputId="9897acb4-d191-46c5-8699-dee3b01b05ff" slideshow={"slide_type": "fragment"}
sub_brown_corpus = brown_corpus[:100000]
print("Sub brown_corpus tipos:", len(Counter(sub_brown_corpus)))
sub_brown_corpus[-5:]

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Lemmatizando ando

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 37078, "status": "ok", "timestamp": 1696889818657, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="p98fuuZvhh3n" outputId="9571e111-ad73-4311-d32a-ac75488d7e6a" slideshow={"slide_type": "fragment"}
# !python -m spacy download en_core_web_sm
# !python -m spacy download es_core_news_sm

# + editable=true executionInfo={"elapsed": 304, "status": "ok", "timestamp": 1696889839336, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="RV3Baz61e2jR" slideshow={"slide_type": "fragment"}
import spacy

def lemmatize(words: list, lang="en") -> list:
    model = "en_core_web_sm" if lang == "en" else "es_core_news_sm"
    nlp = spacy.load(model)
    nlp.max_length = 1500000
    lemmatizer = nlp.get_pipe("lemmatizer")
    return [token.lemma_ for token in nlp(" ".join(words))]


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 28564, "status": "ok", "timestamp": 1696889940048, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="aAST27R7EiRj" outputId="84c50082-20d4-477f-b82a-8da487a91afb" slideshow={"slide_type": "fragment"}
print("tipos (word-based):", len(Counter(sub_brown_corpus)))
print("Tipos (Lemmatized):", len(Counter(lemmatize(sub_brown_corpus))))

# + [markdown] editable=true id="LSylGoAvlbtz" slideshow={"slide_type": "fragment"}
# - eats -> eat
# - eating -> eat
# - eated -> eat
# - ate -> eat

# + [markdown] editable=true id="o4KuoZO5mgtV" slideshow={"slide_type": "subslide"}
# ### More problems?
#
# <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fpreview.redd.it%2Fjoonhzw1sjq31.png%3Fwidth%3D960%26crop%3Dsmart%26auto%3Dwebp%26s%3D3725297033765336276d49958089880e3f64d288&f=1&nofb=1&ipt=fdcf7c99c6a13417957a3832a14ca0f7ac4a70fc906fec79997bcb9795e31054&ipo=images" width="250" height="250">

# + [markdown] editable=true id="pSNd_HLnnPcs" slideshow={"slide_type": "fragment"}
# - Métodos dependientes de las lenguas
# - Se pierde información
# - Ruled-based (?)

# + [markdown] editable=true id="bvmrBSBgndhs" slideshow={"slide_type": "subslide"}
# ## Subword-tokenization salva el día 🦸🏼‍♀️

# + [markdown] editable=true id="rxVa1ke-tvIa" slideshow={"slide_type": "fragment"}
# - Segmentación de palabras en unidades más pequeñas (*sub-words*)
# - Obtenemos tipos menos variados pero con mayores frecuencias
#     - Esto le gusta modelos basados en métodos estadísticos
# - Palabras frecuentes no deberían separarse
# - Palabras largas y raras debería descomponerse en sub-palabras significativas
# - Hay métodos estadisticos que no requieren conocimiento a priori de las lenguas

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 210, "status": "ok", "timestamp": 1696890142885, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="e0ydfVfXvAA9" outputId="b9bbf662-9738-4520-d9fe-548f22f6cb2d" slideshow={"slide_type": "fragment"}
text = "Let's do tokenization!"
result = ["Let's", "do", "token", "ization", "!"]
print(f"Objetivo: {text} -> {result}")

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="Bi91TzS_vfZ7" slideshow={"slide_type": "subslide"}
# ### Métodos para tokenizar
#

# + [markdown] editable=true id="h3nUEOtMvrHp" slideshow={"slide_type": "fragment"}
# - *Byte-pair Encoding, BPE* (🤗, 💽)
# - *Wordpiece* (🤗)
# - *Unigram* (🤗)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 12991, "status": "ok", "timestamp": 1696890239136, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="S4ekcOvYA94b" outputId="158b081e-9d7e-42ed-d966-4134e0b207a5" slideshow={"slide_type": "fragment"}
# !pip install sentencepiece
# !pip install transformers

# + [markdown] editable=true id="KBESt85twGkZ" slideshow={"slide_type": "subslide"}
# ### BPE

# + [markdown] editable=true id="urnnBA7iwKEW" slideshow={"slide_type": "fragment"}
# - Segmenmentación iterativa, comienza segmentando en secuencias de caracteres
# - Junta los pares más frecuentes (*merge operation*)
# - Termina cuando se llega al número de *merge operations* especificado o número de vocabulario deseado (*hyperparams*, depende de la implementación)
# - Introducido en el paper: [Neural Machine Translation of Rare Words with Subword Units, (Sennrich et al., 2015)](https://arxiv.org/abs/1508.07909)

# + colab={"base_uri": "https://localhost:8080/", "height": 438} editable=true executionInfo={"elapsed": 135, "status": "ok", "timestamp": 1696814490047, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="DvhcUk7fBAIG" outputId="b724caee-22e3-4a9e-c722-da5495aa40cf" slideshow={"slide_type": "subslide"}
# %%HTML
<iframe width="960" height="515" src="https://www.youtube.com/embed/HEikzVL-lZU"></iframe>

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo BPE

# + editable=true executionInfo={"elapsed": 205, "status": "ok", "timestamp": 1696890854277, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="BFGXI-61ELmp" slideshow={"slide_type": "fragment"}
SENTENCE = "Let's do this tokenization to enable hypermodernization on my tokens tokenized 👁️👁️👁️!!!"

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 875, "status": "ok", "timestamp": 1696890896162, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="NnRxpoeOD7Ce" outputId="ea3875dd-007e-4d5c-df54-c2f60f1bfb7b" slideshow={"slide_type": "fragment"}
from transformers import GPT2Tokenizer
bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(bpe_tokenizer.tokenize(SENTENCE))

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 326, "status": "ok", "timestamp": 1696890947381, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="wD1hmCbmGyHp" outputId="a97e0575-6ad9-408a-83ab-fb653e0b2d39" slideshow={"slide_type": "fragment"}
encoded_tokens = bpe_tokenizer(SENTENCE)
encoded_tokens["input_ids"]

# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true executionInfo={"elapsed": 214, "status": "ok", "timestamp": 1696891012506, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="zZISJ-PjG6nN" outputId="9d4573d3-bc3f-421d-a94f-5ec2d579c64a" slideshow={"slide_type": "fragment"}
bpe_tokenizer.decode(encoded_tokens["input_ids"])

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="lLEf-PySGBMx" slideshow={"slide_type": "subslide"}
# - En realidad GPT-2 usa *Byte-Level BPE*
#     - Evitamos vocabularios de inicio grandes (Ej: unicode)
#     - Usamos bytes como vocabulario base
#     - Evitamos *Out Of Vocabulary, OOV* (aka `[UKW]`) (?)

# + [markdown] editable=true id="xnaxqfOHHdEo" slideshow={"slide_type": "subslide"}
# ### WordPiece

# + [markdown] editable=true id="G_GezIRiHfdl" slideshow={"slide_type": "fragment"}
# - Descrito en el paper: [Japanese and Korean voice search, (Schuster et al., 2012) ](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
# - Similar a BPE, inicia el vocabulario con todos los caracteres y aprende los merges
# - En contraste con BPE, no elige con base en los pares más frecuentes si no los pares que maximicen la probabilidad de aparecer en los datos una vez que se agregan al vocabulario
#
# $$score(a_i,b_j) = \frac{f(a_i,b_j)}{f(a_i)f(b_j)}$$
#
# - Esto quiere decir que evalua la perdida de realizar un *merge* asegurandoce que vale la pena hacerlo
#
# - Algoritmo usado en `BERT`

# + editable=true slideshow={"slide_type": "subslide"}
# %%HTML
<iframe width="960" height="500" src="https://www.youtube.com/embed/qpv6ms_t_1A"></iframe>

# + editable=true slideshow={"slide_type": ""}


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 648, "status": "ok", "timestamp": 1696891176419, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="R3GbJuV7A6be" outputId="877f6f0b-9456-4a52-a3ae-4b91a07f765e" slideshow={"slide_type": "subslide"}
from transformers import BertTokenizer
SENTENCE = "🌽" + SENTENCE + "🔥"
wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(wp_tokenizer.tokenize(SENTENCE))

# + [markdown] editable=true id="v2S-v6_IJiKH" slideshow={"slide_type": "fragment"}
# <center><img src="https://us-tuna-sounds-images.voicemod.net/9cf541d2-dd7f-4c1c-ae37-8bc671c855fe-1665957161744.jpg"></center>

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 231, "status": "ok", "timestamp": 1696891246358, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="U_l1y4DDJvSq" outputId="73cc32d1-4bc7-41e2-8d7f-c5abd9f8dc0a" slideshow={"slide_type": "fragment"}
wp_tokenizer(SENTENCE)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="LsnetsJ-gWHM" slideshow={"slide_type": "subslide"}
# ### Unigram

# + [markdown] editable=true id="JKia-OlqgYTn" slideshow={"slide_type": "fragment"}
# - Algoritmo de subpword tokenization introducido en el paper: [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018)](https://arxiv.org/pdf/1804.10959.pdf)
# - En contraste con BPE o WordPiece, este algoritmo inicia con un vocabulario muy grande y va reduciendolo hasta llegar tener un vocabulario deseado
# - En cada iteración se calcula la perdida de quitar cierto elemento del vocabulario
#     - Se quitará `p%` elementos que menos aumenten la perdida en esa iteración
# - El algoritmo termina cuando se alcanza el tamaño deseado del vocabulario

# + [markdown] editable=true id="28IEp2eLwFxJ" slideshow={"slide_type": "fragment"}
# <center><img src="img/unigram_loss.png" width=500 height=500></center>

# + [markdown] editable=true id="eqS3rQ56z7oW" slideshow={"slide_type": "subslide"}
# Sin embargo, *Unigram* no se usa por si mismo en algun modelo de Hugging Face:
# > "Unigram is not used directly for any of the models in the transformers, but it’s used in conjunction with SentencePiece." - Hugging face guy

# + [markdown] editable=true id="oaIJpqsY0Hj5" slideshow={"slide_type": "subslide"}
# ### SentencePiece
#

# + [markdown] editable=true id="2Hu7plAT0MFz" slideshow={"slide_type": "fragment"}
# - No asume que las palabras estan divididas por espacios
# - Trata la entrada de texto como un *stream* de datos crudos. Esto incluye al espacio como un caractér a usar
# - Utiliza BPE o Unigram para construir el vocabulario

# + colab={"base_uri": "https://localhost:8080/", "height": 530, "referenced_widgets": ["79162b831cda4631a6213149b61470c4", "e9356268f592457798613eed0481811b", "6ac25032053745ed85299fa6e3e669c2", "64dbfe6a4b27433c9ec39bf267c5b6a3", "361c1d6e88d34840865938090db25ab4", "cc7f7b64537f40ecb4681dbc3bbd02c0", "cc54cb3f9b034407943c41309f6f8b6a", "ef017799af19450985ae4487b4d979c1", "ec826e27c60d40339fea9c4a25edebbe", "a6b888ab688b4374b72489120e69af76", "4504ceecfa6a4a418de3c60b8a9f700b", "272775fb3e1b4eafa9bfb974165b827f", "143bd9cd6990432ca2d668f8054ea2e2", "56ea7547905f44749457685c11399e57", "10418a182a1c40188a996e80642f378f", "9cfcb4be0bca41229ca77f6ef79ab591", "7fec9fcccc984706a4b0a2a6cb4e428d", "c402cd96d2f1427aa547eab16e8fd9a2", "37d4442e5e8f4cb39c55434d19f4fd56", "d685f2635bc84ca4adb7c482bdd05c4c", "7f22daa35eec41ce8e35a38f58cac51b", "f2930bbcd73e4d66a83b07141d1c1ba4", "7757044d639b40b78dd99971d62c6225", "4e04cde4fcfd4318b76eb2d504ebbd39", "4c67e6761dad42718951f9ba69170654", "ab27ed7caeba4b59a34e57555812f401", "4af7e76583074d038a0a7cc15a1de2e0", "ef18442cac994ffda285017ed1e26c89", "b5bc36408cf643a691716e36807f7060", "433723554d85449e800e315722f0958d", "2e3e6f7e1b974553a01e6a277ea17a26", "70e9aea62e9f486d907a43ff66ce8f5b", "4455897594f045138fab5fc1ce70c137"]} editable=true executionInfo={"elapsed": 1909, "status": "ok", "timestamp": 1696891434432, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="S_C5ypqsy81u" outputId="cd90cd67-102c-48c6-e7d3-37f3259d0738" slideshow={"slide_type": "fragment"}
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
print(tokenizer.tokenize(SENTENCE))

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="e0OBaH-s2McQ" slideshow={"slide_type": "subslide"}
# ### Objetivo de los subword tokenizers
#

# + [markdown] editable=true id="b-UWPtZD2sV5" slideshow={"slide_type": "fragment"}
# - Buscamos que modelos de redes neuronales tenga datos mas frecuentes
# - Esto ayuda a que en principio "aprendan" mejor
# - Reducir el numero de tipos (?)
# - Reducir el numero de OOV (?)
# - Reducir la entropia (?)

# + [markdown] editable=true id="rMbXi89C3END" slideshow={"slide_type": "slide"}
# ## Vamos a tokenizar 🌈
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fi.pinimg.com%2F736x%2F75%2F28%2Fe7%2F7528e71db75a37f0dcf5be8a54e0523f.jpg&f=1&nofb=1&ipt=d08ba1ed7fa9af9c3692703a667271740c22bb8e8f5b9f5f7acb44715e7d47d8&ipo=images)

# + [markdown] editable=true id="SNl_R1mqQqWs" slideshow={"slide_type": "subslide"}
# ### Corpus Español: CESS

# + editable=true slideshow={"slide_type": "fragment"}
def normalize_sent(sent: list[str]) -> list[str]:
    return [word.lower() for word in sent if re.match("\w", word)]


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 254, "status": "ok", "timestamp": 1696891556311, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="VCXtaN9fRBCn" outputId="c5004ba1-d603-4ceb-f3f8-529939afa098" slideshow={"slide_type": "fragment"}
nltk.download("cess_esp")

# + editable=true executionInfo={"elapsed": 317, "status": "ok", "timestamp": 1696891564449, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="DtnmYlySQsFE" slideshow={"slide_type": "fragment"}
from nltk.corpus import cess_esp as cess

cess_sents = cess.sents()

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 5931, "status": "ok", "timestamp": 1696891581742, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="Jdd5tW35RPRH" outputId="0d9f1321-38ed-4636-b931-07411d1c03d0" slideshow={"slide_type": "fragment"}
len(cess_sents)

# + editable=true slideshow={"slide_type": "fragment"}
" ".join(cess_sents[0])

# + editable=true executionInfo={"elapsed": 7179, "status": "ok", "timestamp": 1696891605296, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="xbanfyLMSeq1" slideshow={"slide_type": "fragment"}
cess_plain_text = "\n".join([" ".join(normalize_sent(sentence)) for sentence in cess_sents])
cess_plain_text = re.sub(r"[-|_]", " ", cess_plain_text)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1696891605297, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="92qjQXedUIui" outputId="c36944ff-66c3-446c-83af-8bd2631da8b6" slideshow={"slide_type": "subslide"}
len(cess_plain_text)

# + colab={"base_uri": "https://localhost:8080/", "height": 52} editable=true executionInfo={"elapsed": 325, "status": "ok", "timestamp": 1696891618127, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="8ojFGeSCTlmj" outputId="8ab04372-59c6-4041-ac12-deee1b526e6c" slideshow={"slide_type": "fragment"}
print(cess_plain_text[300:600])

# + editable=true slideshow={"slide_type": "fragment"}
cess_words = cess_plain_text.split()

# + editable=true slideshow={"slide_type": "fragment"}
print(cess_words[:100])

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="Ig_H8-oEUwU2" slideshow={"slide_type": "subslide"}
# ### Corpus Inglés: Gutenberg 

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 230, "status": "ok", "timestamp": 1696891629989, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="QPstlJ20UkbE" outputId="bf6bc088-3f02-4411-ddba-1d87752d814f" slideshow={"slide_type": "fragment"}
nltk.download('gutenberg')

# + editable=true executionInfo={"elapsed": 184, "status": "ok", "timestamp": 1696891631728, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="DcDTJyKYQLSJ" slideshow={"slide_type": "fragment"}
from nltk.corpus import gutenberg

gutenberg_sents = gutenberg.sents()[:10000]

# + editable=true slideshow={"slide_type": "fragment"}
len(gutenberg_sents)

# + editable=true slideshow={"slide_type": "fragment"}
" ".join(gutenberg_sents[0])

# + editable=true slideshow={"slide_type": "fragment"}
gutenberg_plain_text = "\n".join([" ".join(normalize_sent(sent)) for sent in gutenberg_sents])

print(gutenberg_plain_text[:100])

# + editable=true slideshow={"slide_type": "skip"}


# + editable=true slideshow={"slide_type": "subslide"}
gutenberg_words = gutenberg_plain_text.split()

# + editable=true slideshow={"slide_type": "fragment"}
gutenberg_words[:10]

# + editable=true slideshow={"slide_type": "fragment"}
len(gutenberg_words)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 215, "status": "ok", "timestamp": 1696891656496, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="r1YjiO5dUpII" outputId="25182cdb-8237-44e4-fff5-c627907c057d" slideshow={"slide_type": "fragment"}
len(gutenberg_plain_text)

# + editable=true slideshow={"slide_type": "fragment"}
with open("corpora/tokenization/gutenberg_plain.txt", "w") as f:
    f.write(gutenberg_plain_text)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="y7ojemCtVEwE" slideshow={"slide_type": "subslide"}
# ### Tokenizando el español con Hugging face

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 182, "status": "ok", "timestamp": 1696891667538, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="-MHPlfhUVHl0" outputId="5c99a957-6e17-4e82-e51d-65d83c2cb583" slideshow={"slide_type": "fragment"}
from transformers import AutoTokenizer

spanish_tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
print(spanish_tokenizer.tokenize(cess_plain_text[1000:1400]))

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 7109, "status": "ok", "timestamp": 1696891722854, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="x-aByv8pXYRQ" outputId="1b455f23-4254-4499-9fc2-7896620d627a" slideshow={"slide_type": "fragment"}
cess_types = Counter(cess_words)
len(cess_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 313, "status": "ok", "timestamp": 1696891737311, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="SbpXXjvKYRW6" outputId="be5ea30d-3c7a-40a1-bbb2-78324deef9f8" slideshow={"slide_type": "fragment"}
print(cess_types.most_common(10))

# + editable=true slideshow={"slide_type": "skip"}


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 1840, "status": "ok", "timestamp": 1696891760449, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="7pYaX_ZEXijm" outputId="90abb358-4222-4c2e-8218-ab544c7bba29" slideshow={"slide_type": "subslide"}
cess_tokenized = spanish_tokenizer.tokenize(cess_plain_text)
cess_tokenized_types = Counter(cess_tokenized)
len(cess_tokenized_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1696891789825, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="MWOUzc9JYG1T" outputId="93c3d1e7-3401-496f-f088-e7a9ed0939a2" slideshow={"slide_type": "fragment"}
print(cess_tokenized_types.most_common(30))

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 45300, "status": "ok", "timestamp": 1696891932543, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="pXxst1GJmnAA" outputId="6d627169-3d8b-42b9-829d-424787e4ef97" slideshow={"slide_type": "fragment"}
cess_lemmatized_types = Counter(lemmatize(cess_words, lang="es"))
len(cess_lemmatized_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 161, "status": "ok", "timestamp": 1696891964154, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="vYPJSUTaoDOY" outputId="3ae68b47-f872-4fac-f987-b17826e270fd" slideshow={"slide_type": "fragment"}
print(cess_lemmatized_types.most_common(30))

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="TI_4lLaDbXLK" slideshow={"slide_type": "subslide"}
# ### Tokenizando para el inglés

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 698, "status": "ok", "timestamp": 1696891988287, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="gZCtfixJbPuE" outputId="f6239fb0-1c37-4c70-8eb7-894fc618b0de" slideshow={"slide_type": "fragment"}
gutenberg_types = Counter(gutenberg_words)
len(gutenberg_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 8348, "status": "ok", "timestamp": 1696892003509, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="K8YuRqJVbii1" outputId="6944e10f-06ac-492d-83ef-8cb98b1fc002" slideshow={"slide_type": "fragment"}
gutenberg_tokenized = wp_tokenizer.tokenize(gutenberg_plain_text)
gutenberg_tokenized_types = Counter(gutenberg_tokenized)
len(gutenberg_tokenized_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 276, "status": "ok", "timestamp": 1696892031063, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="IVyJ3N9xbz3J" outputId="c144bea0-b51a-4dee-95cc-f2a22c1b0057" slideshow={"slide_type": "fragment"}
print(gutenberg_tokenized_types.most_common(100))

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 63580, "status": "ok", "timestamp": 1696892131600, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="jP-6w2yfoQ5g" outputId="e2f6280f-bb87-42c2-d9d8-f37d643a3826" slideshow={"slide_type": "fragment"}
gutenberg_lemmatized_types = Counter(lemmatize(gutenberg_words))
len(gutenberg_lemmatized_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 212, "status": "ok", "timestamp": 1696875126067, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="3FZ8bI7Do00y" outputId="9174a4e8-aa9b-4963-c2b1-c2776833aaba" slideshow={"slide_type": "fragment"}
print(gutenberg_lemmatized_types.most_common(20))

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### OOV: out of vocabulary

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# Palabras que se vieron en el entrenamiento pero no estan en el test

# + editable=true slideshow={"slide_type": "fragment"}
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(gutenberg_words, test_size=0.3, random_state=42)
print(len(train_data), len(test_data))

# + editable=true slideshow={"slide_type": "fragment"}
s_1 = {"a", "b", "c", "d", "e"}
s_2 = {"a", "x", "y", "d"}
print(s_1 - s_2)
print(s_2 - s_1)

# + editable=true slideshow={"slide_type": "skip"}


# + editable=true slideshow={"slide_type": "subslide"}
oov_test = set(test_data) - set(train_data)
len(oov_test)

# + editable=true slideshow={"slide_type": "fragment"}
for word in list(oov_test)[:3]:
    print(f"{word} in train: {word in set(train_data)}")

# + editable=true slideshow={"slide_type": "fragment"}
train_tokenized, test_tokenized = train_test_split(gutenberg_tokenized, test_size=0.3, random_state=42)
print(len(train_tokenized), len(test_tokenized))

# + editable=true slideshow={"slide_type": "fragment"}
oov_tokenized_test = set(test_tokenized) - set(train_tokenized)
len(oov_tokenized_test)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="QCZ9MV7ecmGo" slideshow={"slide_type": "subslide"}
# ## Entrenando nuestro modelo con BPE
# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2Fd565618bb1217a7c435579d9172270d0%2Ftenor.gif%3Fitemid%3D3379322&f=1&nofb=1&ipt=9719714edb643995ce9d978c8bab77f5310204960093070e37e183d5372096d9&ipo=images)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 6470, "status": "ok", "timestamp": 1696892882328, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="Fb7ajFAHcscl" outputId="11d98cbf-0bf7-497f-94b4-ea0e15f08e40" slideshow={"slide_type": "fragment"}
# !pip install subword-nmt

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 221, "status": "ok", "timestamp": 1696892923889, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="JwyeJ5f0dWjE" outputId="fd9174f9-e1f3-4ecc-c212-31e31dcfd56b" slideshow={"slide_type": "fragment"}
# !ls corpora/tokenization

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 1371, "status": "ok", "timestamp": 1696892946423, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="nLTPhYlzdZMB" outputId="079534e2-e586-4a58-cf8e-6321dd9a547a" slideshow={"slide_type": "fragment"}
# !head corpora/tokenization/gutenberg_plain.txt

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 2338, "status": "ok", "timestamp": 1696893003805, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="E_oASS5Vc1xz" outputId="0f4e29ee-096f-4f4c-bb05-86b2931460f5" slideshow={"slide_type": "subslide"}
# !subword-nmt learn-bpe -s 300 < corpora/tokenization/gutenberg_plain.txt > models/tokenization/gutenberg_low.model

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 231, "status": "ok", "timestamp": 1696893148079, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="FA1pLWxx6ETS" outputId="cb79e6cf-423c-4453-91f1-4dd0be2df523" slideshow={"slide_type": "fragment"}
# !echo "I need to process this sentence because tokenization can be useful" | subword-nmt apply-bpe -c models/tokenization/gutenberg_low.model

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 2338, "status": "ok", "timestamp": 1696893003805, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="E_oASS5Vc1xz" outputId="0f4e29ee-096f-4f4c-bb05-86b2931460f5" slideshow={"slide_type": "fragment"}
# !subword-nmt learn-bpe -s 1500 < corpora/tokenization/gutenberg_plain.txt > models/tokenization/gutenberg_high.model

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 231, "status": "ok", "timestamp": 1696893148079, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="FA1pLWxx6ETS" outputId="cb79e6cf-423c-4453-91f1-4dd0be2df523" slideshow={"slide_type": "fragment"}
# !echo "I need to process this sentence because tokenization can be useful" | subword-nmt apply-bpe -c models/tokenization/gutenberg_high.model

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="sbRQG1WNAbjm" slideshow={"slide_type": "subslide"}
# ## Aplicandolo a otros corpus: La biblia 📖🇻🇦

# + editable=true executionInfo={"elapsed": 276, "status": "ok", "timestamp": 1696893195957, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="6XO4EeM8pkxG" slideshow={"slide_type": "fragment"}
BIBLE_FILE_NAMES = {"spa": "spa-x-bible-reinavaleracontemporanea", "eng": "eng-x-bible-kingjames"}
CORPORA_PATH = "corpora/tokenization/"

# + editable=true executionInfo={"elapsed": 270, "status": "ok", "timestamp": 1696893202511, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="VMNjKQD0NONF" slideshow={"slide_type": "fragment"}
import requests

def get_bible_corpus(lang: str) -> str:
    file_name = BIBLE_FILE_NAMES[lang]
    r = requests.get(f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt")
    return r.text

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Biblia en Inglés

# + editable=true slideshow={"slide_type": "fragment"}
eng_bible_plain_text = get_bible_corpus("eng")
eng_bible_words = eng_bible_plain_text.lower().replace("\n", " ").split()

# + editable=true slideshow={"slide_type": "fragment"}
print(eng_bible_words[:10])

# + editable=true slideshow={"slide_type": "fragment"}
len(eng_bible_words)

# + editable=true slideshow={"slide_type": "fragment"}
from collections import Counter
eng_bible_types = Counter(eng_bible_words)
len(eng_bible_types)

# + editable=true slideshow={"slide_type": "fragment"}
print(eng_bible_types.most_common(30))

# + editable=true slideshow={"slide_type": "skip"}


# + editable=true slideshow={"slide_type": "subslide"}
eng_bible_lemmas_types = Counter(lemmatize(eng_bible_words, lang="en"))
len(eng_bible_lemmas_types)

# + editable=true slideshow={"slide_type": "fragment"}
write_plain_text_corpus(eng_bible_plain_text, CORPORA_PATH + "eng-bible")

# + editable=true slideshow={"slide_type": "fragment"}
# !subword-nmt apply-bpe -c models/tokenization/gutenberg_low.model < corpora/tokenization/eng-bible.txt > corpora/tokenization/eng-bible-tokenized.txt

# + editable=true slideshow={"slide_type": "fragment"}
with open(CORPORA_PATH + "eng-bible-tokenized.txt", 'r') as f:
    tokenized_data = f.read()
eng_bible_tokenized = tokenized_data.split()

# + editable=true slideshow={"slide_type": "fragment"}
print(eng_bible_tokenized[:10])

# + editable=true slideshow={"slide_type": "fragment"}
len(eng_bible_tokenized)

# + editable=true slideshow={"slide_type": "fragment"}
eng_bible_tokenized_types = Counter(eng_bible_tokenized)
len(eng_bible_tokenized_types)

# + editable=true slideshow={"slide_type": "fragment"}
eng_bible_tokenized_types.most_common(30)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### ¿Qué pasa si aplicamos el modelo aprendido con Gutenberg a otras lenguas?

# + editable=true executionInfo={"elapsed": 426, "status": "ok", "timestamp": 1696893222188, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="HnYGl4NtNueJ" slideshow={"slide_type": "fragment"}
spa_bible_plain_text = get_bible_corpus('spa')
spa_bible_words = spa_bible_plain_text.replace("\n", " ").lower().split()

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 287, "status": "ok", "timestamp": 1696893238640, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="brmETSDiPEkV" outputId="0e4a8f0c-c848-4daf-e8fb-831eb4462cfd" slideshow={"slide_type": "fragment"}
spa_bible_words[:10]

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 239, "status": "ok", "timestamp": 1696893250921, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="FdgN5Br7P5-X" outputId="4fe9abfe-67db-40a3-dc09-debc6ab4c24b" slideshow={"slide_type": "fragment"}
len(spa_bible_words)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 331, "status": "ok", "timestamp": 1696893259357, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="otFwhvKCQA7p" outputId="db251a0a-7cbb-44b8-af93-480f0361810b" slideshow={"slide_type": "fragment"}
spa_bible_types = Counter(spa_bible_words)
len(spa_bible_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 198, "status": "ok", "timestamp": 1696893272455, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ynLly3NoQKES" outputId="0e9ac78c-e465-425a-9a79-d3767c033be0" slideshow={"slide_type": "fragment"}
spa_bible_types.most_common(30)

# + editable=true slideshow={"slide_type": "skip"}


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 5318, "status": "ok", "timestamp": 1696893302964, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="qLLWH8vmul4d" outputId="ef6995f8-703c-4a84-d16c-8b182ba0a2ee" slideshow={"slide_type": "subslide"}
spa_bible_lemmas_types = Counter(lemmatize(spa_bible_words, lang="es"))
len(spa_bible_lemmas_types)

# + editable=true executionInfo={"elapsed": 191, "status": "ok", "timestamp": 1696893334265, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="o3x3DZ8zr8uc" slideshow={"slide_type": "fragment"}
write_plain_text_corpus(spa_bible_plain_text, CORPORA_PATH + "spa-bible")

# + editable=true executionInfo={"elapsed": 557, "status": "ok", "timestamp": 1696893370878, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="i8pEgnudKvWQ" slideshow={"slide_type": "fragment"}
# !subword-nmt apply-bpe -c models/tokenization/gutenberg_high.model < corpora/tokenization/spa-bible.txt > corpora/tokenization/spa-bible-tokenized.txt

# + editable=true executionInfo={"elapsed": 321, "status": "ok", "timestamp": 1696893376561, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="mbS7LM26LMBA" slideshow={"slide_type": "fragment"}
with open(CORPORA_PATH + "spa-bible-tokenized.txt", "r") as f:
    tokenized_text = f.read()
spa_bible_tokenized = tokenized_text.split()

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 217, "status": "ok", "timestamp": 1696893379500, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="hB60BSTYMOD-" outputId="6d4d4028-5855-4e97-d6bc-bcd5f70eb41d" slideshow={"slide_type": "fragment"}
spa_bible_tokenized[:10]

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 226, "status": "ok", "timestamp": 1696893408605, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="bqMFr9h3L0Km" outputId="ac3ddfe9-26da-409b-e07d-597fb0f09d22" slideshow={"slide_type": "fragment"}
len(spa_bible_tokenized)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 249, "status": "ok", "timestamp": 1696893417347, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="lrXsWTwXMLoD" outputId="178b48e7-a346-4b95-9b16-4a8c90a39513" slideshow={"slide_type": "fragment"}
spa_bible_tokenized_types = Counter(spa_bible_tokenized)
len(spa_bible_tokenized_types)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 210, "status": "ok", "timestamp": 1696893508094, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="tmF_dkiDMeaA" outputId="72221965-f2e1-4d7a-b8a0-b0f3207ef53c" slideshow={"slide_type": "fragment"}
spa_bible_tokenized_types.most_common(40)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="pxkw79ldAnrP" slideshow={"slide_type": "subslide"}
# ### Type-token Ratio (TTR)
#
# - Una forma de medir la variazión del vocabulario en un corpus
# - Este se calcula como $TTR = \frac{len(types)}{len(tokens)}$
# - Puede ser útil para monitorear la variación lexica de un texto

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 242, "status": "ok", "timestamp": 1696893598928, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="b5ey_DGoAj9-" outputId="4358fc71-b419-4b6e-cfb5-98a05fa4e829" slideshow={"slide_type": "subslide"}
print("Información de la biblia en Inglés")
print("Tokens:", len(eng_bible_words))
print("Types (word-base):", len(eng_bible_types))
print("Types (lemmatized)", len(eng_bible_lemmas_types))
print("Types (BPE):", len(eng_bible_tokenized_types))
print("TTR (word-base):", len(eng_bible_types)/len(eng_bible_words))
print("TTR (BPE):", len(eng_bible_tokenized_types)/len(eng_bible_tokenized))

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 242, "status": "ok", "timestamp": 1696893598928, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="b5ey_DGoAj9-" outputId="4358fc71-b419-4b6e-cfb5-98a05fa4e829" slideshow={"slide_type": "fragment"}
print("Bible Spanish Information")
print("Tokens:", len(spa_bible_words))
print("Types (word-base):", len(spa_bible_types))
print("Types (lemmatized)", len(spa_bible_lemmas_types))
print("Types (BPE):", len(spa_bible_tokenized_types))
print("TTR (word-base):", len(spa_bible_types)/len(spa_bible_words))
print("TTR (BPE):", len(spa_bible_tokenized_types)/len(spa_bible_tokenized))

# + [markdown] editable=true id="kHfNlbERPfSQ" slideshow={"slide_type": "subslide"}
# ## Entrenando BPE con corpus en Nahuatl

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 5640, "status": "ok", "timestamp": 1696893718718, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="zxCDOFXGU2v7" outputId="11cb3ce9-fa83-4026-92c0-561ee5781867" slideshow={"slide_type": "fragment"}
# !pip install elotl

# + editable=true executionInfo={"elapsed": 260, "status": "ok", "timestamp": 1696893722506, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="GAG2Psj3TYTV" slideshow={"slide_type": "fragment"}
import elotl.corpus
axolotl = elotl.corpus.load("axolotl")

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 234, "status": "ok", "timestamp": 1696893727132, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ySdf892GPn76" outputId="4ebcc481-2bf7-451b-aaac-f6ceb013122a" slideshow={"slide_type": "fragment"}
len(axolotl)

# + editable=true executionInfo={"elapsed": 171, "status": "ok", "timestamp": 1696893752688, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="xIAqGcxXPqT-" slideshow={"slide_type": "fragment"}
train_rows_count = len(axolotl) - round(len(axolotl)*.30)

# + editable=true executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1696893753055, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="YWaDYGr2PiVB" slideshow={"slide_type": "fragment"}
axolotl_train = axolotl[:train_rows_count]
axolotl_test = axolotl[train_rows_count:]
# -

axolotl_train[3]

# + editable=true slideshow={"slide_type": "skip"}


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 225, "status": "ok", "timestamp": 1696893763780, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="dr8YKgz5QGDl" outputId="74863977-c160-465d-af89-928f4471f0e4" slideshow={"slide_type": "subslide"}
print("Axolotl train len:", len(axolotl_train))
print("Axolotl test len:", len(axolotl_test))
print("Total:", len(axolotl_test) + len(axolotl_train))

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 296, "status": "ok", "timestamp": 1696893780461, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="EDj2ufhZQftm" outputId="d7f522dc-c86d-45c2-c8c7-34726de5dd51" slideshow={"slide_type": "fragment"}
axolotl_train[:3]

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 228, "status": "ok", "timestamp": 1696893816717, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="vRX0WIdpJHb7" outputId="9e4bebe9-e7a5-453b-845b-b0fbf8d39266" slideshow={"slide_type": "fragment"}
axolotl_words_train = [word for row in axolotl_train for word in row[1].lower().split()]
len(axolotl_words_train)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1696893835790, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="zsqd8CtnwOms" outputId="f4e0d4dc-98f7-40f1-d514-886d62bf6103" slideshow={"slide_type": "fragment"}
print(axolotl_words_train[:10])

# + editable=true executionInfo={"elapsed": 186, "status": "ok", "timestamp": 1696893853253, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="oHTjFtMuOEsx" slideshow={"slide_type": "fragment"}
write_plain_text_corpus(" ".join(axolotl_words_train), CORPORA_PATH + "axolotl_plain")

# + editable=true slideshow={"slide_type": "skip"}


# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 6148, "status": "ok", "timestamp": 1696893898840, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="wexqYXwmQwae" outputId="8ef4a333-57e5-4b78-f287-3ee1ff0822e4" slideshow={"slide_type": "subslide"}
# !subword-nmt learn-bpe -s 500 < corpora/tokenization/axolotl_plain.txt > models/tokenization/axolotl.model

# + editable=true executionInfo={"elapsed": 227, "status": "ok", "timestamp": 1696894044780, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="z-5naiV9JSki" slideshow={"slide_type": "fragment"}
axolotl_test_words = [word for row in axolotl_test for word in row[1].lower().split()]
axolotl_test_types = Counter(axolotl_test_words)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 213, "status": "ok", "timestamp": 1696894048497, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="KbUVcLZtJiHh" outputId="9a403fcb-903d-4f46-b43b-4dab375869a1" slideshow={"slide_type": "fragment"}
print(axolotl_test_types.most_common(10))

# + editable=true executionInfo={"elapsed": 161, "status": "ok", "timestamp": 1696894053220, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="vJ2KxiSQJYXx" slideshow={"slide_type": "fragment"}
axolotl_singletons = [singleton for singleton in axolotl_test_types.items() if singleton[1] == 1]

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 228, "status": "ok", "timestamp": 1696894067746, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="UoKarH04JnYI" outputId="96b3c1ab-0cde-4875-a633-a2a57abe9250" slideshow={"slide_type": "fragment"}
len(axolotl_singletons)

# + editable=true executionInfo={"elapsed": 316, "status": "ok", "timestamp": 1696894075624, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="rffWAyk1SAYD" slideshow={"slide_type": "fragment"}
write_plain_text_corpus(" ".join(axolotl_test_words), CORPORA_PATH + "axolotl_plain_test")

# + editable=true slideshow={"slide_type": "skip"}


# + editable=true executionInfo={"elapsed": 1332, "status": "ok", "timestamp": 1696894084258, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="GOC48EXvSbrJ" slideshow={"slide_type": "subslide"}
# !subword-nmt apply-bpe -c models/tokenization/axolotl.model < corpora/tokenization/axolotl_plain_test.txt > corpora/tokenization/axolotl_tokenized.txt

# + editable=true executionInfo={"elapsed": 229, "status": "ok", "timestamp": 1696894086299, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="_yOkxKJTSrMq" slideshow={"slide_type": "fragment"}
with open(CORPORA_PATH + "axolotl_tokenized.txt") as f:
    axolotl_test_tokenized = f.read().split()

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1696894087376, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="zxy1oY1AS4iB" outputId="6b6eedcd-d9df-4db8-e480-824d2ca7a9ee" slideshow={"slide_type": "fragment"}
len(axolotl_test_tokenized)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 304, "status": "ok", "timestamp": 1696894111214, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="5USR33oDxSdj" outputId="a59c88dc-1ba4-4513-dff2-596cf7086ced" slideshow={"slide_type": "fragment"}
print(axolotl_test_tokenized[:10])

# + editable=true slideshow={"slide_type": "skip"}


# + editable=true executionInfo={"elapsed": 324, "status": "ok", "timestamp": 1696894192432, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="uWilKtKtTFZM" slideshow={"slide_type": "subslide"}
axolotl_test_tokenized_types = Counter(axolotl_test_tokenized)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 254, "status": "ok", "timestamp": 1696894194136, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="7DaOOlwrUtm7" outputId="5c97aec4-d34f-45cd-c9bc-369b8e7cb937" slideshow={"slide_type": "fragment"}
axolotl_test_tokenized_types.most_common(20)

# + colab={"base_uri": "https://localhost:8080/"} editable=true executionInfo={"elapsed": 283, "status": "ok", "timestamp": 1696894219043, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="yUMzlKCQR73z" outputId="3576605a-a856-4250-ceed-c5a58962fc36" slideshow={"slide_type": "subslide"}
print("Axolotl Information")
print("Tokens:", len(axolotl_test_words))
print("Types (word-base):", len(axolotl_test_types))
print("Types (native BPE):", len(axolotl_test_tokenized_types))
print("TTR (word-base):", len(axolotl_test_types)/len(axolotl_test_words))
print("TTR (BPE):", len(spa_bible_tokenized_types)/len(axolotl_test_tokenized))

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="CUG2cVS6DcWL" slideshow={"slide_type": "subslide"}
# ## Normalización

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# <center><img src="img/metro.jpg" width=700 height=700></center>

# + editable=false executionInfo={"elapsed": 263, "status": "ok", "timestamp": 1696894472827, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="UufnThu87JnI" slideshow={"slide_type": "subslide"}
METROFLOG_SENTENCE = "lEt'$ dó tHis béttëŕ :)"

# + colab={"base_uri": "https://localhost:8080/", "height": 463, "referenced_widgets": ["fb12e1ecafa840638ea24dd751097bdd", "5d6221bc224a496b86c50e5803caf346", "25ef9cfaab304df2b69c1510df8e325f", "8dc2d514431449dfac25d20e0e48e9ae", "cf9c14180be04157a44685590a4972e6", "8a7d115132984d54aafc10f116fe8501", "9dc9fee8ec094894b5a63ab99e8d7666", "4d5b3303c5bd419a9925b7bf37a53242", "a82daabdfbbe4ede8fc6fc32e072997b", "ae632eaecce341b8a295de0db8edfaf8", "ae71f7cd385f450587d2062f86a513f2", "4dbfe9144c2542a69184c5d8b26ad395", "22055c67e0654aa2be5126d0f97f5531", "3abde4238f0d4faaad9a51f7694a2d92", "6d8f6d73c769484fa3d1e714e4879707", "137091a4be06419aad255f17030305ef", "c5f62627cb8b401289ee81755e4f7a32", "24ab8acc5ccc42b9a651a7c93d591673", "111b51b9b2e04bf69f16702a52f322f7", "308c1e4f6c2646979bf69926ea4d7363", "8c5517aa266046f993b9c3dc0f65e23d", "199616ac47724f02948bc16bae127809"]} editable=true executionInfo={"elapsed": 1167, "status": "ok", "timestamp": 1696894488236, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="WaCWiKc-7r4Y" outputId="04f91b20-2be5-4369-879d-e3785aa06ac0" slideshow={"slide_type": "fragment"}
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggingface-course/albert-tokenizer-without-normalizer")
tokenizer.convert_ids_to_tokens(tokenizer.encode(METROFLOG_SENTENCE))

# + colab={"base_uri": "https://localhost:8080/", "height": 130, "referenced_widgets": ["3a14af5f7a2944188f5d3d3db323a3ae", "de57c2cccbd34b20af5a2bbe660b4d27", "43b004bbe6804ef5871c7d77e0693986", "117c49ebe8f64f35bba236f6aab5eac3", "a83a0289475747dc9036b11c5d1565e9", "3a71cb07b41f43588fb9b50f550103a4", "5f5dea8822f84844a52ad0add5d0f72f", "7640f56ff48f4768bceff82351121475", "5044856cb39344b582f765c8725341fc", "d0a1b8e69c494c31a207e44040ddfff8", "df6b8a992b9f4e0dae69b8918275819d", "b251ce4bc1874c909379cdbb511216de", "3d9b5101b6694b6f9544a2e26caaa790", "28ec45d45c994f2e810d63d45f72a3cc", "442bd795e70940c48ef700d034b2936c", "786736d601cf49c7b1dbcecc44aa6938", "3ff9ae7241994d5d8f14707aa8e85104", "42c7b6f18b9542c684869b1917228578", "17546e369c3e4004ad141710b9c148c0", "6d6071e55fe3402ca422fc0b57194226", "8d941a53874349f084dd6321cdfb27f4", "212b11204ea8441ba617e4b262239a6a", "c9b6ecf2896140768af9e7d0c754ac36", "9bcca4e6cb9f4937934fb959163269e6", "959fee11ac3149bea4a63fc2c9e99d86", "95536809435f4ce3a94140960051fb7a", "9c93514274014754b64ff5da7184b8c1", "534d6ac48d1540b198ee9978025996d0", "21fbf522fd9a43ada4e33283d223b5d1", "bae6b9331c6042f59c44109eee896e46", "8bcceca86aff4e3484b8c3fc228ed65b", "1ecc97cf470a40f2a3c00bb70d78e268", "457370d233c84f0eb97922f8fe4d6848"]} editable=true executionInfo={"elapsed": 909, "status": "ok", "timestamp": 1696894544364, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="tuZoKC8B8gA1" outputId="fe8a3cd8-5b94-41eb-9f2f-a4e0b2504e2f" slideshow={"slide_type": "fragment"}
tokenizer = AutoTokenizer.from_pretrained("albert-large-v2")
tokenizer.convert_ids_to_tokens(tokenizer.encode(METROFLOG_SENTENCE))

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="fVTN4kut897t" slideshow={"slide_type": "subslide"}
# #### Y para lenguas de bajos recursos digitales?

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - No hay muchos recursos :(
# - Pero para el nahuatl esta `pyelotl` :)

# + [markdown] editable=true id="aNWBVF4c9N3j" slideshow={"slide_type": "subslide"}
# #### Normalizando el Nahuatl

# + editable=true executionInfo={"elapsed": 203, "status": "ok", "timestamp": 1696894643942, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="26GQkUkX9QvB" slideshow={"slide_type": "fragment"}
import elotl.nahuatl.orthography

# + editable=true executionInfo={"elapsed": 198, "status": "ok", "timestamp": 1696894681527, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="thlTYQUqGwRE" slideshow={"slide_type": "fragment"}
# Tres posibles normalizadores: sep, inali, ack
# Sauce: https://pypi.org/project/elotl/

nahuatl_normalizer = elotl.nahuatl.orthography.Normalizer("sep")

# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true executionInfo={"elapsed": 278, "status": "ok", "timestamp": 1696894695041, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="ZLZX7VeNHDxn" outputId="15b780e4-85d9-45a7-9c30-8523fe923228" slideshow={"slide_type": "fragment"}
axolotl[1][1]

# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true executionInfo={"elapsed": 400, "status": "ok", "timestamp": 1696894706726, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="nkC--od4HLao" outputId="cca96afe-8d5c-4828-a0be-ff8aa289e6cc" slideshow={"slide_type": "fragment"}
nahuatl_normalizer.normalize(axolotl[1][1])

# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1696894723197, "user": {"displayName": "Diego Alberto Barriga Mart\u00ednez", "userId": "06235177150913802056"}, "user_tz": 360} id="mIXqhX_uHWJ8" outputId="70719725-e8de-4b07-d3e1-fd4a0e6e5451" slideshow={"slide_type": "fragment"}
nahuatl_normalizer.to_phones(axolotl[1][1])

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ## Entropía de un texto
#
# <center><img src="img/entropy.gif" height=500 width=500></center>

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# <center><img src="img/entropy_eq.png"></center>

# + editable=true slideshow={"slide_type": "fragment"}
import math

def calculate_entropy(corpus: list[str]) -> float:
    words_counts = Counter(corpus)
    total_words = len(corpus)
    probabilities = {word: count / total_words for word, count in words_counts.items()}
    entropy = -sum(p * math.log2(p) for p in probabilities.values())
    return entropy


# + editable=true slideshow={"slide_type": "fragment"}
calculate_entropy(eng_bible_words)

# + editable=true slideshow={"slide_type": "fragment"}
calculate_entropy(eng_bible_tokenized)

# + editable=true slideshow={"slide_type": "skip"}


# + [markdown] editable=true id="S-qcqI8b2m5u" slideshow={"slide_type": "slide"}
# ## Práctica 4: Subword tokenization
# **Fecha de entrega: 24 de Marzo 11:59pm**

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Calcular la entropía de dos textos: brown y axolotl
#     - Calcular para los textos tokenizados word-level
#     - Calcular para los textos tokenizados con BPE
#         - Tokenizar con la biblioteca `subword-nmt`
# - Imprimir en pantalla:
#     - Entropía de axolotl word-base y bpe
#     - Entropía del brown word-base y bpe
# - Responder las preguntas:
#     - ¿Aumento o disminuyó la entropia para los corpus?
#         - axolotl 
#         - brown
#     - ¿Qué significa que la entropia aumente o disminuya en un texto?
#     - ¿Como influye la tokenizacion en la entropía de un texto?

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Extra

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Realizar el proceso de normalización para el texto en Nahuatl
# - Entrenar un modelo con el texto normalizado
#     - Usando BPE `subword-nmt`
# - Comparar entropía, typos, tokens, TTR con las versiones:
#     - tokenizado sin normalizar
#     - tokenizado normalizado

# + [markdown] editable=true id="1qLmH9YN2skH" slideshow={"slide_type": "subslide"}
# ### Referencias:
#
# - [Corpora de la biblia en varios idiomas](https://github.com/ximenina/theturningpoint/tree/main/Detailed/corpora/corpusPBC)
# - [Biblioteca nativa para BPE](https://github.com/rsennrich/subword-nmt)
# - [Tokenizers Hugging face](https://huggingface.co/docs/transformers/tokenizer_summary)

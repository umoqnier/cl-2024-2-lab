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

# + [markdown] editable=true id="a1d17ddb-81e1-492f-a0a5-5a55d94f7e4c" slideshow={"slide_type": "subslide"}
# # 1. Niveles lingüísticos I
#
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/IPA_chart_2020.svg/660px-IPA_chart_2020.svg.png"></center>

# + [markdown] editable=true id="7568912a-5945-46c0-aeb4-6106e8f36635" slideshow={"slide_type": "subslide"}
# ## Objetivo

# + [markdown] editable=true id="1db19ad2-dbd9-4a56-843a-1133e440641e" slideshow={"slide_type": "fragment"}
# - Læs alumnæs entenderán que es la fonología y un alfabeto fonético
# - Manipularan y recuperará información de datasets disponibles en repositorios de Github para resolver una tarea específica 
# - Læs alumnæs tendrán un acercamiento a la tarea de análisis morfológico
# - Hacer una comparación entre un enfoque basado en reglas y uno estadístico para tareas de NLP

# + [markdown] editable=true id="54563b4d-3c1a-4a23-9716-73554e263fb0" slideshow={"slide_type": "subslide"}
# ## ¿Qué es la fonología?

# + [markdown] editable=true id="3fb13d94-e6af-44dc-aa94-608b99feafff" slideshow={"slide_type": "fragment"}
# - La fonología es una rama de la Lingüística que estudia como las lenguajes sistematicamente organizan los fonemas
# - Estudia como los humanos producimos y percibimos el lenguaje
#     - Producción: La forma en que producimos el lenguaje
#     - Percepción: La forma en que interpretamos el lenguaje
#
# > Wikipedia contributors. Phonology. In Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/w/index.php?title=Phonology&oldid=1206207687

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ## ¿Qué es la fonética?

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - El estudio de los sonidos físicos del discurso humano. Es la rama de la lingüística que estudia la producción y percepción de los sonidos de una lengua con respecto a sus manifestaciones físicas.
#
# > Fonética. Wikipedia, La enciclopedia libre. https://es.wikipedia.org/w/index.php?title=Fon%C3%A9tica&oldid=155764166. 

# + colab={"base_uri": "https://localhost:8080/", "height": 439} editable=true id="e529deca-fb69-4072-91ed-7a5063443d62" outputId="c91b7852-3add-4a5c-f32f-94d5a70864b5" slideshow={"slide_type": "subslide"}
# %%HTML
<center><iframe width='900' height='600' src='https://www.youtube.com/embed/DcNMCB-Gsn8?controls=1'></iframe></center>

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Formas comunes

# + [markdown] editable=true id="ae09ec8d-b9e9-4e66-9878-63c514fb57f0" slideshow={"slide_type": "fragment"}
# - Oral-Aural
#     - Producción: La boca
#     - Percepción: Oidos

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Manual-visual
#     - Producción: Manual usando las manos
#     - Percepción: Visual

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Manual-Manual
#     - Producción: Manual usando las manos
#     - Percepción: Manual usando las manos

# + [markdown] editable=true id="029a8e17-03ed-417c-b2f3-512fd1ee5fd9" slideshow={"slide_type": "subslide"}
# #### International Phonetic Alphabet (IPA)

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Las lenguas naturales tienen muchos sonidos diferentes por lo que necesitamos una forma de describirlos independientemente de las lenguas
# - Por ejemplo: Los sonidos del habla se determinan por los movimientos de la boca necesarios para producirlos
# - Las dos grandes categorías: Consonantes y Vocales
# - IPA es una representación escrita de los [sonidos](https://www.ipachart.com/) del [habla](http://ipa-reader.xyz/)

# + [markdown] editable=true id="COzTRH3QXdWl" slideshow={"slide_type": "slide"}
# ## Dataset: IPA-dict de open-dict
#
# - Diccionario de palabras para varios idiomas con su representación fonética
# - Representación simple, una palabra por renglon con el formato:

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ```
# [PALABRA][TAB][IPA]
#
# Ejemplos
# mariguana	/maɾiɣwana/
# zyuganov's   /ˈzjuɡɑnɑvz/, /ˈzuɡɑnɑvz/
# ```

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - [Github repo](https://github.com/open-dict-data/ipa-dict)
#   - [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
#   - URL: `https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/<iso-lang>`

# + [markdown] editable=true id="c-Q91_zR859L" slideshow={"slide_type": "subslide"}
# ### Explorando el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 121} editable=true id="dfCkH58988vq" outputId="10dd9d94-cb8a-4ba1-ef88-5bb2214c94d2" slideshow={"slide_type": "fragment"}
# Explorando el corpus
import requests as r

response = r.get("https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt")
response.text[:100]

# + editable=true slideshow={"slide_type": "fragment"}
from pprint import pprint as pp
ipa_data = response.text.split("\n")
#print(ipa_data[-4:])
ipa_data[-1]
pp(ipa_data[400:410])

# + editable=true slideshow={"slide_type": "subslide"}
# Puede haber mas de una transcipcion asociada a una palabra
print(ipa_data[-3].split("\t"))
for data in ipa_data[300:500]:
    word, ipa = data.split('\t')
    representations = ipa.split(", ")
    if len(representations) >= 2:
        print(f"{word} --> {representations}")


# + [markdown] editable=true id="cJMkPF06jJJp" slideshow={"slide_type": "subslide"}
# ### Obteniendo el corpus

# + editable=true slideshow={"slide_type": "subslide"}
def response_to_dict(ipa_list: list) -> dict:
    """Parse to dict the list of word-IPA

    Each element of text have the format:
    [WORD][TAB][IPA]

    Parameters
    ----------
    ipa_list: list
        List with each row of ipa-dict raw dataset file

    Returns
    -------
    dict:
        A dictionary with the word as key and the phonetic
        representation as value
    """
    result = {}
    for item in ipa_list:
       item_list = item.split("\t")
       result[item_list[0]] = item_list[1]
    return result


# + editable=true slideshow={"slide_type": "fragment"}
response_to_dict(ipa_data[:100])["ababa"]


# + editable=true slideshow={"slide_type": "subslide"}
def get_ipa_dict(iso_lang: str) -> dict:
    """Get ipa-dict file from Github

    Parameters:
    -----------
    iso_lang:
        Language as iso code

    Results:
    --------
    dict:
        Dictionary with words as keys and phonetic representation
        as values for a given lang code
    """
    print(f"Downloading {iso_lang}", end=" ")
    response = r.get(f"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{iso_lang}.txt") 
    raw_data = response.text.split("\n")
    print(f"status:{response.status_code}")
    return response_to_dict(raw_data[:-1])


# + editable=true slideshow={"slide_type": "fragment"}
es_mx_ipa = get_ipa_dict("es_MX")


# + editable=true id="3vfeGyqYkI9V" slideshow={"slide_type": "subslide"}
def query_ipa_transcriptions(word: str, dataset: dict) -> list[str]:
    """Search for a word in an IPA phonetics dict
 
    Given a word this function return the IPA transcriptions

    Parameters:
    -----------
    word: str
        A word to search in the dataset
    dataset: dict
        A dataset for a given language code
    
    Returns
    -------
    list[str]:
        List with posible transcriptions if any, 
        else a list with the string "NOT FOUND" 
    """
    return dataset.get(word.lower(), "NOT FOUND").split(", ")


# + editable=true slideshow={"slide_type": "fragment"}
query_ipa_transcriptions("mayonesa", es_mx_ipa)

# + [markdown] editable=true id="h9Ri8YmwMnxR" slideshow={"slide_type": "subslide"}
# #### Obtengamos un par de datasets

# + editable=true id="SDspkhcdLmtx" slideshow={"slide_type": "fragment"}
# Get datasets
dataset_mx = get_ipa_dict("es_MX")
dataset_us = get_ipa_dict("en_US")

# + editable=true slideshow={"slide_type": "fragment"}
# Simple query
query_ipa_transcriptions("beautiful", dataset_us)

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="abpslzJRMvD6" outputId="113ba799-8003-4109-c3d0-ac320ac1064f" slideshow={"slide_type": "fragment"}
# Examples
print(f"dog -> {query_ipa_transcriptions('dog', dataset_us)}🐶")
print(f"mariguana -> {query_ipa_transcriptions('mariguana', dataset_mx)} 🪴")

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# #### Diferentes formas de pronunciar dependiendo la lengua, aunque la ortografía se parezca. Ejemplo:

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="drw73avU9-ct" outputId="44caf0d9-b0c2-4a58-9805-3a623354cf2c" slideshow={"slide_type": "fragment"}
# Ilustrative example
print("[es_MX] hotel |", query_ipa_transcriptions("hotel", dataset_mx))
print("[en_US] hotel |", query_ipa_transcriptions("hotel", dataset_us))

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Obteniendo corpora desde GitHub

# + editable=true id="YSRb9cx5jM8d" slideshow={"slide_type": "fragment"}
lang_codes = {
  "ar": "Arabic (Modern Standard)",
  "de": "German",
  "en_UK": "English (Received Pronunciation)",
  "en_US": "English (General American)",
  "eo": "Esperanto",
  "es_ES": "Spanish (Spain)",
  "es_MX": "Spanish (Mexico)",
  "fa": "Persian",
  "fi": "Finnish",
  "fr_FR": "French (France)",
  "fr_QC": "French (Québec)",
  "is": "Icelandic",
  "ja": "Japanese",
  "jam": "Jamaican Creole",
  "km": "Khmer",
  "ko": "Korean",
  "ma": "Malay (Malaysian and Indonesian)",
  "nb": "Norwegian Bokmål",
  "nl": "Dutch",
  "or": "Odia",
  "ro": "Romanian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tts": "Isan",
  "vi_C": "Vietnamese (Central)",
  "vi_N": "Vietnamese (Northern)",
  "vi_S": "Vietnamese (Southern)",
  "yue": "Cantonese",
  "zh": "Mandarin"
}
iso_lang_codes = list(lang_codes.keys())


# + editable=true id="WcCmgrgnT9wK" slideshow={"slide_type": "subslide"}
def get_dataset() -> dict:
    """Download corpora from ipa-dict github

    Given a list of iso lang codes download available datasets.

    Returns
    -------
    dict
        Lang codes as keys and dictionary with words-transcriptions
        as values
    """
    return {code: get_ipa_dict(code) for code in iso_lang_codes}

dataset = get_dataset()

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Creando aplicaciones con estos datos

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### 1. Busquedas básicas automatizada
# Buscador de representaciones foneticas de palabras automatizado en diferentes idiomas

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="f5b7fbc2-4b95-4233-9fa6-17493bb2afb3" outputId="97f5e5f8-216f-4f5f-8368-e0ab4fd015cb" slideshow={"slide_type": "fragment"}
print("Representación fonética de palabras")

print("Lenguas disponibles:")
for lang_key in dataset.keys():
    print(f"{lang_key}: {lang_codes[lang_key]}")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")
while lang:
    # El programa comeinza aqui
    sub_data = dataset[lang]
    query = input(f"[{lang}] word>> ")
    results = query_ipa_transcriptions(query, sub_data)
    print(query, " | ", results)
    while query:
        query = input(f"[{lang}] word>> ")
        if query:
            results = query_ipa_transcriptions(query, sub_data)
            print(query, " | ", results)
    lang = input("lang>> ")

# + [markdown] editable=true id="8TLGghJWFbIZ" slideshow={"slide_type": "subslide"}
# ### 2. Encontrando palabras que tengan terminación similar
#
# Dada una oración agrupar las palabras que tengan una pronunciación similar

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="B1JaTm-lUy3c" outputId="bd91020c-b52b-430b-f3bc-23b5718c57e9" slideshow={"slide_type": "fragment"}
from collections import defaultdict

#sentence = "There once was a cat that ate a rat and after that sat on a yellow mat"
#sentence = "the cat sat on the mat and looked at the rat."
#sentence = "If you drop the ball it will fall on the doll"
sentence = "cuando juego con fuego siento como brilla la orilla de mi corazón"

#lang = "en_US"
lang = "es_MX"
words = sentence.split(" ")

# Get words and IPA transciptions map
word_ipa_map = {}
for word in words:
    ipa_transcriptions = query_ipa_transcriptions(word.lower(), dataset.get(lang))
    ipa_transcriptions = [_.strip("/") for _ in ipa_transcriptions]
    word_ipa_map.update({word.lower(): ipa_transcriptions})

patterns = defaultdict(list)
for word, ipa_list in word_ipa_map.items():
    for ipa in ipa_list:
        ipa_pattern = ipa[-2:]
        patterns[ipa_pattern].append(word)

for pattern, words in patterns.items():
    if len(set(words)) > 1:
        print(f"{pattern}:: {words}")

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Morfología y análisis morfológico

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### ¿Qué es la morfología?

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# La morfología es uno de los niveles de la lengua que estudia los procesos que conforman una palabra.
#
# > Morfología es el estudio de la estructura interna de las palabras (Bauer, 2003)
#
# - niñ-o
# - niñ-a
# - niñ-o-s
# - gat-a-

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Morfemas

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Con la morfología podemos identificar como se modifica el significado variando la estructura de las palabras
# - Tambien las reglas para producir:
#     - niño -> niños
#     - niño -> niña
# - Tenemos elementos mínimos, intercambiables que varian el significado de las palabras: **morfemas**
#
# > Un morfema es la unidad mínima con significado en la producción lingüística (Mijangos, 2020)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Tipos de morfemas

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - Bases: Subcadenas que aportan información léxica de la palabra
#     - sol
#     - frasada
# - Afijos: Subcadenas que se adhieren a las bases para añadir información (flexiva, derivativa)
#     - Prefijos
#         - *in*-parable
#     - Subfijos
#         - pan-*ecitos*, come-*mos*

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Aplicaciones relacionadas a la morfología en NLP

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# #### Análisis morfológico

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# La morfología es uno de los niveles más básicos del lenguaje que se puede estudiar. En ese sentido, una de las tareas más básicas del NLP es el análisis morfológico:
#
# > El análisis morfológico es la determinación de las partes que componen la palabra y su representación lingüística, es una especie de etiquetado
#
# Los elementos morfológicos son analizados para:
#
# - Determinar la función morfológica de las palabras
# - Hacer filtrado y pre-procesamiento de texto

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Ejemplo: Parsing con expresiones regulares
#
# La estructura del sustantivo en español es:
#
# ` BASE+AFIJOS (marcas flexivas)   --> Base+DIM+GEN+NUM`

# + editable=true slideshow={"slide_type": "fragment"}
palabras = [
    'niño',
    'niños',
    'niñas',
    'niñitos',
    'gato',
    'gatos',
    'gatitos',
    'perritos',
    'paloma',
    'palomita',
    'palomas',
    'flores',
    'flor',
    'florecita',
    'lápiz',
    'lápices',
    #'chiquitititititos',
    #'curriculum', # curricula
    #'campus', # campi
]

# + editable=true slideshow={"slide_type": "subslide"}
import re

def morph_parser_rules(words: list[str]) -> list[str]:
    """Aplica reglas morfológicas a una lista de palabras para realizar
    un análisis morfológico.

    Parameters:
    ----------
    words : list of str
        Lista de palabras a las que se les aplicarán las reglas morfológicas.

    Returns:
    -------
    list of str
        Una lista de palabras después de aplicar las reglas morfológicas.
    """

    #Lista para guardar las palabras parseadas
    morph_parsing = []

    # Reglas que capturan ciertos morfemas
    # {ecita, itos, as, os}
    for w in words:
        #ecit -> DIM
        R0 = re.sub(r'([^ ]+)ecit([a|o|as|os])', r'\1-DIM\2', w)
        #it -> DIM
        R1 = re.sub(r'([^ ]+)it([a|o|as|os])', r'\1-DIM\2', R0)
        #a(s) -> FEM
        R2 = re.sub(r'([^ ]+)a(s)', r'\1-FEM\2', R1)
        #a -> FEM
        R3 = re.sub(r'([^ ]+)a\b', r'\1-FEM', R2)
        #o(s) -> MSC
        R4 = re.sub(r'([^ ]+)o(s)', r'\1-MSC\2', R3)
        #o .> MSC
        R5 = re.sub(r'([^ ]+)o\b', r'\1-MSC', R4)
        #es -> PL
        R6 = re.sub(r'([^ ]+)es\b', r'\1-PL', R5)
        #s -> PL
        R7 = re.sub(r'([^ ]+)s\b', r'\1-PL', R6)
        #Sustituye la c por z cuando es necesario
        parse = re.sub(r'c-', r'z-', R7)

        #Guarda los parseos
        morph_parsing.append(parse)
    return morph_parsing


# + editable=true slideshow={"slide_type": "subslide"}
morph_parsing = morph_parser_rules(palabras)
for palabra, parseo in zip(palabras, morph_parsing):
    print(palabra, "-->", parseo)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### Preguntas 🤔
# - ¿Qué pasa con las reglas en lenguas donde son más comunes los prefijos y no los sufijos?
# - ¿Cómo podríamos identificar características de lenguas

# + [markdown] editable=true id="pgmRkPhOgn1d" slideshow={"slide_type": "subslide"}
# ## Corpus: [SIGMORPHON 2022 Shared Task on Morpheme Segmentation](https://github.com/sigmorphon/2022SegmentationST/tree/main)

# + [markdown] editable=true id="j84yrvsZAnyt" slideshow={"slide_type": "fragment"}
# - Shared task donde se buscaba convertir las palabras en una secuencia de morfemas
#     - ¿Qué es un shared task?
# - Dividido en dos partes:
#     - Segmentación a nivel de palabras (nos enfocaremos en esta)
#

# + [markdown] editable=true id="MN-HDX6hHu4e" slideshow={"slide_type": "subslide"}
# ### Track: WORDS

# + [markdown] editable=true id="yYHYlHGuhqbY" slideshow={"slide_type": "fragment"}
# | word class | Description                      | English example (input ==> output)     |
# |------------|----------------------------------|----------------------------------------|
# | 100        | Inflection only                  | played ==> play @@ed                   |
# | 010        | Derivation only                  | player ==> play @@er                   |
# | 101        | Inflection and Compound          | wheelbands ==> wheel @@band @@s        |
# | 000        | Root words                       | progress ==> progress                  |
# | 011        | Derivation and Compound          | tankbuster ==> tank @@bust @@er        |
# | 110        | Inflection and Derivation        | urbanizes ==> urban @@ize @@s          |
# | 001        | Compound only                    | hotpot ==> hot @@pot                   |
# | 111        | Inflection, Derivation, Compound | trackworkers ==> track @@work @@er @@s |

# + [markdown] editable=true id="2y6V_yeYP3Fi" slideshow={"slide_type": "subslide"}
# ### Obteniendo el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 105} editable=true id="rxCKmWqFZXvF" outputId="a7efe1af-e0e0-4407-df07-d2a17798fc07" slideshow={"slide_type": "fragment"}
response = r.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/spa.word.test.gold.tsv")
response.text[:100]
# -

raw_data = response.text.split("\n")
raw_data[:10]

element = raw_data[0].split("\t")
element[1].split()

for row in raw_data[:10]:
    word, morphs, category = row.split("\t")
    print(word, morphs, category)
    print(morphs.split())

# + editable=true slideshow={"slide_type": "subslide"}
import pandas as pd

LANGS = {
    "ces": "Czech",
    "eng": "English",
    "fra": "French",
    "hun": "Hungarian",
    "spa": "Spanish",
    "ita": "Italian",
    "lat": "Latin",
    "rus": "Russian",
}
CATEGORIES = {
    "100": "Inflection",
    "010": "Derivation",
    "101": "Inflection, Compound",
    "000": "Root",
    "011": "Derivation, Compound",
    "110": "Inflection, Derivation",
    "001": "Compound",
    "111": "Inflection, Derivation, Compound"
}


# + editable=true id="I6nB9uAyBkmN" slideshow={"slide_type": "subslide"}
def get_files(lang: str, track: str = "word") -> list[str]:
    """Genera una lista de nombres de archivo basados en el idioma y el track

    Parameters:
    ----------
    lang : str
        Idioma para el cual se generarán los nombres de archivo.
    track : str, optional
        Track del shared task de donde vienen los datos (por defecto es "word").

    Returns:
    -------
    list[str]
        Una lista de nombres de archivo generados para el idioma y el track especificados.
    """
    base = f"{lang}.{track}"
    return [
        f"{base}.dev.tsv",
        #f"{base}.train.tsv",
        f"{base}.test.gold.tsv"
    ]


# + editable=true slideshow={"slide_type": "subslide"}
def get_raw_corpus(files: list) -> list:
    """Descarga y concatena los datos de los archivos tsv desde una URL base.

    Parameters:
    ----------
    files : list
        Lista de nombres de archivos (sin extensión) que se descargarán
        y concatenarán.

    Returns:
    -------
    list
        Una lista que contiene los contenidos descargados y concatenados
        de los archivos tsv.
    """
    result = []
    for file in files:
        print(f"Downloading {file}.tsv")
        response = r.get(f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}")
        response_list = response.text.split("\n")
        result.extend(response_list[:-1]) # Last element is empty string ''
    return result


# -

get_raw_corpus(get_files(lang="ita"))[:10]


# + editable=true slideshow={"slide_type": "subslide"}
def raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame:
    """Convierte una lista de datos de corpus en un DataFrame

    Parameters:
    ----------
    corpus_list : list
        Lista de líneas del corpus a convertir en DataFrame.
    lang : str
        Idioma al que pertenecen los datos del corpus.

    Returns:
    -------
    pd.DataFrame
        Un DataFrame de pandas que contiene los datos del corpus procesados.
    """
    data = []
    for row in corpus_list:
        try:
            word, morphs, category = row.split("\t")
        except ValueError:
            # Caso donde no hay categoria
            word, morphs = row.split("\t")
            category = "N/A"
        morphs = morphs.split()
        data.append({"words": word, "morphs": morphs, "category": category, "lang": lang})
    df = pd.DataFrame(data)
    df["word_len"] = df["words"].apply(lambda x: len(x))
    df["morphs_count"] = df["morphs"].apply(lambda x: len(x))
    return df


# + editable=true slideshow={"slide_type": "subslide"}
# Get data
files = get_files("spa")
raw_data = get_raw_corpus(files)
df = raw_corpus_to_dataframe(raw_data, lang="spa")
# -

df

# + [markdown] editable=true id="pCorWx_TIEOd" slideshow={"slide_type": "subslide"}
# ### Análisis cuantitativo para el Español
# -

df["category"].value_counts().head(30)

df["morphs_count"].mean()

df["word_len"].mean()

# + colab={"base_uri": "https://localhost:8080/", "height": 472} editable=true id="fWl5tJAfsR9K" outputId="4300e452-1c53-4e18-c3f1-0e7b06f82e1f" slideshow={"slide_type": "subslide"}
import matplotlib.pyplot as plt
plt.hist(df["word_len"], bins=10, edgecolor="black")
plt.xlabel("Word len")
plt.ylabel("Freq")
plt.show()


# + editable=true slideshow={"slide_type": "subslide"}
def plot_histogram(df, kind, lang):
    """Genera un histograma de frecuencia para una columna específica
    en un DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame que contiene los datos para generar el histograma.
    kind : str
        Nombre de la columna para la cual se generará el histograma.
    lang : str
        Idioma asociado a los datos.

    Returns:
    -------
    None
        Esta función muestra el histograma usando matplotlib.
    """
    counts = df[kind].value_counts().head(30)
    plt.bar(counts.index, counts.values)
    plt.xlabel(kind.upper())
    plt.ylabel('Frequency')
    plt.title(f'{kind} Frequency Graph for {lang}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# + editable=true slideshow={"slide_type": "fragment"}
plot_histogram(df, "category", "spa")

# + editable=true slideshow={"slide_type": "fragment"}
len(df[df["category"] == "001"])


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="PklDrh1Jz4I4" outputId="811ffb53-8fe7-4658-d32a-9ba0225d39b5" slideshow={"slide_type": "fragment"}
def get_corpora() -> pd.DataFrame:
    """Obtiene y combina datos de corpus de diferentes idiomas en un DataFrame
    obteniendo corpora multilingüe

    Returns:
    -------
    pd.DataFrame
        Un DataFrame que contiene los datos de corpus combinados de varios idiomas.
    """
    corpora = pd.DataFrame()
    for lang in LANGS:
        files = get_files(lang)
        raw_data = get_raw_corpus(files)
        dataframe = raw_corpus_to_dataframe(raw_data, lang)
        corpora = dataframe if corpora.empty else pd.concat([corpora, dataframe], ignore_index=True)
    return corpora

corpora = get_corpora()

# + editable=true slideshow={"slide_type": "fragment"}
corpora

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="c4Tu7qGcpfCp" outputId="71773b57-cd87-437f-a22e-92cc877f0631" slideshow={"slide_type": "subslide"}
for lang in LANGS:
    df = corpora[corpora["lang"] == lang]
    print(f"Basic stats for {LANGS[lang]}")
    print("Total words:", len(df["words"].unique()))
    print("Mean morphs: ", df["morphs_count"].mean())
    most_common_cat = df["category"].mode()[0]
    print("Most common category:", most_common_cat, CATEGORIES.get(most_common_cat, ""))
    print("="*30)

# + editable=true id="KzvrehOyMxBJ" slideshow={"slide_type": "subslide"}
# Lista de lenguas con sus colores
LANG_COLORS = {
    'ces': 'blue',
    'eng': 'green',
    'fra': 'red',
    'hun': 'purple',
    'spa': 'orange',
    'ita': 'brown',
    'lat': 'pink',
    'rus': 'gray'
}

def plot_multi_lang(column: str) -> None:
    """Genera un conjunto de subplots para mostrar la distribución
    de los datos de cierta columna en un dataframe para idiomas disponibles.

    Parameters:
    ----------
    column : str
        Nombre de la columna cuya distribución se va a graficar.

    Returns:
    -------
    None
        Esta función muestra los subplots usando matplotlib.
    """
    # Creando plots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
    fig.suptitle(f"{column.upper()} Distribution by Language", fontsize=16)
    # Iteramos sobre las lenguas y sus colores para plotearlos
    for i, (lang, color) in enumerate(LANG_COLORS.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        corpus = corpora[corpora['lang'] == lang]
        ax.hist(corpus[column], bins=10, edgecolor='black', alpha=0.7, color=color)
        ax.set_title(LANGS[lang])
        ax.set_xlabel(f"{column}")
        ax.set_ylabel("Frequency")

    # Ajustando el layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Mostramos el plot
    plt.show()


# + editable=true slideshow={"slide_type": "subslide"}
plot_multi_lang("word_len")

# + editable=true slideshow={"slide_type": "fragment"}
plot_multi_lang("morphs_count")

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# ### Práctica 1: Niveles lingüísticos I
#
# **Fecha de entrega: Domingo 3 de Marzo 2024 11:59pm**

# + [markdown] editable=true id="bFfv_5FnANgs" slideshow={"slide_type": "subslide"}
# 1. Agregar un nuevo modo de búsqueda donde se extienda el comportamiento básico del buscador para ahora buscar por frases. Ejemplo:
#
# ```
# [es_MX]>>> Hola que hace
#  /ola/ /ke/ /ase/
# ```
#
# 2. Agregar un modo de búsqueda donde dada una palabra te muestre sus *homofonos*[1]. Debe mostrar su representación IPA y la lista de homofonos (si existen)
#
# ```
# [es_MX]>>> habares
# /aβaɾes/
# ['abares', 'habares', 'havares']
# ```
#
# [1]: palabras con el mismo sonido pero distinta ortografía
#
# 3. Observe las distribuciones de longitud de palabra y de número de morfemas por palabra para todas lenguas. Basado en esos datos, haga un comentario sobre las diferencias en morfología de las lenguas
#
#

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# #### EXTRA
#
# A. Mejorar la solución en el escenario cuando no se encuentran las palabras en el dataset (incisos 1. y 2.) mostrando palabras similares. Ejemplo:
#
# ```
# [es_MX]>> pero
# No se encontro <<pero>> en el dataset. Palabras aproximadas:
# perro /pero/
# perno /peɾno/
# [es_MX]>>
# ```

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Links de interés

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# - [Regex 101](https://regex101.com/)

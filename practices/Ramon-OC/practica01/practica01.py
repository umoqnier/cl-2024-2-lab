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

# + [markdown] editable=true id="a5846871-b517-4373-946a-f7575ff2f848"
# # 1. Niveles lingüísticos I
#
# <center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/IPA_chart_2020.svg/660px-IPA_chart_2020.svg.png"></center>

# + [markdown] editable=true id="7568912a-5945-46c0-aeb4-6106e8f36635"
# ## Objetivo

# + [markdown] editable=true id="1db19ad2-dbd9-4a56-843a-1133e440641e"
# - Læs alumnæs entenderán que es la fonología y un alfabeto fonético
# - Manipularan y recuperará información de datasets disponibles en repositorios de Github para resolver una tarea específica
# - Læs alumnæs tendrán un acercamiento a la tarea de análisis morfológico
# - Hacer una comparación entre un enfoque basado en reglas y uno estadístico para tareas de NLP

# + [markdown] editable=true id="54563b4d-3c1a-4a23-9716-73554e263fb0"
# ## ¿Qué es la fonología?

# + [markdown] editable=true id="3fb13d94-e6af-44dc-aa94-608b99feafff"
# - La fonología es una rama de la Lingüística que estudia como las lenguajes sistematicamente organizan los fonemas
# - Estudia como los humanos producimos y percibimos el lenguaje
#     - Producción: La forma en que producimos el lenguaje
#     - Percepción: La forma en que interpretamos el lenguaje
#
# > Wikipedia contributors. Phonology. In Wikipedia, The Free Encyclopedia. https://en.wikipedia.org/w/index.php?title=Phonology&oldid=1206207687

# + [markdown] editable=true id="db43f194-732e-419f-b1c7-f5b0311f975d"
# ## ¿Qué es la fonética?

# + [markdown] editable=true id="8b2782c9-e7b5-4174-bae5-1a177e328783"
# - El estudio de los sonidos físicos del discurso humano. Es la rama de la lingüística que estudia la producción y percepción de los sonidos de una lengua con respecto a sus manifestaciones físicas.
#
# > Fonética. Wikipedia, La enciclopedia libre. https://es.wikipedia.org/w/index.php?title=Fon%C3%A9tica&oldid=155764166.

# + colab={"base_uri": "https://localhost:8080/", "height": 623} editable=true id="e529deca-fb69-4072-91ed-7a5063443d62" outputId="a64a9b5f-13ea-4b3a-ad94-d3c8dfddc2cd"
# %%HTML
# <center><iframe width='900' height='600' src='https://www.youtube.com/embed/DcNMCB-Gsn8?controls=1'></iframe></center>

# + [markdown] editable=true id="aa2812ca-f45c-4b74-aad7-54e50a0494e4"
# #### Formas comunes

# + [markdown] editable=true id="ae09ec8d-b9e9-4e66-9878-63c514fb57f0"
# - Oral-Aural
#     - Producción: La boca
#     - Percepción: Oidos

# + [markdown] editable=true id="0d2b82ad-715f-48df-b889-a3779444d943"
# - Manual-visual
#     - Producción: Manual usando las manos
#     - Percepción: Visual

# + [markdown] editable=true id="569d5a53-e809-4ff3-b865-f6d75973c48b"
# - Manual-Manual
#     - Producción: Manual usando las manos
#     - Percepción: Manual usando las manos

# + [markdown] editable=true id="029a8e17-03ed-417c-b2f3-512fd1ee5fd9"
# #### International Phonetic Alphabet (IPA)

# + [markdown] editable=true id="e4f827da-b836-47f5-9107-6aaabd04521b"
# - Las lenguas naturales tienen muchos sonidos diferentes por lo que necesitamos una forma de describirlos independientemente de las lenguas
# - Por ejemplo: Los sonidos del habla se determinan por los movimientos de la boca necesarios para producirlos
# - Las dos grandes categorías: Consonantes y Vocales
# - IPA es una representación escrita de los [sonidos](https://www.ipachart.com/) del [habla](http://ipa-reader.xyz/)

# + [markdown] editable=true id="COzTRH3QXdWl"
# ## Dataset: IPA-dict de open-dict
#
# - Diccionario de palabras para varios idiomas con su representación fonética
# - Representación simple, una palabra por renglon con el formato:

# + [markdown] editable=true id="309485ed-958e-4617-b4bc-b7dffe50ab07"
# ```
# [PALABRA][TAB][IPA]
#
# Ejemplos
# mariguana	/maɾiɣwana/
# zyuganov's   /ˈzjuɡɑnɑvz/, /ˈzuɡɑnɑvz/
# ```

# + [markdown] editable=true id="323ee664-922c-4842-843d-bee050f8a6b6"
# - [Github repo](https://github.com/open-dict-data/ipa-dict)
#   - [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
#   - URL: `https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/<iso-lang>`

# + [markdown] editable=true id="c-Q91_zR859L"
# ### Explorando el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true id="dfCkH58988vq" outputId="bac11d40-6472-46d1-f5a9-f5b4b38396e2"
# Explorando el corpus
import requests as r
from difflib import get_close_matches

response = r.get("https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt")
response.text[:100]

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="15c95896-097c-498e-97bd-d1d6872b4e3f" outputId="a63d726e-8662-4f7b-949e-638d2176fca3"
from pprint import pprint as pp
ipa_data = response.text.split("\n")
#print(ipa_data[-4:])
ipa_data[-1]
pp(ipa_data[400:410])

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="7127b08f-8b34-4276-9887-63d9da9b937f" outputId="f6ca9278-e96a-4ed5-86e4-d768ad111d02"
# Puede haber mas de una transcipcion asociada a una palabra
print(ipa_data[-3].split("\t"))
for data in ipa_data[300:500]:
    word, ipa = data.split('\t')
    representations = ipa.split(", ")
    if len(representations) >= 2:
        print(f"{word} --> {representations}")


# + [markdown] editable=true id="cJMkPF06jJJp"
# ### Obteniendo el corpus

# + editable=true id="34e980ae-c902-4f62-b1d2-a41b9e9939e3"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true id="89e9db85-e866-47fc-bf69-bce17fdb44a2" outputId="a1e2ffce-e701-459e-966b-b2493cae9f19"
response_to_dict(ipa_data[:100])["ababa"]


# + editable=true id="8e446564-cb5a-41a0-ac9a-e26f91e25273"
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


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="4473a933-2aa1-4f06-a475-03e17ca5e9dc" outputId="fb25eb45-097d-4929-88f5-4b9374cd3b0c"
es_mx_ipa = get_ipa_dict("es_MX")


# + editable=true id="3vfeGyqYkI9V"
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


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="30a6c732-0bd8-49da-9139-403e8237f701" outputId="f7dd282c-c2c7-4657-b3d9-169b72fe3cee"
query_ipa_transcriptions("mayonesa", es_mx_ipa)

# + [markdown] editable=true id="h9Ri8YmwMnxR"
# #### Obtengamos un par de datasets

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="SDspkhcdLmtx" outputId="164361b8-da8c-4e43-c4e3-6cbbadec6401"
# Get datasets
dataset_mx = get_ipa_dict("es_MX")
dataset_us = get_ipa_dict("en_US")

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="460eff42-895a-4bb8-bb5d-29761dfc71f7" outputId="a00d4314-2d7c-4c8f-a658-fcade7ee1835"
# Simple query
query_ipa_transcriptions("beautiful", dataset_us)

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="abpslzJRMvD6" outputId="fc722302-f38f-4a1a-f460-d4a94a68180a"
# Examples
print(f"dog -> {query_ipa_transcriptions('dog', dataset_us)}🐶")
print(f"mariguana -> {query_ipa_transcriptions('mariguana', dataset_mx)} 🪴")

# + [markdown] editable=true id="0669fac2-84a9-4fb4-8794-606001ef3043"
# #### Diferentes formas de pronunciar dependiendo la lengua, aunque la ortografía se parezca. Ejemplo:

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="drw73avU9-ct" outputId="ed815f1a-5418-4089-8bc3-36478d5a19a2"
# Ilustrative example
print("[es_MX] hotel |", query_ipa_transcriptions("hotel", dataset_mx))
print("[en_US] hotel |", query_ipa_transcriptions("hotel", dataset_us))

# + [markdown] editable=true id="0cd27423-8154-48e2-92c2-b527dcc4833a"
# ### Obteniendo corpora desde GitHub

# + editable=true id="YSRb9cx5jM8d"
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


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="WcCmgrgnT9wK" outputId="5e072170-8907-45a1-cce5-3d74ffe49265"
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

# + [markdown] editable=true id="cbe43776-565d-4b0f-aaae-e031576893a5"
# ### Creando aplicaciones con estos datos

# + [markdown] editable=true id="9c107020-3eee-41b8-b0aa-182cf142b71d"
# #### 1. Busquedas básicas automatizada
# Buscador de representaciones foneticas de palabras automatizado en diferentes idiomas

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="f5b7fbc2-4b95-4233-9fa6-17493bb2afb3" outputId="5c5f8099-458f-403a-84c7-71daa1510e52"
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

# + [markdown] editable=true id="8TLGghJWFbIZ"
# ### 2. Encontrando palabras que tengan terminación similar
#
# Dada una oración agrupar las palabras que tengan una pronunciación similar

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="B1JaTm-lUy3c" outputId="741d8c1b-9677-4483-a8d6-83155bbb5bca"
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

# + [markdown] editable=true id="1e792826-9085-4aaa-835e-62f83de99c2b"
# ## Morfología y análisis morfológico

# + [markdown] editable=true id="b25df665-b41f-42a8-a6b4-6ff42ec0157e"
# ### ¿Qué es la morfología?

# + [markdown] editable=true id="1bdec1d2-c01c-48fb-a008-d72f403e5638"
# La morfología es uno de los niveles de la lengua que estudia los procesos que conforman una palabra.
#
# > Morfología es el estudio de la estructura interna de las palabras (Bauer, 2003)
#
# - niñ-o
# - niñ-a
# - niñ-o-s
# - gat-a-

# + [markdown] editable=true id="775314dd-f32c-42c2-aa8b-5c85af7122b4"
# ### Morfemas

# + [markdown] editable=true id="8f15acbd-8061-4188-840f-9d19c52e4da3"
# - Con la morfología podemos identificar como se modifica el significado variando la estructura de las palabras
# - Tambien las reglas para producir:
#     - niño -> niños
#     - niño -> niña
# - Tenemos elementos mínimos, intercambiables que varian el significado de las palabras: **morfemas**
#
# > Un morfema es la unidad mínima con significado en la producción lingüística (Mijangos, 2020)

# + [markdown] editable=true id="73d27bcd-238f-49ba-beff-adf14b89d155"
# #### Tipos de morfemas

# + [markdown] editable=true id="835da80a-1aba-4488-b268-9e7c1ecaf2af"
# - Bases: Subcadenas que aportan información léxica de la palabra
#     - sol
#     - frasada
# - Afijos: Subcadenas que se adhieren a las bases para añadir información (flexiva, derivativa)
#     - Prefijos
#         - *in*-parable
#     - Subfijos
#         - pan-*ecitos*, come-*mos*

# + [markdown] editable=true id="70b6a504-e35e-49a7-b5ab-d6472f133179"
# ### Aplicaciones relacionadas a la morfología en NLP

# + [markdown] editable=true id="2d554d7f-818f-4d28-a7c3-a68cdbb94a5f"
# #### Análisis morfológico

# + [markdown] editable=true id="51d55f76-bb9e-42a8-81b3-e52b34956891"
# La morfología es uno de los niveles más básicos del lenguaje que se puede estudiar. En ese sentido, una de las tareas más básicas del NLP es el análisis morfológico:
#
# > El análisis morfológico es la determinación de las partes que componen la palabra y su representación lingüística, es una especie de etiquetado
#
# Los elementos morfológicos son analizados para:
#
# - Determinar la función morfológica de las palabras
# - Hacer filtrado y pre-procesamiento de texto

# + [markdown] editable=true id="e7f5a261-3350-4b31-9310-fb977fc51e65"
# #### Ejemplo: Parsing con expresiones regulares
#
# La estructura del sustantivo en español es:
#
# ` BASE+AFIJOS (marcas flexivas)   --> Base+DIM+GEN+NUM`

# + editable=true id="c814978a-77ee-475f-b83e-cbe01b34a222"
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

# + editable=true id="896e1feb-ea78-4356-8ff5-db361f3ac827"
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


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="a44f95cd-30b1-4487-a042-7fa6f7386825" outputId="1c122171-ea19-430e-ae0a-332e3ff74534"
morph_parsing = morph_parser_rules(palabras)
for palabra, parseo in zip(palabras, morph_parsing):
    print(palabra, "-->", parseo)

# + [markdown] editable=true id="85e002ad-5faa-4c9e-94ef-f28fda5ddec1"
# #### Preguntas 🤔
# - ¿Qué pasa con las reglas en lenguas donde son más comunes los prefijos y no los sufijos?
# - ¿Cómo podríamos identificar características de lenguas

# + [markdown] editable=true id="81c78c72-e782-4263-a4c9-fd201479507c"
# ## Corpus: [SIGMORPHON 2022 Shared Task on Morpheme Segmentation](https://github.com/sigmorphon/2022SegmentationST/tree/main)

# + [markdown] editable=true id="d7920f28-4d04-4a60-a946-04dfc82c62fd"
# - Shared task donde se buscaba convertir las palabras en una secuencia de morfemas
#     - ¿Qué es un shared task?
# - Dividido en dos partes:
#     - Segmentación a nivel de palabras (nos enfocaremos en esta)
#

# + [markdown] editable=true id="23e1527c-64e3-401b-88b3-59fc2347762a"
# ### Track: WORDS

# + [markdown] editable=true id="6859cdcb-c1db-418f-812b-d4a038a21ce5"
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

# + [markdown] editable=true id="84feb532-32a6-4b75-bb1a-30d1f0663e8c"
# ### Obteniendo el corpus

# + colab={"base_uri": "https://localhost:8080/", "height": 34} editable=true id="df8c5d00-cb65-450e-a20a-9d619006e1fc" outputId="2d2db30f-2b95-4430-b058-a20b0348e21d"
response = r.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/spa.word.test.gold.tsv")
response.text[:100]

# + colab={"base_uri": "https://localhost:8080/"} id="9e1bbf4f-8631-40a0-9674-f8d75b9c8d7e" outputId="2b1455ff-3b52-4ff8-e86c-79a840acc0f8"
raw_data = response.text.split("\n")
raw_data[:10]

# + colab={"base_uri": "https://localhost:8080/"} id="5ce284f4-f500-4eca-a77f-5af219d0c44a" outputId="cc8a52e2-ec8a-495c-c609-f27951011bf1"
element = raw_data[0].split("\t")
element[1].split()

# + colab={"base_uri": "https://localhost:8080/"} id="2de42da7-bf33-4aa9-99b4-9c8c24cd032d" outputId="b4fd57c5-b944-49a0-8d17-c7aab485f911"
for row in raw_data[:10]:
    word, morphs, category = row.split("\t")
    print(word, morphs, category)
    print(morphs.split())

# + editable=true id="0011c59e-c368-4256-bd9d-60dd88627574"
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


# + editable=true id="c50cebf5-7429-46fd-b443-2effe5107963"
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


# + editable=true id="1e70648e-81f1-4ed9-b94c-d0e01172e2b6"
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


# + colab={"base_uri": "https://localhost:8080/"} id="ef96f081-3979-4779-8973-2d63e0446f0c" outputId="49355474-bda8-4e13-95a2-7004eba63bc5"
get_raw_corpus(get_files(lang="ita"))[:10]


# + editable=true id="9fa8d8b8-b574-4547-ad95-37507b229bab"
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


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="62e5ac8c-2df7-4700-9fb4-cdbe0789e42f" outputId="031d1ecd-8449-44ee-dc18-d6ae5e69ac3a"
# Get data
files = get_files("spa")
raw_data = get_raw_corpus(files)
df = raw_corpus_to_dataframe(raw_data, lang="spa")

# + colab={"base_uri": "https://localhost:8080/", "height": 411} id="cea3c988-eea7-4a0b-bfa8-98d090e09393" outputId="3ecc2495-2ef0-4b3b-aeef-105e6cf2c1cf"
df

# + [markdown] editable=true id="94947a56-33f4-4440-9209-92f2c84d31c2"
# ### Análisis cuantitativo para el Español

# + colab={"base_uri": "https://localhost:8080/"} id="4f758040-00ac-46b8-a4ce-229f32c1edbb" outputId="e44dcece-1746-45bf-dadc-c05864433c34"
df["category"].value_counts().head(30)

# + colab={"base_uri": "https://localhost:8080/"} id="8dee95c6-07ce-4740-8174-fed77aab2d5a" outputId="6d9bf19d-6ef0-4f4d-cd43-76378a8cb827"
df["morphs_count"].mean()

# + colab={"base_uri": "https://localhost:8080/"} id="8747dd4f-0a5e-4186-bdad-24a4b309aa34" outputId="baa7d709-90e3-4862-b562-80d1b5d6df5a"
df["word_len"].mean()

# + colab={"base_uri": "https://localhost:8080/", "height": 448} editable=true id="40922870-0769-49d5-b551-618f8429f959" outputId="b49f7c0a-4f89-4818-b4d4-713aa7fb641f"
import matplotlib.pyplot as plt
plt.hist(df["word_len"], bins=10, edgecolor="black")
plt.xlabel("Word len")
plt.ylabel("Freq")
plt.show()


# + editable=true id="a438d893-3bf1-4779-bee4-f1d1cc016e51"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 487} editable=true id="11f5da0a-3283-47b6-b3ba-035f0f153823" outputId="afe55e19-d336-4b30-e499-5402721be317"
plot_histogram(df, "category", "spa")

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="5482f50d-b767-4a68-9101-f75b1889c3be" outputId="e501aad1-a8f8-4e03-e2d9-1fc9dcaeb45a"
len(df[df["category"] == "001"])


# + colab={"base_uri": "https://localhost:8080/"} editable=true id="b3aeb0b8-3972-4b8e-b92d-2a8d03279a7f" outputId="da4f7518-a6fe-4237-ad45-87392028d99a"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 411} editable=true id="ba5e9b14-703b-4dad-8a58-5f3b5ccb09a2" outputId="77e37bf9-7ff3-4a29-9fd5-265c6f893fe3"
corpora

# + colab={"base_uri": "https://localhost:8080/"} editable=true id="460dee5f-69b8-4ea0-b6a0-1b6b7e1f0515" outputId="34f441e6-07c8-445b-dd21-a341de867072"
for lang in LANGS:
    df = corpora[corpora["lang"] == lang]
    print(f"Basic stats for {LANGS[lang]}")
    print("Total words:", len(df["words"].unique()))
    print("Mean morphs: ", df["morphs_count"].mean())
    most_common_cat = df["category"].mode()[0]
    print("Most common category:", most_common_cat, CATEGORIES.get(most_common_cat, ""))
    print("="*30)

# + editable=true id="d81aaf52-f7c0-4c83-9945-2cd9163295a5"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} editable=true id="beb66fff-35aa-4a7d-96e5-d0eba598feb5" outputId="8058c7a0-fa27-4e6e-8432-ffe0440b788e"
plot_multi_lang("word_len")

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} editable=true id="b70289ac-12a0-4acd-998e-5600356a9b20" outputId="7df74d10-3595-4578-e46c-826fc5792a04"
plot_multi_lang("morphs_count")


# + [markdown] editable=true id="7f66a8a0-c52c-4b1e-aa7d-761650f256df"
# ### Práctica 1: Niveles lingüísticos I
#
# **Fecha de entrega: Domingo 3 de Marzo 2024 11:59pm**

# + [markdown] editable=true id="bFfv_5FnANgs"
# 1. Agregar un nuevo modo de búsqueda donde se extienda el comportamiento básico del buscador para ahora buscar por frases. Ejemplo:
#
# ```
# [es_MX]>>> Hola que hace
#  /ola/ /ke/ /ase/
# ```

# + id="c445e62f"
def sentence_IPA(sentence: str, dataset: dict) -> str:
    """Search for a sentence in an IPA phonetics dict

      Given a sentence this function returns a string with its IPA phonetics

      Parameters:
      -----------
      sentence: str
          A sentence to search in the dataset
      dataset: dict
          A dataset for a given language code
      Returns
      -------
      str:
          String with posible homophones.
      """
    x = sentence.split()
    result = ""
    for word in x:
      #print(word)
      word = search_word(word, dataset);
      wordResult = query_ipa_transcriptions(word, dataset);
      if (wordResult[0] != "NOT FOUND"):
          result += wordResult[0] + " "
    return result


# + [markdown] id="db2b7bfb"
#
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

# + id="50a18dd8"
def query_homophones(searchWord: str, dataset: dict) -> list[str]:
    """Search for a word in an IPA phonetics dict

      Given a sentence this function returns a string with its homophones

      Parameters:
      -----------
      searchWord: str
          A word to search in the dataset
      dataset: dict
          A dataset for a given language code
      Returns
      -------
      str:
          String with posible homophones.
      """
    ipa = query_ipa_transcriptions(searchWord, dataset);
    homophones = []
    if(ipa[0] != "NOT FOUND"):
      for word in dataset:
        if dataset[word] == ipa[0]:
          homophones.append(word)

    return homophones


# + [markdown] id="0ddecfa2"
#
# 3. Observe las distribuciones de longitud de palabra y de número de morfemas por palabra para todas lenguas. Basado en esos datos, haga un comentario sobre las diferencias en morfología de las lenguas

# + [markdown] id="e7f98ba8"
# Lenguas como el francés, inglés, español e italiano comparten una longitud promedio de palabra bastante similar, aunque se puede notar una variación ligera en la distribución morfológica de sus palabras. A pesar de esto, su morfología exhibe un comportamiento comparable, estos lenguajes tienden a mantenerse dentro de valores bajos en la construcción de palabras dada su morfología. Destaca de los resultados el latín, donde todas sus palabras generalmente se conforman de tres morfemas como máximo, con la mayoría de las palabras contenidas en el rango de dos a tres. Sin embargo, la longitud de sus palabras guarda gran similitud con la del italiano.
# En el caso del ruso, se observa un comportamiento particular en cuanto al número de morfemas, con una distribución equilibrada y la mayoría de sus palabras oscilando entre dos y cinco morfemas. La longitud de sus palabras, al igual que en el resto de las lenguas mencionadas, se distribuye de forma similar. Finalmente, se detecta en el húngaro y el checo una distribución análoga, aunque con una diferencia de cinco letras en la extensión de la palabra, cinco palabras más en el húngaro. A pesar de ello, el número de morfemas es muy semejante en ambas.

# + [markdown] editable=true id="4db5d5d4-39b5-4fb8-a582-546e0732e197"
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

# + id="97125f31"
def search_word(word: str, dataset: dict) -> str:
  """ Search for a word in an IPA phonetics dict, if it´s not present it shows multiple
      suggestions

      Parameters:
      -----------
      dataset: dict
          A dataset for a given language code
      Returns
      -------
      str:
          String with a valid word for the dataset.
      """
  transcription = query_ipa_transcriptions(word, dataset)
  if(transcription[0] != "NOT FOUND"):
    return word
  else:
    while transcription[0] == "NOT FOUND":
      sugerencias = get_close_matches(word, dataset.keys(), n=3, cutoff=0.6)
      if sugerencias:
        mensaje = f"No se encontró '{word}' en el dataset. Palabras aproximadas:"
        for sugerencia in sugerencias:
            mensaje += f"\n{sugerencia}"
        print(mensaje)
        word = input(f" word>> ")
        transcription = query_ipa_transcriptions(word, dataset)
      else:
        print("No se encontraron palabras similares a '{word}' en el dataset.")
        word = input(f" word>> ")
        transcription = query_ipa_transcriptions(word, dataset)

  return word


# + id="65a1e3df"
def word_IPA(searchWord: str, dataset: dict) -> str:
    """Search for a word in an IPA phonetics dict

      Given a string this function returns IPA representation
      Parameters:
      -----------
      searchWord: str
          A word to search in the dataset
      dataset: dict
          A dataset for a given language code
      Returns
      -------
      str:
          String with original word.
      """
    result = query_ipa_transcriptions(searchWord, dataset)
    return searchWord + " | " + result[0]


# + [markdown] id="6e4de293"
# ### Browser with multiple options:
#
# 1. Get IPA transcript (word)
# 2. Get IPA transcript (phrase)
# 3. Homophone search
#

# + colab={"base_uri": "https://localhost:8080/"} id="201bbfdb" outputId="056fd66d-62c1-447a-8879-e886250f1bd7"
def browser():
    print("Lenguas disponibles:")
    for lang_key in dataset.keys():
      print(f"{lang_key}: {lang_codes[lang_key]}")

    lang = input("lang>> ")
    sub_data = dataset[lang]

    print("1. Obtener transcripción IPA (palabra) \n2. Obtener transcripción IPA (frase)\n3. Búsqueda de homófonos")

    while True:
      mode = input("Seleccione el modo (1, 2 o 3): ")
      if mode not in ['1', '2', '3']:
        print("Modo no válido.")
        break
      else:
        if mode == '1':
            word = input(f" word>> ")
            word = search_word(word, sub_data)
            result = word_IPA(word, sub_data)
            print(result)
        elif mode == '2':
            sentence = input(f"[{lang}] sentence>> ")
            result = sentence_IPA(sentence, sub_data)
            print(result)
        else:
            word = input(f" word>> ")
            word = search_word(word, sub_data)
            results = query_homophones(word, sub_data)
            print(results)

browser()

# + [markdown] editable=true id="fe21d1d3-3f45-4a43-aec3-0496c535c371"
# ### Links de interés

# + [markdown] editable=true id="c742951f-6322-4863-b91a-9b5716b11190"
# - [Regex 101](https://regex101.com/)

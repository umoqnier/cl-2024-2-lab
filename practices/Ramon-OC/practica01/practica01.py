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

# + [markdown] id="ioqX3obg5WcJ"
# # Práctica 1: Niveles lingüísticos I
# **Fecha de entrega: Domingo 3 de Marzo 2024 11:59pm**

# + id="dfCkH58988vq"
import requests as r
from difflib import get_close_matches

response = r.get("https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt")


# + [markdown] id="cJMkPF06jJJp"
# ### Obteniendo el corpus

# + id="34e980ae-c902-4f62-b1d2-a41b9e9939e3"
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


# + id="8e446564-cb5a-41a0-ac9a-e26f91e25273"
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


# + id="3vfeGyqYkI9V"
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


# + [markdown] id="0cd27423-8154-48e2-92c2-b527dcc4833a"
# ### Obteniendo corpora desde GitHub

# + id="YSRb9cx5jM8d"
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


# + id="WcCmgrgnT9wK" colab={"base_uri": "https://localhost:8080/"} outputId="5f5e11fa-5a30-4182-d21e-e8411af4f3ba"
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


# + [markdown] id="bFfv_5FnANgs"
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
      word = validate_word(word, dataset);
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


# + [markdown] id="4db5d5d4-39b5-4fb8-a582-546e0732e197"
# #### EXTRA

# + id="97125f31"
def validate_word(word: str, dataset: dict) -> str:
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


# + [markdown] id="6e4de293"
# ### Browser with multiple options:
#
# 1. Get IPA transcript (word)
# 2. Get IPA transcript (phrase)
# 3. Homophone search
#

# + colab={"base_uri": "https://localhost:8080/"} id="201bbfdb" outputId="eaa5a819-7647-40ab-9fb3-7f6255ca014f"
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
            word = validate_word(word, sub_data)
            result = word_IPA(word, sub_data)
            print(result)
        elif mode == '2':
            sentence = input(f"[{lang}] sentence>> ")
            result = sentence_IPA(sentence, sub_data)
            print(result)
        else:
            word = input(f" word>> ")
            word = validate_word(word, sub_data)
            results = query_homophones(word, sub_data)
            print(results)

browser()
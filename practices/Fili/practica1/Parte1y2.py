#!/usr/bin/env python
# coding: utf-8

# ## PRACTICA 1
# 1. Agrega un nuevo modo de búsqueda donde se extienda el comportamiento básico del buscador para ahora buscar por frases.
# 2. Agregar un modo de búsqueda donde dada una palabra te muestre sus homofonos[1]. Debe mostrar su representación IPA y la lista de homofonos (si existen)
# 3. Observe las distribuciones de longitud de palabra y de número de morfemas por palabra para todas lenguas. Basado en esos datos, haga un comentario sobre las diferencias en morfología de las lenguas
# 
# [1]: palabras con el mismo sonido pero distinta ortografía

# ### Preprocesamiento

# In[4]:


import requests as r
response = r.get("https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt")


# In[6]:


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


# In[7]:


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


# In[22]:


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


# In[36]:


def query_ipa_transcriptions(word: str, dataset: dict) -> tuple:
    """Buscar representaciones fonéticas y homófonos de una palabra en el dataset.

    Parameters:
    ----------
    word : str
        La palabra para la cual se buscarán las representaciones fonéticas y los homófonos.
    dataset : dict
        El conjunto de datos en el que buscar.

    Returns:
    -------
    tuple
        Una tupla que contiene la representación fonética de la palabra (str) y una lista de homófonos (list).
    """
    ipa_transcriptions = dataset.get(word.lower(), "NOT FOUND").split(", ")
    homophones = [w for w in dataset.keys() if dataset[w] == dataset[word.lower()]]
    return ipa_transcriptions, homophones


# In[8]:


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


# In[9]:


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


# In[ ]:





# ### 1. y 2.

# In[18]:


print("BUSQUEDA FONÉTICA DE FRASES")
print("Lenguajes disponibles:\n")
for lang_key in dataset.keys():
    print(f"{lang_key}: {lang_codes[lang_key]}")


# In[45]:


lang = input("lang>>")
print(f"Selecciona un lenguaje: {lang_codes[lang]}") if lang else print("Adios")


# In[46]:


while lang:
    # El programa comienza aquí
    sub_data = dataset[lang]
    
    # Pregunta al usuario si quiere buscar por palabra o por frase
    search_mode = input(f"[{lang}] ¿Qué desea buscar? (palabra/frase/homofonos)>> ").lower()

    if search_mode == 'palabra':
        while True:
            # Modo de búsqueda por palabra
            query = input(f"[{lang}] palabra>> ")
            if not query:
                break  # Salir del bucle si no se ingresa ninguna palabra
            ipa_transcriptions, homophones = query_ipa_transcriptions(query, sub_data)
            print(f"Representación IPA de '{query}': {ipa_transcriptions}")
            if homophones:
                print(f"Homófonos de '{query}': {homophones}")
            else:
                print(f"No se encontraron homófonos para '{query}'")

    elif search_mode == 'frase':
        while True:
            # Modo de búsqueda por frase
            phrase = input(f"[{lang}] frase>> ")
            if not phrase:
                break  # Salir del bucle si no se ingresa ninguna frase
            
            # Dividir la frase en palabras y buscar cada palabra individualmente
            words_in_phrase = phrase.split()
            for word in words_in_phrase:
                results = query_ipa_transcriptions(word, sub_data)
                print(word, " | ", results)
    elif search_mode == 'homofonos':
        while True:
            # Modo de búsqueda por homófonos
            word = input(f"[{lang}] palabra>> ")
            if not word:
                break  # Salir del bucle si no se ingresa ninguna palabra
            homophones = query_ipa_transcriptions(word, sub_data)[1]
            if homophones:
                print(f"Homófonos de '{word}': {homophones}")
            else:
                print(f"No se encontraron homófonos para '{word}'")

    else:
        print("Modo de búsqueda no válido. Por favor, elija 'palabra' o 'frase'.")

    # Pedir al usuario si desea cambiar de idioma o salir
    lang = input("lang>> ")


# In[ ]:





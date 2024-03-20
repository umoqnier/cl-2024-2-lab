
import requests as r

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


"""--------------- EJERCICIO 1 -------------------"""
print("representación fonética de frases")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")
while lang:
    # El programa comeinza aqui
    sub_data = dataset[lang]
    query = input(f"[{lang}] phrase>> ")
    
    # Separando por espacios
    word_list = query.split()
    results = []
    
    # Separando las palabras de la frase
    for word in word_list:
        results += query_ipa_transcriptions(word, sub_data)
    
    print(query, " | ", results)

    lang = input("lang>> ")

    
"""--------------- EJERCICIO 2 -------------------"""
print("Buscador de homofonos")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")
while lang:
    # El programa comeinza aqui
    sub_data = dataset[lang]
    query = input(f"[{lang}] word>> ")
    
    # Separando por espacios
    word_list = query.split()
    results = []
    
    # Separando las palabras de la frase
    for word in word_list:
        results += query_ipa_transcriptions(word, sub_data)
    
    print(query, " | ", results)

    """ -------- aquí empuieza la busqueda de homofonos --------"""
    
    response = r.get(f"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{lang}.txt") 
    raw_data = response.text.split("\n")

    inverted_list = []
    for item in raw_data[:-1]:
        iterator = item.split("\t")
        inverted_list.append(iterator[1]+"\t"+iterator[0])

    # verificando si en la lista hay coincidencias en los ipas
    homofonos = ""
    for item in inverted_list:
        iterator = item.split("\t")
        
        if results[0] == iterator[0]:            
            homofonos += iterator[1]+", "

    print("homofonos encontrados: "+homofonos)
    
    lang = input("lang>> ")

    
"""--------------- EJERCICIO 3 -------------------

lo que notabamos y lo que se discutía en clase acerca de las lenguas
que tieden a ser muy aglutinantes es que dicen mucho más con una palabra,
es decir que en una palabra viene más información a diferencia de las lenguas
no aglutinantes. """
        

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

print("Descargando recursos...")
dataset = get_dataset()

# Motor de búsqueda

print("Representación fonética de palabras")

print("Lenguas disponibles:")
for lang_key in dataset.keys():
    print(f"{lang_key}: {lang_codes[lang_key]}")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("📷")
while lang:
    sub_data = dataset[lang]
    keys = list(sub_data.keys())
    values = list(sub_data.values())
    query = input(f"[{lang}]>> ")
    while query:
        response=""
        hom = ""
        query_arr = query.split(" ")
        if len(query_arr) == 1:
            while hom != "S" and hom != "N":
                hom = input("Mostrar homofonos (S/N): ")
        for word in query_arr:
            results = query_ipa_transcriptions(word, sub_data)
            if len(results) > 1:
                response += "("+" | ".join(results)+")"+" "
            else:
                response += results[0]+" "
        print(response)
        if hom == "S":
            h = []
            for result in results:                
                i = values.index(result)
                h.append(keys[i])
                while result in values[i+2:]:                    
                    i = values.index(result,i+2)
                    h.append(keys[i])        
            print(", ".join(h))
        query = input(f"[{lang}]>> ")
    lang = input("lang>> ")

# EJERCICIO 3
# En el caso particular del ruso es más notoria una interpretación de la Ley de Menzerath-Altmann: a mayor longitud de una palabra corresponden componentes (morfemas) más pequeñas, donde la media de la longitud de una palabra es de 10 caracteres y cada palabra consta alrededor de 2 a 5 morfemas
# Es interesante notar que las distribuciones de las lenguas que derivan del latín manitienen cierta relación, como doblar el rango de los morfemas y mantener una longitud similar en cuanto a sus palabras, características que constrastan con el resto de lenguas
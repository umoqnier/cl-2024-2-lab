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

# + id="ay9LxfZZKunR"
import requests as r
import re


# + id="t4vGBh3GKxCc"
def response_to_dict(ipa_list: list) -> dict:

    result = {}
    for item in ipa_list:
       item_list = item.split("\t")
       result[item_list[0]] = item_list[1]
    return result


# + id="y2HYo40tKyur"
def get_ipa_dict(iso_lang: str) -> dict:

    print(f"Downloading {iso_lang}", end=" ")
    response = r.get(f"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{iso_lang}.txt")
    raw_data = response.text.split("\n")
    print(f"status:{response.status_code}")
    return response_to_dict(raw_data[:-1])


# + id="IwcC2AgqK0W1"
def query_ipa_transcriptions(word: str, dataset: dict) -> list[str]:

    return dataset.get(word.lower(), "NOT FOUND").split(", ")


# + id="h-RCXSEIK2U-"
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
  "zh_hans": "Mandarin (Simplified)"
}
iso_lang_codes = list(lang_codes.keys())


# + colab={"base_uri": "https://localhost:8080/"} id="_-zYXH3KK4SC" outputId="4ac07b2c-663c-46f2-adf6-252a63d4dd50"
def get_dataset() -> dict:

    return {code: get_ipa_dict(code) for code in iso_lang_codes}

dataset = get_dataset()


# + [markdown] id="LPNHD8yJ84F5"
# #1. Modo de Oraciones.

# + id="3Rdo4w7yK6Z_"
def modo_sentence(lang,query):
    if lang not in ['zh_hans', 'ja', 'ko', 'yue']:
        palabras = re.sub(r"[,./|]", '', query).split(" ")
        results = ''
        for i in palabras:
            results += query_ipa_transcriptions(i, sub_data)[0]
        print(query, " | ", results)
    else:
        palabras = list(re.sub(r"[,./|。 ]", '', query))
        results = ""
        for i in palabras:
            results += query_ipa_transcriptions(i, sub_data)[0]
        print(query, " | ", results)



# + [markdown] id="abx0EUs_8-JD"
# # 2. Modo de homófonos

# + id="03F_2HMELHmq"
def modo_homophones(lang, query):
  l = list()
  t = query_ipa_transcriptions(query, dataset[lang])
  s = ''
  for i in t:
    s += i + '\t'
  print(s)
  for i in dataset[lang].keys():
    b = False
    for u in t:
      for j in query_ipa_transcriptions(i, dataset[lang]):
        if j == u:
          b = True
    if b:
      l.append(i)
  if len(l) == 0:
    print('NO HOMOPHONES FOUND')
  else:
    print(l)



# + [markdown] id="gxKM53vN9IEU"
# # Extra: palabras similares

# + id="QThM5rcqLKX6"
def similares(lang, query):
  l = list()
  for i in dataset[lang].keys():
    for p in range(len(query)):
      s = ''
      for j in range(len(query)):
        if j != p:
          s = s + query[j]
        else:
          s = s + '.?'
      if re.fullmatch(s, i):
        l.append(i)
  return l


# + id="jtQqT8GHLNMT"
def modo_word(lang, query):
  if lang in ['zh_hans', 'ja', 'ko', 'yue']:
    print(query_ipa_transcriptions(query, dataset[lang]))
  elif query_ipa_transcriptions(query, dataset[lang])[0] == 'NOT FOUND':
    print(f"{query} no fue encontrada. Palabras similares:\n {similares(lang,query)}")
  else:
    print(query_ipa_transcriptions(query, dataset[lang])[0])


# + id="gzrv0dNzLPEQ"
def fun(lang):
  modo = input(f"[{lang}] modo >> ")
  while modo:
    match int(modo):
        case 1 :
            query = input(f"[{lang}] sentence>> ")
            modo_sentence(lang, query)
            while query:
                query = input(f"[{lang}] sentence >> ")
                if query:
                    modo_sentence(lang, query)
        case 2 :
            query = input(f"[{lang}] word >> ")
            modo_word(lang, query)
            while query:
                query = input(f"[{lang}] word >> ")
                if query:
                    modo_word(lang, query)
        case 3 :
            query = input(f"[{lang}] homófonos >> ")
            modo_homophones(lang, query)
            while query:
                query = input(f"[{lang}] homófonos >> ")
                if query:
                    modo_word(lang, query)
        case _ :
            break
    modo = input(f"[{lang}] modo >> ")


# + [markdown] id="nEEbrMUl9NqR"
# # Función principal.

# + colab={"base_uri": "https://localhost:8080/"} id="6q994M5yLWMH" outputId="6987aa02-e98f-4121-fc3a-d5c1febf2be9"

print("Lenguas disponibles:")
for lang_key in dataset.keys():
    print(f"{lang_key}: {lang_codes[lang_key]}")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adiós ")
while lang:
    sub_data = dataset[lang]
    print("Modos disponibles: \n1) Oraciones\n2) Palabras\n3) Homofónos")
    fun(lang)
    lang = input("lang>> ")
    print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")

# + id="wV-RfjnlLZy-"
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


# + id="7u-Egmx0MXyk"
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


# + id="Z_ZzxE3yMaH-"
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


# + id="odbWc0fTMb9N"
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


# + colab={"base_uri": "https://localhost:8080/"} id="xIt7X3NnMfLH" outputId="d3f351d8-8fcf-4538-bbc4-4e72b29a8b9f"
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

# + id="VRHSeGvQMk-8"
import matplotlib.pyplot as plt
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


# + colab={"base_uri": "https://localhost:8080/", "height": 887} id="Fut-UUkxMo9o" outputId="d65bc9b9-de79-40dc-8f01-6fd04c6ee032"
plot_multi_lang("word_len")

# + colab={"base_uri": "https://localhost:8080/", "height": 888} id="BK0lM0TDMq8l" outputId="4e00945e-d7b0-4af4-d4aa-ca4bf1deb6f9"
plot_multi_lang("morphs_count")

# + [markdown] id="-qXbEErd8PRd"
# # 3. Comentario de distribuciones.
# Al observar las gráficas de las distribuciones de longitud de palabras y morfemas podemos ver que reflejan la forma en que estos idomas se hablan y que comparten ciertas características. Por ejemplo si observamos las gráficas de morfemas de los idiomas eslavos, ruso y checo, notamos que por la naturaleza de los idiomas eslavos tienen una media notablemente superior a la de las lenguas romances y el ingés ya que en éstas últimas se ha preferido aportar información por medio de más palabras como temporalidad, posición, relación con otros sustantivos, etc. como las part y en el ruso y checo hace uso de afijos y casos para añadir información y comunicarse de forma precisa. Pero en el caso de longitud de palabras notamos que estos idiomas eslavos difieren considerablemente. A esto podemos suponer varias explicaciones como por ejemplo como se muestra en las gráficas de morfemas, el checo tiene una media inferior en morquemas o que en checo se usa el alfabeto latino con dicríticos lo que puede condensar la pronunciación de ciertos sonidos en menos caracteres, otra posible razón es que el checo tiende a una escritura más fonética y las reducciones vocales o consonantes mudas se mantienen en el ruso. En el resto de idiomas de las gráficas con excepción del latín, tienen una cantidad de morfemas y longitud de palabras similar que personalmente creo es por la naturaleza del dataset que selecciona cierto tipo de palabras

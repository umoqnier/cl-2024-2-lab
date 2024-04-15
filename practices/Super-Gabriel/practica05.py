from gensim.models import Word2Vec
import gensim.downloader as api



# Descargando el corpus de Wikipedia
corpus = api.load('text8')

# Entrenando el modelo Word2Vec
model = Word2Vec(corpus)

# guardando el modelo para no repetir siempre que
# se interprete el archivo.
#model.save("model.bin")



# cargando el modelo ya entrenado (las lineas anteriores son para entrenarlo)
#model = Word2Vec.load("modelP05.bin")


from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Obteniendo las palabras y sus frecuencias
word_freq = [(word, model.wv.key_to_index[word]) for word in model.wv.index_to_key]
word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)[:1000]


# Extrayendo las palabras más frecuentes y sus vectores
words_top_1000 = [word for word, _ in word_freq]
word_vectors_top_1000 = model.wv[words_top_1000]

# Reducción de dimensionalidad utilizando PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(word_vectors_top_1000)

# Reducción de dimensionalidad utilizando t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(word_vectors_top_1000)

# Reducción de dimensionalidad utilizando SVD
svd = TruncatedSVD(n_components=2)
svd_result = svd.fit_transform(word_vectors_top_1000)



def show_pca():
    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    for i, word in enumerate(words_top_1000):
        plt.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]), fontsize=8)
        
    plt.title('PCA - Top 1000 palabras más frecuentes')
    plt.show()

def show_tsne():
    # Graficar los resultados de t-SNE
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    for i, word in enumerate(words_top_1000):
        plt.annotate(word, xy=(tsne_result[i, 0], tsne_result[i, 1]), fontsize=8)
        
    plt.title('t-SNE - Top 1000 palabras más frecuentes')
    plt.show()

def show_svd():
    # Graficar los resultados de SVD
    plt.figure(figsize=(10, 6))
    plt.scatter(svd_result[:, 0], svd_result[:, 1], alpha=0.5)
    for i, word in enumerate(words_top_1000):
        plt.annotate(word, xy=(svd_result[i, 0], svd_result[i, 1]), fontsize=8)
        
    plt.title('SVD - Top 1000 palabras más frecuentes')
    plt.show()


while True:
    print("\nMenú:")
    print("1. mostrar reduccion pca")
    print("2. mostrar reduccion tsne")
    print("3. mostrar reduccion svd")
    print("0. Salir")
    
    opcion = input("Seleccione una opción: ")
    
    if opcion == "1":
        show_pca()
    elif opcion == "2":
        show_tsne()
    elif opcion == "3":
        show_svd()
    elif opcion == "0":
        print("adios :c")
        break
    else:
        print("Opción inválida. Por favor, seleccione una opción válida.")


"""

¿Se guardan las relaciones semánticas? si o no y ¿por que?

sí ya que los modelos de palabras a vectores capturan relaciones semánticas
entre las palabras debido a cómo se entrenan para representar la estructura
semántica del lenguaje en el espacio vectorial.

¿Qué metodo de reducción de dimensionalidad consideras que es mejor?

pienso que cada metodo tiene sus puntos buenos o malos por ejemplo en pca es
más rápido y eficiente en grandes conjuntos de datos a comparación de tsne
que es lento en grandes conjuntos de datos, pero a diferencia de pca este sí
captura estructuras no lineales.

"""

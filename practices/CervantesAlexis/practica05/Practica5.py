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

# +
import nltk

nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
# -

from gensim.models import word2vec


def load_model(model_path: str):
    try:
        return word2vec.Word2Vec.load(model_path)
    except:
        print(f"[WARN] Model not found in path {model_path}")
        return None


model = load_model('models/word2vec/' + "practica5-eswiki.model")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
vectores_pca = pca.fit_transform(model.wv.vectors)

# +
import matplotlib.pyplot as plt

palabras = [model.wv.index_to_key[i] for i in range(1000)]

plt.figure(figsize=(10, 10))
plt.scatter(vectores_pca[:1000][:, 0], vectores_pca[:1000][:, 1])

for word, (x, y) in zip(palabras, vectores_pca[:1000]):
    plt.text(x, y, word, fontsize=10)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Reducción PCA")
plt.show()
# -

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
model_tsne = tsne.fit_transform(model.wv.vectors)


# +
most_frequent_words = [model.wv.index_to_key[i] for i in range(1000)]
model_tsne_most_frequent = model_tsne[0:1000]

plt.figure(figsize=(10, 10))
plt.scatter(model_tsne_most_frequent[:, 0], model_tsne_most_frequent[:, 1])

for word, (x, y) in zip(most_frequent_words, model_tsne_most_frequent):
    plt.text(x, y, word, fontsize=10)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("1000 most frequent words in model_tsne")
plt.show()
# -

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2)
vectores_svd = svd.fit_transform(model.wv.vectors)


# +
import matplotlib.pyplot as plt

palabras = [model.wv.index_to_key[i] for i in range(1000)]

plt.figure(figsize=(10, 10))
plt.scatter(vectores_svd[:1000][:, 0], vectores_svd[:1000][:, 1])

for word, (x, y) in zip(palabras, vectores_svd[:1000]):
    plt.text(x, y, word, fontsize=10)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Reducción SVD")
plt.show()
# -

# ## ¿Se guardan las relaciones semánticas? si o no y ¿porqué?
# ### Pues más o menos. Podemos asumir algunas relaciones entre las palabras como que las personas se concentran un lugar, conceptos relacionados con medidas en otro, e incluso temas como política, entretenimiento pero no se puede garantizar una preservación completa de las relaciones semánticas dado que dejamos de lado la mayoría de dimensiones que guardaban información importante dado que eran densos.
# ## ¿Qué método de reducción de dimensaionalidad consideras que es mejor?
# ### PCA porque siento que puedo percibir una relación entre palabras cercanas ligeramente mejor que en SVD y tarda mucho menos en entrenar que TSNE



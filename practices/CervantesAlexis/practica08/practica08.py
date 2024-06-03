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

# + colab={"base_uri": "https://localhost:8080/"} id="HaF1BP689XEW" outputId="5e7c1cf0-6fe4-4971-f4ba-f6490567a68e"
import nltk
nltk.download('punkt')
from nltk import ngrams
from collections import Counter, defaultdict
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# + id="r4D0_fiI9YBP"
import re

def preprocess_corpus(corpus: list[str]) -> list[str]:
    """Función de preprocesamiento

    Agrega tokens de inicio y fin, normaliza todo a minusculas
    """
    preprocessed_corpus = []
    for sent in corpus:
      result = [word.lower() for word in sent]
      # Al final de la oración
      result[-1] = "<EOS>"
      result.insert(0, "<BOS>")
      preprocessed_corpus.append(result)
    return preprocessed_corpus


# + id="u31cDD4rAk5y"
def get_words_freqs(corpus: list[list[str]]):
    words_freqs = {}
    for sentence in corpus:
        for word in sentence:
            words_freqs[word] = words_freqs.get(word, 0) + 1
    return words_freqs


# + id="5n7FSjq59eat"
UNK_LABEL = "<UNK>"
def get_words_indexes(words_freqs: dict) -> dict:
    result = {}
    for idx, word in enumerate(words_freqs.keys()):
        # Happax legomena happends
        if words_freqs[word] == 1:
            # Temp index for unknowns
            result[UNK_LABEL] = len(words_freqs)
        else:
            result[word] = idx

    return {word: idx for idx, word in enumerate(result.keys())}, {idx: word for idx, word in enumerate(result.keys())}


# + id="msyZ0yLV9iOE"
with open('2000-0.txt', 'r', encoding='utf-8-sig') as file:
    text = file.read()
    text = text.replace('\n', ' ')
    text = text.replace(',', '')
sentences = sent_tokenize(text)
sentences = [word_tokenize(sent) for sent in sentences]

# + id="kZEEUnwR-XZX"
corpus = preprocess_corpus(sentences)

# + id="fasUemvR-u80"
words_freqs = get_words_freqs(corpus)

# + id="gTsd9-OhAavN"
words_indexes, index_to_word = get_words_indexes(words_freqs)


# + id="fHsHdlKUAuFJ"
def get_word_id(words_indexes: dict, word: str) -> int:
    unk_word_id = words_indexes[UNK_LABEL]
    return words_indexes.get(word, unk_word_id)


# + id="R9fejIEjCiZ1"
def get_train_test_data(corpus: list[list[str]], words_indexes: dict, n: int) -> tuple[list, list]:
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append([get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)])
            y_train.append([get_word_id(words_indexes, w3)])
    return x_train, y_train


# + id="jBvXnmWxCKto"
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# + id="6i6TomoICxFt"
# Setup de parametros
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2
BATCH_SIZE = 256
H = 100
torch.manual_seed(19)
# Tamaño del Vocabulario
V = len(words_indexes)

# + id="qCpPtlr5C6GJ"
x_train, y_train = get_train_test_data(corpus, words_indexes, n=3)

# + id="DFgSp_qqDIVh"
train_set = np.concatenate((x_train, y_train), axis=1)
# partimos los datos de entrada en batches
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE)


# + id="j7VZ2t0ADMkP"
# Trigram Neural Network Model
class TrigramModel(nn.Module):
    """Clase padre: https://pytorch.org/docs/stable/generated/torch.nn.Module.html"""

    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        # x': concatenation of x1 and x2 embeddings   -->
        #self.embeddings regresa un vector por cada uno de los índices que se les pase como entrada. view() les cambia el tamaño para concatenarlos
        embeds = self.embeddings(inputs).view((-1,self.context_size * self.embedding_dim))
        # h: tanh(W_1.x' + b)  -->
        out = torch.tanh(self.linear1(embeds))
        # W_2.h                 -->
        out = self.linear2(out)
        # log_softmax(W_2.h)      -->
        # dim=1 para que opere sobre renglones, pues al usar batchs tenemos varios vectores de salida
        log_probs = F.log_softmax(out, dim=1)

        return log_probs


# + id="V4cJ2xVYSgE6" colab={"base_uri": "https://localhost:8080/"} outputId="5740b547-58b4-429e-c920-c15c7701454a"
from google.colab import drive
drive.mount('/content/drive')

# + colab={"base_uri": "https://localhost:8080/"} id="I10iXZjMR6EQ" outputId="8c95fdd3-1cd2-4204-814b-0a18c38f7dd1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Pérdida. Negative log-likelihood loss
loss_function = nn.NLLLoss()

# 2. Instanciar el modelo y enviarlo a device
model = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H).to(device)

# 3. Optimización. ADAM optimizer
optimizer = optim.Adam(model.parameters(), lr = 2e-3)

# ------------------------- TRAIN & SAVE MODEL ------------------------
EPOCHS = 3
for epoch in range(EPOCHS):
    st = time.time()
    print("\n--- Training model Epoch: {} ---".format(epoch))
    for it, data_tensor in enumerate(train_loader):
        # Mover los datos a la GPU
        context_tensor = data_tensor[:,0:2].to(device)
        target_tensor = data_tensor[:,2].to(device)

        model.zero_grad()

        # FORWARD:
        log_probs = model(context_tensor)

        # compute loss function
        loss = loss_function(log_probs, target_tensor)

        # BACKWARD:
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print("Training Iteration {} of epoch {} complete. Loss: {}; Time taken (s): {}".format(it, epoch, loss.item(), (time.time()-st)))
            st = time.time()

    # saving model
    model_path = 'drive/MyDrive/LM_neuronal/model_gpu_{}.dat'.format(epoch)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved for epoch={epoch} at {model_path}")


# + id="nsLGxHshDZWR"
def get_model(path: str) -> TrigramModel:
    model_loaded = TrigramModel(V, EMBEDDING_DIM, CONTEXT_SIZE, H)
    model_loaded.load_state_dict(torch.load(path))
    model_loaded.eval()
    return model_loaded


# + id="tbsZmIt7EI-M"
PATH = "drive/MyDrive/LM_neuronal/model_gpu_2.dat"

# + id="-JGupPyoEN5s"
model = get_model(PATH)
W1 = "dulcinea"
W2 = "del"

IDX1 = get_word_id(words_indexes, W1)
IDX2 = get_word_id(words_indexes, W2)

#Obtenemos Log probabidades p(W3|W2,W1)
probs = model(torch.tensor([[IDX1,  IDX2]])).detach().tolist()

# + id="upcjO7srESZm" colab={"base_uri": "https://localhost:8080/"} outputId="adb3bb96-278d-47ed-8b20-1f9df1a22ac7"
model_probs = {}
for idx, p in enumerate(probs[0]):
  model_probs[idx] = p

# Sort:
model_probs_sorted = sorted(((prob, idx) for idx, prob in model_probs.items()), reverse=True)

# Printing word  and prob (retrieving the idx):
topcandidates = 0
for prob, idx in model_probs_sorted:
  #Retrieve the word associated with that idx
  word = index_to_word[idx]
  print(idx, word, prob)

  topcandidates += 1

  if topcandidates > 100:
    break

# + id="PLGiEpTCEiQ3"
index_to_word.get(model_probs_sorted[0][0])


# + id="LmucOrOYEz9X"
def get_likely_words(model: TrigramModel, context: str, words_indexes: dict, index_to_word: dict, top_count: int=10) -> list[tuple]:
    model_probs = {}
    words = context.split()
    idx_word_1 = get_word_id(words_indexes, words[0])
    idx_word_2 = get_word_id(words_indexes, words[1])
    probs = model(torch.tensor([[idx_word_1, idx_word_2]])).detach().tolist()

    for idx, p in enumerate(probs[0]):
        model_probs[idx] = p

    # Strategy: Sort and get top-K words to generate text
    return sorted(((prob, index_to_word[idx]) for idx, prob in model_probs.items()), reverse=True)[:top_count]


# + colab={"base_uri": "https://localhost:8080/"} id="pyblo91kE4Fb" outputId="e5770738-cdbc-4067-ee82-79dcfbc7c42b"
sentence = "dulcinea del"
get_likely_words(model, sentence, words_indexes, index_to_word, 3)

# + id="C4-MmKXAE8TX"
from random import randint

def get_next_word(words: list[tuple[float, str]]) -> str:
    # From a top-K list of words get a random word
    return words[randint(0, len(words)-1)][1]


# + id="pR7Qvbu0nLnU"
import numpy as np

def get_next_word(words: list[tuple[float, str]]) -> str:
    lam = min(len(words) - 1, max(1, len(words) ))//2

    index = np.random.poisson(lam)
    index = min(index, len(words) - 1)

    return words[index][1]


# + id="mHhHMZzDFDY4"
MAX_TOKENS = 30
TOP_COUNT = 4
def generate_text(model: TrigramModel, history: str, words_indexes: dict, index_to_word: dict, tokens_count: int=0) -> None:
    next_word = get_next_word(get_likely_words(model, history, words_indexes, index_to_word, top_count=TOP_COUNT))
    print(next_word, end=" ")
    tokens_count += 1
    if tokens_count == MAX_TOKENS or next_word == "<EOS>":
        return
    generate_text(model, history.split()[1]+ " " + next_word, words_indexes, index_to_word, tokens_count)


# + colab={"base_uri": "https://localhost:8080/"} id="IBCSTD03TwjA" outputId="321d9a12-b9d6-4078-ec54-01287613e14e"
for i in range(3):
  sentence = "caballero que"
  print(sentence, end=" ")
  generate_text(model, sentence, words_indexes, index_to_word)
  print("\n")

# + [markdown] id="NDSQ3278lG2N"
# link al modelo: https://drive.google.com/drive/folders/19HzTRLuj8k_-XEkxrruiOl3hnxWMZSq_?usp=sharing

# + [markdown] id="F9OeJjiYnyKN"
# ### Usamos una distribución poisson en lugar de una distribución uniforme para preferir la bolsa más probable pero no ciclarnos en las palabras más probables del corpus
# ### La verdad no se ve superior pero en los ejemplos que vi más bien parece una frase compleja a la que le falta contexto

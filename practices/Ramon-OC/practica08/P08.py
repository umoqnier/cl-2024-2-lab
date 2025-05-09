# -*- coding: utf-8 -*-
"""Untitled16.ipynb

Automatically generated by Colab.

Original file is located at
"""

!pip install optim

import re
import time
from collections import Counter, defaultdict

import nltk
from nltk import ngrams
from nltk.corpus import reuters

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.data import DataLoader, TensorDataset

import requests
from collections import deque
import torch.nn as nn
import torch.optim as optim

url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
response = requests.get(url)
text = response.text

lines = text.splitlines()
no_empty_lines = [line.strip() for line in lines if line.strip()]
no_special_chars = [re.sub(r'[^\w\s]', '', line) for line in no_empty_lines]
lowercased = [line.lower() for line in no_special_chars]
dataset = [sentence.split() for sentence in lowercased]

preprocessed_corpus = []
for sent in dataset:
    result = [word.lower() for word in sent]
    result.append("<EOS>")
    result.insert(0, "<BOS>")
    preprocessed_corpus.append(result)
corpus = preprocessed_corpus

words_freqs = {}
for sentence in corpus:
    for word in sentence:
        words_freqs[word] = words_freqs.get(word, 0) + 1

unk = "<UNK>"
result = {}
for idx, word in enumerate(words_freqs.keys()):
    if words_freqs[word] == 1:
        result[unk] = len(words_freqs)
    else:
        result[word] = idx

words_indexes = {word: idx for idx, word in enumerate(result.keys())}
index_to_word = {idx: word for idx, word in enumerate(result.keys())}

def get_word_id(words_indexes, word):
    return words_indexes.get(word, words_indexes[unk])

def get_train_test_data(corpus, words_indexes, n):
    x_train = []
    y_train = []
    for sent in corpus:
        n_grams = ngrams(sent, n)
        for w1, w2, w3 in n_grams:
            x_train.append([get_word_id(words_indexes, w1), get_word_id(words_indexes, w2)])
            y_train.append([get_word_id(words_indexes, w3)])
    return x_train, y_train


x_train, y_train = get_train_test_data(corpus, words_indexes, n=3)

embedding_dim = 200
context_size = 2
batch_size = 256
hidden = 100 # Dimsension de capa oculta
torch.manual_seed(19)


class TrigramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, h):
        super(TrigramModel, self).__init__()
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, h)
        self.linear2 = nn.Linear(h, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((-1, self.context_size * self.embedding_dim))
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

x_train_tensor = torch.tensor(x_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze()

train_set = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

loss_function = nn.NLLLoss()
model = TrigramModel(len(words_indexes), embedding_dim, context_size, hidden)
optimizer = optim.Adam(model.parameters(), lr=2e-3)

EPOCHS = 1
for epoch in range(EPOCHS):
    st = time.time()
    for it, (context_tensor, target_tensor) in enumerate(train_loader):
        model.zero_grad()
        log_probs = model(context_tensor)
        loss = loss_function(log_probs, target_tensor)
        loss.backward()
        optimizer.step()

from random import randint

def get_likely_words(model, context, word_indices, index_to_word, top_count=10):
    model_probs = {}
    words = context.split()
    idx_word_1 = word_indices[words[0]]
    idx_word_2 = word_indices[words[1]]
    probs = model(torch.tensor([[idx_word_1, idx_word_2]])).detach().tolist()

    for idx, p in enumerate(probs[0]):
        model_probs[idx] = p

    return sorted(((prob, index_to_word[idx]) for idx, prob in model_probs.items()), reverse=True)[:top_count]

def get_next_word(words):
    return words[randint(0, len(words)-1)][1]


model_loaded = TrigramModel(len(words_indexes), embedding_dim, context_size, hidden)
model_loaded.eval()
model = model_loaded

for _ in range(3):
    history = "<BOS> it"
    tokens_count = 0
    max_tokens = 10
    top_count = 5

    while True:
        next_word = get_next_word(get_likely_words(model, history, words_indexes, index_to_word, top_count=top_count))
        print(next_word, end=" ")
        tokens_count += 1
        if tokens_count == max_tokens or next_word == "<EOS>":
            break
        history = history.split()[1] + " " + next_word

    print("\n")
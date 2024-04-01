from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 4]
import numpy as np
import random

# ---
# Ejercico 2
# ---

# Lenguaje ifiil

def rword():
    na = list(range(102,111))
    word = ''
    for _ in range(0,random.randint(2,10)):
        word += chr(na[random.randint(0,len(na)-1)])
    return(word)

vocabulary = Counter([rword() for _ in range(0,2000)])

def get_frequencies(vocabulary: Counter, n: int) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]

def plot_frequencies(frequencies: list, title="Freq of words", log_scale=False):
    x = list(range(1, len(frequencies)+1))
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)
    plt.show()

print("Lenguaje ifiil")
print("Oración ejemplo: "+" ".join([rword() for _ in range(0,4)]))
print(vocabulary)

frequencies = get_frequencies(vocabulary, 100)
plot_frequencies(frequencies)

# La ley de Zipf se cumple, pues al tratarse de un lenguaje aleatorio, hay menos posibilidades para palabras cortas, y por lo tanto más ocurrencias
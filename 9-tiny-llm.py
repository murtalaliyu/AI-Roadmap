import math, random
from story import words, the_last_bus

vocab = sorted(set(the_last_bus))
W = {w: [0.0] * len(vocab) for w in vocab}

def chances(a):
    e = [math.exp(s) for s in W[a]]
    total = sum(e)
    return [v / total for v in e]

pairs = list(zip(the_last_bus, the_last_bus[1:]))

epochs = 500
learning_rate = 0.1

for epoch in range(epochs):
    print("epoch:", epoch + 1)
    for a, b in pairs:
        P = chances(a)
        for c in range(len(vocab)):
            W[a][c] -= learning_rate * (P[c] - (vocab[c] == b))

random.seed(1)
while True:
    while True:
        w  = input("\nEnter the first word: ")
        if w in vocab:
            break
        print(w, "is not in the vocabulary. Try again.")
    num_words = int(input("Enter the number of generated words: "))

    for k in range(num_words):
        w = random.choices(vocab, chances(w))[0]
        print(w, end=' ')

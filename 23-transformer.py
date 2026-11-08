import numpy as np
from pets import train, test, words, pets

r = np.random.default_rng(31)

Wq, Wk, Wv = r.normal(0, 0.1, (3, len(words), 8))

def look(s):                                        # how much "it" looks at each word
    i = [words.index(w) for w in s]
    q = Wq[i[s.index('it')]]                        # the query of "it"
    a = np.exp(Wk[i] @ q)
    return i, q, a / a.sum()

for s, y in train * 40:                             # y: the pet "it" means
    i, q, a = look(s)
    K = Wk[i]
    V = Wv[i]
    p = np.exp(a @ V)
    p /= p.sum()
    p[y] -= 1
    g = a * (V @ p - a @ V @ p)                     # backpropagation
    np.add.at(Wv, i, -np.outer(a, p))               # values
    np.add.at(Wk, i, -np.outer(g, q))               # keys
    Wq[i[s.index('it')]] -= K.T @ g                 # query

w = [look(s)[2][s.index(pets[y])] for s, y in test]
print(f'{np.mean(w):.0%} on the right word')

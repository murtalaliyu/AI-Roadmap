# Retrieval, Augmentation, Generation
import re
import numpy as np
from contract import pieces, questions, common

def words(t):                                       # Lower-case words, no common ones
    ws = re.findall('[a-z]+', t.lower())
    return [w for w in ws if w not in common]

vocab = sorted(set(sum(map(words, pieces), [])))

def embed(t):                                       # text -> list of numbers
    ws = words(t)                                   # count every vocabulary word
    v = np.array([ws.count(w) for w in vocab])
    return v / max(np.linalg.norm(v), 1e-9)

E = np.array([embed(p) for p in pieces])

def retrieve(q, k=2):                               # step 1: closest pieces
    return np.argsort(-(E @ embed(q)))[:k]          # cosine similarity scores

hit = [retrieve(q)[0] == a for q, a in questions]

if __name__ == '__main__':
    print(sum(hit), 'of', len(hit), 'questions')

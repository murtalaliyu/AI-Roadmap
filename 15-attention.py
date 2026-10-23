import numpy as np

w = ("the animal didn't cross the street because it was tired").split()

# made-up features (columns): thing, alive, place, pronoun
f = {"animal": [1, 1, 0, 0],
     "street": [1, 0, 1, 0],
     "it":     [0, 0, 0, 1]}

X = np.array([f.get(t, [0] * 4) for t in w])

# Query: What am I looking for?
# Key: What can I offer?
# Value: What do I pass on?

Wq = np.array([[0, 0],      # thing
               [0, 0],      # alive
               [0, 0],      # place
               [2, 4]])     # pronoun

Wk = np.array([[1, 0],      # thing
               [0, 1],      # alive
               [0, 0],      # place
               [0, 0]])     # pronoun

Wv = np.eye(4)

Q, K, V = X @ Wq, X @ Wk, X @ Wv

S = Q @ K.T / np.sqrt(2)    # scaled dot products

A = np.exp(S) / np.exp(S).sum(1, keepdims=True)

new = A @ V     # weighted sum

for t, a in zip(w, A[7]):
    print(t, a.round(2))

print("it now:", new[7].round(2))

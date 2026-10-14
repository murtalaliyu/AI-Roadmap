import math
from digits import train, test  # 16x16 images

W = [[0.0] * 256 for d in range(10)]    # 10 neurons

def see(x):
    s = [sum(w * p for w, p in zip(W[d], x)) for d in range(10)]    # 10 scores
    e = [math.exp(v - max(s)) for v in s]   # for softmax stability, prevents overflow/underflow
    return [v / sum(e) for v in e]  # softmax

for x, y in train * 6:
    P = see(x)
    for d in range(10):
        g = (P[d] - (d == y)) * 0.01    # blame
        W[d] = [w - g*p for w, p in zip(W[d], x)]

right = [see(x).index(max(see(x))) == y for x, y in test]

print(sum(right), 'of', len(test))

import numpy as np
from digits_cnn import X, y, T, t, win              # 8x8 digits

r = np.random.default_rng(32)

K = r.normal(0, 0.5, (9, 8))
W = np.zeros((72, 10))

# filters slide, ReLU, pooling, layer
def cnn(X):
    Q = win(X, (3, 3), (1, 2)).reshape(-1, 9)       # apply sliding window
    B = (Q @ K).clip(0).reshape(-1, 3, 2, 3, 2, 8)  # ReLU
    M = B.max((2, 4), keepdims=True)                # max pooling
    S = B == M
    S = S / S.sum((2, 4), keepdims=1)
    return Q, S * (M > 0), M, M.reshape(-1, 72) @ W

for b in r.integers(0, len(X), (10000, 32)):
    Q, S, M, Z = cnn(X[b])
    p = np.exp(Z)
    E = p / p.sum(1)[:, None] - np.eye(10)[y[b]]    # softmax, error
    D = S * (E @ W.T).reshape(M.shape)              # backprop
    W -= 0.003 * M.reshape(-1, 72).T @ E            # update classifier weights
    K -= 0.003 * Q.T @ D.reshape(-1, 8)             # update filter weights

print(sum(cnn(T)[3].argmax(1) == t), 'of', len(t))

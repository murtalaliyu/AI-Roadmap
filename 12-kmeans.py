import numpy as np
from photos import X                        # 600 photos

k = 3                                       # number of groups
rng = np.random.default_rng(17)
best = np.inf                               # no try yet

for start in range(10):                     # ten random starts
    C = X[rng.choice(len(X), k, replace=False)]
    while True:
        old = C.copy()                      # remember centers
        d = ((X[:, None] - C) ** 2).sum(2)
        g = d.argmin(1)                     # nearest center
        
        for j in range(k):
            C[j] = X[g == j].mean(0)
        if (C == old).all():
            break

    s = d.min(1).sum()                      # total spread
    if s < best:
        best, keep = s, g
print(np.bincount(keep))                    # photos per group

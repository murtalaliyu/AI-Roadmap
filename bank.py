import numpy as np

rng = np.random.default_rng(13)
income = rng.integers(20, 151, size=500)
missed = rng.integers(0, 6, size=500)

X = np.column_stack((income, missed))
y = ((income >= 50) & (missed <= 1)).astype(int)
#print(X, y)

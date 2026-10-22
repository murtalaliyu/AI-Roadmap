# %%
import numpy as np
import digits
import itertools as it
import matplotlib.pyplot as plt

# %%
X = np.array([x for x, _ in digits.train])

# %%
r = np.random.default_rng(2)

# %%
M, V = [0]*4, [0]*4

# %%
W = [r.normal(0, a ** -0.5, (a + 1, b)) for a, b in zip([256, 64, 2, 64], [64, 2, 64, 256])]

# %%
layer = lambda a, w: np.tanh(a @ w[:-1] + w[-1])

# %%
for b in r.integers(0, 3000, (8000, 256)):
    A = list(it.accumulate(W, layer, initial=X[b]))
    d = A[4] - (2 * A[0] - 1)                           # compare
    for i in 3, 2, 1, 0:                                # send it back
        g = np.r_[A[i].T @ d, [d.sum(0)]]               # slope
        d = d @ W[i][:-1].T * (1 - A[i] ** 2)
        M[i] = 0.9 * M[i] + 0.1 * g                     # Adam-style
        V[i] = 0.999 * V[i] + 0.001 * g * g
        W[i] -= 0.003 * M[i] / (V[i] ** 0.5 + 1e-8)

# %%
latent = np.array([[0.29, -0.2]])
new = layer(layer(latent, W[2]), W[3])
image = ((new.reshape(16, 16) + 1) / 2).clip(0, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=1)
plt.title(f"Generated from latent input {latent[0].tolist()}")
plt.axis("off")
plt.show()

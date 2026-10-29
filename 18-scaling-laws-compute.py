import numpy as np

rng = np.random.default_rng(26)
j, ph = np.arange(1, 129), rng.uniform(0, 7, 128)

f = lambda x: np.sin(np.outer(x, j) + ph) @ (1/j)

X = rng.uniform(-3, 3, 4000)
Y = f(X)                                                # test

for n in (4, 8, 16, 32, 64, 128, 256):                  # neurons
    p, ops = np.linspace(-3, 3, n), 0
    W = 100 * np.array(([p**0, -p, 0*p]))
    M = 0 * W

    h = lambda x: np.tanh(np.outer(x, W[0]) + W[1])

    for step in range(8000):
        x = rng.uniform(-3, 3, 256)
        a = h(x)
        e = (a @ W[2] - f(x)) / 256
        g = np.outer(e, W[2]) * (1 - a * a)
        M = 0.8 * M + [x @ g, g.sum(0), a.T @ e]
        W -= M / n / 2
        ops += 18 * n * 256

    print(n, ops, ((h(X) @ W[2] - Y)**2).mean())
    
import numpy as np

r = np.random.default_rng(32)

F = [0xE8C62E, 0xE210C4, 0x1F0BA0F, 0xF83A0F,
     0x1087E31, 0xF83C3F, 0xE8BC2E, 0x42221F,
     0xE8BA2E, 0xE87A2E]                               # 10 digits, 5x5 pixels

P = (np.array(F)[:, None] >> np.arange(25)) & 1
P = np.pad(P.reshape(10, 5, 5), ((0,), (3,), (3,)))

win = np.lib.stride_tricks.sliding_window_view

P = win(P, (8, 8), (1, 2))                              # 16 spots in 8x8

def draw(n):                                            # n digits, random spot, 5% noise
    y, a, b = r.integers(0, (10, 4, 4), (n, 3)).T
    x = P[y, a, b] ^ (r.random((n, 8, 8)) < 0.05)
    return 2 * x - 1, y                                 # ink 1, paper -1

X, y = draw(10000)                                      # to learn from
T, t = draw(500)                                        # to test

new = [not (X == x).all((1, 2)).any() for x in T]
T, t = T[new], t[new]                                   # only digits never seen

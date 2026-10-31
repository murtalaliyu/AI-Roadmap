import numpy as np

r = np.random.default_rng(29)

def svr(x, y, e=0.35):                              # tolerance band: +- e
    w, b = 0.0, 0.0
    for t in range(1, 50001):
        d = y - (w*x + b)                           # errors
        s = np.sign(d) * (abs(d) > e)               # outside
        g = s / (0.3 * t)                           # the same push for all
        w = w - w / t + g @ x
        b = b + g.sum()
    return w, b

x = r.uniform(-2, 2, 30)                            # made up inputs
y = 0.5 * x + 0.3 + r.normal(0, 0.35, 30)           # returns

w, b = svr(x, y)                                    # the function w x + b
d = abs(y - (w*x + b))                              # errors after training

print(w, b)
print(np.sum(d < 0.35), 'of 30 in the band')
print(np.polyfit(x, y, 1))                          # classic regression

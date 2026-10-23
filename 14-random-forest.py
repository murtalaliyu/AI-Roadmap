import numpy as np
from bank import X, y               # 500 customers

gini = lambda y: 2 * y.sum() * (1 - y.mean())
rng = np.random.default_rng(20)     # part 20

def grow(i):                        # i: customers here
    best, F, f = gini(y[i]), None, rng.integers(2)
    for t in np.unique(X[i, f])[:-1]:
        L = X[i, f] > t             # answer is yes
        g = gini(y[i][L]) + gini(y[i][~L])
    
        if g < best:
            best, F, T, S = g, f, t, L
    
    if F is None:
        return lambda x: y[i].mean() > 0.5
    
    a, b = grow(i[S]), grow(i[~S])
    return lambda x: a(x) if x[F] > T else b(x)

V = np.sum([[*map(grow(rng.choice(400, 400)), X)] for _ in range(100)], 0)  # 100 trees
print(((V > 50) == y)[400:].mean(), V[400])

model = grow(np.arange(400))
while True:
    try:
        income = int(input("Income (20-150): "))
        missed = int(input("Missed payments (0-5): "))
        if not 20 <= income <= 150 or not 0 <= missed <= 5:
            raise ValueError
        break
    except ValueError:
        print("Enter an income from 20 to 150 and missed payments from 0 to 5.")

customer = np.array([income, missed])
print("Predicted class:", int(model(customer)))

import numpy as np
from bank import X, y       # 500 customers

gini = lambda y: 2 * y.sum() * (1 - y.mean())

def grow(i, d):             # i: customers here
    best, F, v = gini(y[i]), None, y[i].mean() > 0.5
    for f in (0, 1):        # income, missed
        for t in np.unique(X[i, f])[:-1]:
            L = X[i, f] > t     # answer is yes
            g = gini(y[i][L]) + gini(y[i][~L])

            if g < best:
                best, F, T, S = g, f, t, L
    if d == 0 or F is None:
        return lambda x: v
    a, b = grow(i[S], d - 1), grow(i[~S], d - 1)
    return lambda x: a(x) if x[F] > T else b(x)

for d in (2, np.inf):       # 2 or no limit
    p = [*map(grow(np.arange(400), d), X)] == y
    print(d, p[:400].mean(), p[400:].mean())

model = grow(np.arange(400), 2)
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

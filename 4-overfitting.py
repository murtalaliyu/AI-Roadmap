xs = [-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9]
ys = [0.9, 0.6, 0, -0.1, 0.1, 0.4, 0.4, 0.6]

hidden = [(-0.8, 0.6), (-0.5, 0.2), (-0.2, 0.1), (0, 0), (0.3, 0.1), (0.5, 0.2), (0.8, 0.6)]

K = 10
w = [0.0] * K
pred = lambda x: sum(w[k]*x**k for k in range(K))
lr = 0.3

for epoch in range(5000):
    train = 0
    for x, y in zip(xs, ys):
        err = pred(x) - y
        train += err**2
        for k in range(K):
            w[k] -= lr * err * x**k
    test = sum((pred(x)-y)**2 for x, y in hidden)

print("train:", train)
print("test", test)

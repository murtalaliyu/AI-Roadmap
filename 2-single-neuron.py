import math

xs = [[1,2], [2,1], [2,3], [3,2], [6,7], [7,6], [7,8], [8,7]]
ys = [0,0,0,0,1,1,1,1]

w1, w2, b = 0.0, 0.0, 0.0
lr = 0.1

for epoch in range(200):
    for x, y in zip(xs, ys):
        z = w1*x[0] + w2*x[1] + b
        p = 1 / (1 + math.exp(-z))
        err = p - y
        w1 -= lr * err * x[0]
        w2 -= lr * err * x[1]
        b -= lr * err

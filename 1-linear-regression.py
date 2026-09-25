import matplotlib.pyplot as plt

xs = [1,2,3,4,5,6,7,8]
ys = [2.9, 3.4, 4.9, 4.7, 6.2, 6.9, 7.3, 8.6]

w, b = 0.0, 0.0
lr = 0.01

for epoch in range(1000):
    dw, db = 0.0, 0.0
    for x, y in zip(xs, ys):
        pred = w * x + b    # linear equation: y = wx + b
        err = pred - y
        dw += 2 * err * x / len(xs)     # gradient of the loss with respect to w — how much to nudge the slope to reduce error
        db += 2 * err / len(xs)         # gradient of the loss with respect to b — how much to nudge the intercept to reduce error
    # gradient descent
    w -= lr * dw;
    b -= lr * db;

# Plot prediction
preds = [w * x + b for x in xs]
plt.scatter(xs, ys, label="data")
plt.plot(xs, preds, color="red", label=f"fit: y = {w:.2f}x + {b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

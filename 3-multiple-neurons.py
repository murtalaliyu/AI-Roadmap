import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

DATA = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

def sig(z):
    return 1 / (1 + math.exp(-z))

# weights:
# w[0], w[1], w[2] -> hidden neuron a
# w[3], w[4], w[5] -> hidden neuron b
# w[6], w[7], w[8] -> output neuron
w = [0.3, -0.8, 0.5, -0.2, 0.9, 0.1, 0.7, -0.6, 0.2]
lr = 0.5
epochs = 20000
N = 50

history = [w.copy()]

for epoch in range(epochs):
    for x0, x1, y in DATA:
        # forward pass
        a = sig(w[0] * x0 + w[1] * x1 + w[2])
        b = sig(w[3] * x0 + w[4] * x1 + w[5])
        p = sig(w[6] * a + w[7] * b + w[8])

        # output loss derivative for sigmoid output
        # this is the standard sigmoid derivative term
        d_out = (p - y) * p * (1 - p)

        # backpropagate through hidden neurons.
        # use the derivative of the sigmoid to propagate that error backward.
        # The hidden neurons are connected to the output neuron through weights w[6] and w[7], so the error must be scaled by those weights.
        d_a = d_out * w[6] * a * (1 - a)
        d_b = d_out * w[7] * b * (1 - b)

        # gradients for all weights
        grad = [
            d_a * x0,   # w0
            d_a * x1,   # w1
            d_a,        # w2
            d_b * x0,   # w3
            d_b * x1,   # w4
            d_b,        # w5
            d_out * a,  # w6
            d_out * b,  # w7
            d_out       # w8
        ]

        # gradient descent update
        for i in range(9):
            w[i] -= lr * grad[i]

    if (epoch + 1) % N == 0:
        history.append(w.copy())

# Final output for printing
for x0, x1, y in DATA:
    a = sig(w[0] * x0 + w[1] * x1 + w[2])
    b = sig(w[3] * x0 + w[4] * x1 + w[5])
    p = sig(w[6] * a + w[7] * b + w[8])
    print((x0, x1), "target =", y, "prediction =", round(p, 4))

# Live visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_points, ax_surface = axes

xs = np.linspace(0, 1, 60)
ys = np.linspace(0, 1, 60)
xx, yy = np.meshgrid(xs, ys)

# scatter points for XOR dataset
x0_vals = [x for x, _, _ in DATA]
x1_vals = [y for _, y, _ in DATA]
labels = [target for _, _, target in DATA]
colors = ["tab:blue" if target == 0 else "tab:orange" for target in labels]

ax_points.scatter(x0_vals, x1_vals, c=colors, s=120, edgecolor="black", zorder=3)
ax_points.set_title("XOR training data")
ax_points.set_xlim(-0.1, 1.1)
ax_points.set_ylim(-0.1, 1.1)
ax_points.set_xlabel("x0")
ax_points.set_ylabel("x1")
ax_points.set_aspect("equal")

def predict_surface(weights):
    zz = np.zeros_like(xx, dtype=float)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x0 = xx[i, j]
            x1 = yy[i, j]
            a = sig(weights[0] * x0 + weights[1] * x1 + weights[2])
            b = sig(weights[3] * x0 + weights[4] * x1 + weights[5])
            p = sig(weights[6] * a + weights[7] * b + weights[8])
            zz[i, j] = p
    return zz

def update(frame):
    weights = history[frame]
    zz = predict_surface(weights)

    ax_surface.clear()
    ax_surface.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.95)
    ax_surface.scatter(x0_vals, x1_vals, c=colors, s=120, edgecolor="black", zorder=3)
    ax_surface.set_title(f"network output at epoch {frame * N}")
    ax_surface.set_xlim(-0.1, 1.1)
    ax_surface.set_ylim(-0.1, 1.1)
    ax_surface.set_xlabel("x0")
    ax_surface.set_ylabel("x1")
    ax_surface.set_aspect("equal")
    return []

ani = FuncAnimation(fig, update, frames=len(history), interval=60, blit=False, repeat=False)
plt.tight_layout()
plt.show()

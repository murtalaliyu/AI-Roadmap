import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [2.9, 3.4, 4.9, 4.7, 6.2, 6.9, 7.3, 8.6]

w, b = 0.0, 0.0
lr = 0.01
epochs = 1000
N = 20  # update the plot every N epochs

# Snapshot (w, b) every N epochs so we can animate gradient descent
history = [(w, b)]

for epoch in range(epochs):
    dw, db = 0.0, 0.0
    for x, y in zip(xs, ys):
        pred = w * x + b  # linear equation: y = wx + b
        err = pred - y
        dw += 2 * err * x / len(xs)  # gradient of the loss w.r.t. w
        db += 2 * err / len(xs)  # gradient of the loss w.r.t. b
    # gradient descent
    w -= lr * dw
    b -= lr * db

    if (epoch + 1) % N == 0:
        history.append((w, b))

fig, ax = plt.subplots()
ax.scatter(xs, ys, label="data")
(line,) = ax.plot([], [], color="red", label="")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
ax.set_ylim(min(ys) - 1, max(ys) + 1)
ax.legend(loc="upper left")
title = ax.set_title("")


def update(frame):
    w_i, b_i = history[frame]
    preds = [w_i * x + b_i for x in xs]
    line.set_data(xs, preds)
    line.set_label(f"fit: y = {w_i:.2f}x + {b_i:.2f}")
    ax.legend(loc="upper left")
    epoch_shown = frame * N
    title.set_text(f"epoch {epoch_shown} / {epochs}")
    return line, title


ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=False, repeat=False)
plt.show()

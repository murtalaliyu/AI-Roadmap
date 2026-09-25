import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

xs = [[1, 2], [2, 1], [2, 3], [3, 2], [6, 7], [7, 6], [7, 8], [8, 7]]
ys = [0, 0, 0, 0, 1, 1, 1, 1]

w1, w2, b = 0.0, 0.0, 0.0
lr = 0.1
epochs = 200
N = 5  # snapshot every N epochs for the animation

history = [(w1, w2, b)]

for epoch in range(epochs):
    for x, y in zip(xs, ys):
        z = w1 * x[0] + w2 * x[1] + b
        p = 1 / (1 + math.exp(-z))
        err = p - y
        w1 -= lr * err * x[0]
        w2 -= lr * err * x[1]
        b -= lr * err

    if (epoch + 1) % N == 0:
        history.append((w1, w2, b))

# Split points by class for coloring
xs0 = [x[0] for x, y in zip(xs, ys) if y == 0]
ys0 = [x[1] for x, y in zip(xs, ys) if y == 0]
xs1 = [x[0] for x, y in zip(xs, ys) if y == 1]
ys1 = [x[1] for x, y in zip(xs, ys) if y == 1]

fig, ax = plt.subplots()
ax.scatter(xs0, ys0, color="tab:blue", label="class 0")
ax.scatter(xs1, ys1, color="tab:orange", label="class 1")
(line,) = ax.plot([], [], color="red", label="")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.legend(loc="upper left")
title = ax.set_title("")


def decision_boundary(w1_i, w2_i, b_i, x_min=0, x_max=9):
    """Line where z = 0: w1*x1 + w2*x2 + b = 0 → x2 = -(w1*x1 + b) / w2."""
    if abs(w2_i) < 1e-8:
        return [], []
    x_line = [x_min, x_max]
    y_line = [-(w1_i * x + b_i) / w2_i for x in x_line]
    return x_line, y_line


def update(frame):
    w1_i, w2_i, b_i = history[frame]
    x_line, y_line = decision_boundary(w1_i, w2_i, b_i)
    line.set_data(x_line, y_line)
    line.set_label(f"boundary: {w1_i:.2f}x1 + {w2_i:.2f}x2 + {b_i:.2f} = 0")
    ax.legend(loc="upper left")
    epoch_shown = frame * N
    title.set_text(f"epoch {epoch_shown} / {epochs}")
    return line, title


ani = FuncAnimation(fig, update, frames=len(history), interval=80, blit=False, repeat=False)
plt.show()

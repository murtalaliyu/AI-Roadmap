import matplotlib.pyplot as plt

xs = [-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9]
ys = [0.9, 0.6, 0, -0.1, 0.1, 0.4, 0.4, 0.6]

hidden = [(-0.8, 0.6), (-0.5, 0.2), (-0.2, 0.1), (0, 0), (0.3, 0.1), (0.5, 0.2), (0.8, 0.6)]

K = 10
w = [0.0] * K
pred = lambda x: sum(w[k]*x**k for k in range(K))
lr = 0.3
x_axis = [i / 100 for i in range(-150, 151)]

plt.ion()
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.grid(True)
ax.set_title("Polynomial fitting and overfitting")

ax.plot(xs, ys, 'o', color='blue', label='Training data')
ax.plot([x for x, _ in hidden], [y for _, y in hidden], 'x', color='orange', label='Hidden data')
line, = ax.plot([], [], 'r-', linewidth=2, label='Model')
ax.legend()

for epoch in range(2500):
    train = 0
    for x, y in zip(xs, ys):
        err = pred(x) - y
        train += err**2
        for k in range(K):
            w[k] -= lr * err * x**k
    test = sum((pred(x)-y)**2 for x, y in hidden)

    if epoch % 50 == 0 or epoch == 4999:
        curve = [pred(x) for x in x_axis]
        line.set_data(x_axis, curve)
        ax.set_title(f"Epoch {epoch} | train={train:.6f} | test={test:.6f}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

print("train:", train)
print("test", test)

plt.ioff()
plt.show()

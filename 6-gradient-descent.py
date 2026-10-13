import matplotlib.pyplot as plt

xs = [1, 2, 3, 4, 5]                # hours studied
ys = [2.9, 5.2, 7.1, 8.8, 11.1]     # points scored

def loss(w, b):  # error as one number
    err = 0
    for x, y in zip(xs, ys):
        err += (w * x + b - y) ** 2
    return err / len(xs)

def slope(w, b, h=1e-6):  # how the ground tilts
    dw = (loss(w + h, b) - loss(w, b)) / h
    db = (loss(w, b + h) - loss(w, b)) / h
    return dw, db

w, b, rate = 0.0, 0.0, 0.05

plt.ion()
fig, ax = plt.subplots()
ax.scatter(xs, ys, color='blue', label='data points')
line, = ax.plot(xs, [w * x + b for x in xs], color='red', label='best fit line')
ax.set_title('Gradient Descent')
ax.set_xlabel('Hours studied')
ax.set_ylabel('Points scored')
ax.set_xlim(0, 6)
ax.set_ylim(0, 15)
ax.legend()

for step in range(300):
    dw, db = slope(w, b)          # which way is uphill
    w, b = w - rate * dw, b - rate * db  # downhill

    line.set_ydata([w * x + b for x in xs])
    ax.set_title(f'Gradient Descent - step {step + 1}')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

print(round(w, 2), round(b, 2), loss(w, b))
plt.ioff()
plt.show()

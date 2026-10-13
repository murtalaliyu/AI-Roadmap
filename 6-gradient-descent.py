xs = [1, 2, 3, 4, 5]                # hours studied
ys = [2.9, 5.2, 7.1, 8.8, 11.1]     # points scored

def loss(w, b): # error as one number
    err = 0
    for x, y in zip(xs, ys):
        err += (w*x + b - y) ** 2
    return err / len(xs)

def slope(w, b, h=1e-6):    # how the ground tilts
    dw = (loss(w+h, b) - loss(w, b)) / h
    db = (loss(w, b+h) - loss(w, b)) / h
    return dw, db

w, b, rate = 0.0, 0.0, 0.05
for step in range(300):
    dw, db = slope(w, b)    # which way is uphill
    w, b = w - rate*dw, b - rate*db     # downhill

print(round(w, 2), round(b, 2), loss(w, b))

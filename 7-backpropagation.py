import math

x = [1.0, 0.5]                  # input
y = 1.0                         # target
W1 = [[0.5, -0.4], [0.8, 0.3]]  # layer one
W2 = [0.9, 0.46]                # layer two

def sig(z):
    return 1 / (1 + math.exp(-z))

h = [sig(W1[j][0]*x[0] + W1[j][1]*x[1]) for j in range(len(W1))]   # forward pass
o = sig(W2[0]*h[0] + W2[1]*h[1])

loss = (o - y) ** 2             # how wrong we are

d_o = 2*(o - y) * o*(1 - o)     # blame at output
d_h = [d_o * W2[j] * h[j]*(1 - h[j]) for j in range(len(W2))]    # blame one layer back

gW2 = [d_o * h[j] for j in range(2)]
gW1 = [[d_h[j]*x[i] for i in range(2)] for j in range(2)]   # every weight at once

print(round(o, 2), round(loss, 3), gW2)

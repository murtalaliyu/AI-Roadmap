import math

data = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

sig = lambda z: 1 / (1 + math.exp(-z))

# weights:
# w[0], w[1], w[2] -> hidden neuron a
# w[3], w[4], w[5] -> hidden neuron b
# w[6], w[7], w[8] -> output neuron
w = [0.3, -0.8, 0.5, -0.2, 0.9, 0.1, 0.7, -0.6, 0.2]
lr = 0.5

for epoch in range(20000):
    for x0, x1, y in data:
        # forward pass
        a = sig(w[0] * x0 + w[1] * x1 + w[2])
        b = sig(w[3] * x0 + w[4] * x1 + w[5])
        p = sig(w[6] * a + w[7] * b + w[8])

        # output loss derivative for sigmoid output
        # this is the standard sigmoid derivative term
        d_out = (p - y) * p * (1 - p)

        # backpropagate through hidden neurons
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

# show final predictions
for x0, x1, y in data:
    a = sig(w[0] * x0 + w[1] * x1 + w[2])
    b = sig(w[3] * x0 + w[4] * x1 + w[5])
    p = sig(w[6] * a + w[7] * b + w[8])
    print((x0, x1), "target =", y, "prediction =", round(p, 4))
    
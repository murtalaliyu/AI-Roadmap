from learn import learn, grade, exp

def none(z): return z
def sigmoid(z): return 1 / (1 + exp(-z))
def relu(z): return max(0, z)

# a neuron: weighted sum plus bias, then a curve
def neuron(x, w, b, act):
    z = w[0]*x[0] + w[1]*x[1] + b
    return act(z)

# XOR: same inputs 0, different inputs 1
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [[0], [1], [1], [0]]

for act in (none, sigmoid, relu):
    net = learn(xs, ys, hidden=2, act=act)
    print(act.__name__, grade(net, xs, ys))

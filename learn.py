import math
import random

exp = math.exp

def _activation_derivative(z, act):
    if act.__name__ == "sigmoid":
        value = act(z)
        return value * (1 - value)
    if act.__name__ == "relu":
        return 1.0 if z > 0 else 0.0
    return 1.0

def learn(xs, ys, hidden=2, act=None, epochs=10000, lr=0.3):
    if not xs or len(xs) != len(ys):
        raise ValueError("xs and ys must contain the same non-zero number of examples")
    if hidden < 1:
        raise ValueError("hidden must be at least 1")
    if act is None:
        raise ValueError("act must be an activation function")

    input_size = len(xs[0])
    output_size = len(ys[0])
    if any(len(x) != input_size for x in xs):
        raise ValueError("all input examples must have the same size")
    if any(len(y) != output_size for y in ys):
        raise ValueError("all target examples must have the same size")

    rng = random.Random(0)
    input_weights = [
        [rng.uniform(-1, 1) for _ in range(input_size)]
        for _ in range(hidden)
    ]
    hidden_biases = [0.1 for _ in range(hidden)]
    output_weights = [
        [rng.uniform(-1, 1) for _ in range(hidden)]
        for _ in range(output_size)
    ]
    output_biases = [0.1 for _ in range(output_size)]

    def predict(inputs):
        hidden_values = [
            act(sum(weight * value for weight, value in zip(weights, inputs)) + bias)
            for weights, bias in zip(input_weights, hidden_biases)
        ]
        return [
            act(sum(weight * value for weight, value in zip(weights, hidden_values)) + bias)
            for weights, bias in zip(output_weights, output_biases)
        ]

    for _ in range(epochs):
        for inputs, targets in zip(xs, ys):
            hidden_z = [
                sum(weight * value for weight, value in zip(weights, inputs)) + bias
                for weights, bias in zip(input_weights, hidden_biases)
            ]
            hidden_values = [act(z) for z in hidden_z]
            output_z = [
                sum(weight * value for weight, value in zip(weights, hidden_values)) + bias
                for weights, bias in zip(output_weights, output_biases)
            ]
            predictions = [act(z) for z in output_z]

            output_deltas = [
                (prediction - target) * _activation_derivative(z, act)
                for prediction, target, z in zip(predictions, targets, output_z)
            ]
            hidden_deltas = [
                sum(output_deltas[o] * output_weights[o][h] for o in range(output_size))
                * _activation_derivative(hidden_z[h], act)
                for h in range(hidden)
            ]

            for o in range(output_size):
                for h in range(hidden):
                    output_weights[o][h] -= lr * output_deltas[o] * hidden_values[h]
                output_biases[o] -= lr * output_deltas[o]

            for h in range(hidden):
                for i in range(input_size):
                    input_weights[h][i] -= lr * hidden_deltas[h] * inputs[i]
                hidden_biases[h] -= lr * hidden_deltas[h]

    return predict

def grade(net, xs, ys):
    if len(xs) != len(ys):
        raise ValueError("xs and ys must contain the same number of examples")
    return sum(
        (prediction - target) ** 2
        for inputs, targets in zip(xs, ys)
        for prediction, target in zip(net(inputs), targets)
    )

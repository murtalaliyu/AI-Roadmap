# Import NumPy for array operations, vectorized math, and random sampling.
import numpy as np

# Create a reproducible random-number generator using seed 28.
rng = np.random.default_rng(28)

# Create integer frequencies from 1 through 128 for the target function.
j = np.arange(1, 129)
# Give each frequency an independently sampled phase offset between 0 and 7.
ph = rng.uniform(0, 7, 128)

# Define a target as a weighted sum of sine waves, with frequency j weighted by 1/j.
f = lambda x: np.sin(np.outer(x, j) + ph) @ (1/j)

# Sample 4,000 inputs uniformly from the interval [-3, 3] for evaluation.
X = rng.uniform(-3, 3, 4000)
# Compute the target function values for those held-out evaluation inputs.
Y = f(X)

# Train and evaluate separate networks with the listed numbers of hidden neurons.
for n in (4, 8, 16, 32, 64, 128):
    # Place the neurons' initial transition centers evenly across [-3, 3].
    p = np.linspace(-3, 3, n)
    # Initialize the elementwise squared-gradient accumulator with a tiny positive value.
    V = 1e-18
    # Initialize input weights to 100, biases to -100 times each center, and output weights to zero.
    W = 100 * np.array([p**0, -p, 0*p])

    # Define hidden activations: each neuron applies tanh to its weighted input plus bias.
    h = lambda x: np.tanh(np.outer(x, W[0]) + W[1])

    # Train longer for larger networks, using one update for each loop iteration.
    for step in range(100000 + 1000 * n):
        # Draw a fresh mini-batch of 64 training inputs from [-3, 3].
        x = rng.uniform(-3, 3, 64)
        # Evaluate all hidden neurons for each input in the current mini-batch.
        a = h(x)
        # Compute prediction errors against the target, scaled by the batch size.
        e = (a @ W[2] - f(x)) / 64
        # Backpropagate the errors through the output weights and tanh derivative.
        g = np.outer(e, W[2]) * ( 1 - a * a)
        # Assemble gradients for input weights, biases, and output weights into a 3-by-n array.
        G = np.array([x @ g, g.sum(0), a.T @ e])
        # Accumulate squared gradients elementwise to adapt each parameter's update scale.
        V += G * G
        # Apply the gradient update, normalized by accumulated gradient magnitude and neuron count.
        W -= 2 * G / V**0.5 / n

    # Print neuron count, total parameter count, and mean squared error on the held-out inputs.
    print(n, W.size, ((h(X) @ W[2] - Y)**2).mean())

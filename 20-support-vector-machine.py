# Import NumPy for arrays, vectorized calculations, and random data generation.
import numpy as np

# Create a reproducible random-number generator using seed 29.
r = np.random.default_rng(29)

# Learn a linear SVM from feature rows X and labels y, aiming for a wide margin.
def svm(X, y):
    # Initialize one weight per feature; multiplying the first row by zero preserves its shape.
    w = 0 * X[0]
    # Initialize the scalar intercept (bias) to zero.
    b = 0.0
    # Perform 50,000 iterative updates, with t also controlling the shrinking step size.
    for t in range(1, 50001):
        # Select examples whose signed margin is below 1, meaning they are inside the margin or misclassified.
        s = y * (X @ w + b) < 1
        # Scale the selected labels by 10/t to form their contribution to the parameter updates.
        g = 10 * y[s] / t
        # Shrink the weights by 1/t, then add the selected examples' weighted feature vectors.
        w = w - w/t + g @ X[s]
        # Adjust the intercept by the sum of the selected examples' scaled label contributions.
        b = b + g.sum()
    # Return the learned weight vector and intercept defining the decision boundary.
    return w, b

# Repeat the two class labels 1,020 times each, producing 2,040 labels total.
y = np.tile([-1, 1], 1020)
# Generate three noisy features per example, centered at a class-dependent location.
X = r.normal(np.outer(y, [1.6, 1.1, 1]), 0.6)

# Run one experiment with the first two features and another with all three features.
for A in X[:, :2], X:
    # Fit the model using only the first 40 feature rows and their corresponding labels.
    w, b = svm(A[:40], y[:40])
    # Compute signed margins for every example; positive values indicate correct-side predictions.
    m = y * (A @ w + b)

    # Print the distance between the two margin boundaries, equal to 2 divided by the weight norm.
    print(2 / (w @ w)**0.5)
    # Print indices of training examples at or very near the margin (signed margin below 1.01).
    print(np.where(m[:40] < 1.01)[0])
    # Print the fraction of the remaining 2,000 examples classified on the correct side, then a blank line.
    print(np.mean(m[40:] > 0), "\n")

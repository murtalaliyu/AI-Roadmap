import math
import matplotlib.pyplot as plt
from digits import train, test  # 16x16 images

W = [[0.0] * 256 for d in range(10)]    # 10 neurons

def see(x):
    s = [sum(w * p for w, p in zip(W[d], x)) for d in range(10)]    # 10 scores
    e = [math.exp(v - max(s)) for v in s]   # for softmax stability, prevents overflow/underflow
    return [v / sum(e) for v in e]  # softmax

plt.ion()  # turn on interactive plotting

fig, (image_ax, probabilities_ax, accuracy_ax) = plt.subplots(
    1, 3, figsize=(12, 4)
)
fig.tight_layout(pad=3)

# Show the current training image.
image = image_ax.imshow([[0] * 16 for _ in range(16)], cmap="gray", vmin=0, vmax=1)
image_ax.axis("off")

# Show the model's probabilities for digits 0 through 9.
bars = probabilities_ax.bar(range(10), [0] * 10)
probabilities_ax.set_title("Prediction probabilities")
probabilities_ax.set_xlabel("Digit")
probabilities_ax.set_ylim(0, 1)

# Track test accuracy across epochs.
accuracy_line, = accuracy_ax.plot([], [], marker="o")
accuracy_ax.set_title("Test accuracy")
accuracy_ax.set_xlabel("Epoch")
accuracy_ax.set_ylabel("Accuracy")
accuracy_ax.set_xlim(1, 6)
accuracy_ax.set_ylim(0, 1)

epoch_numbers = []
accuracies = []

for epoch in range(6):
    for x, y in train:
        P = see(x)

        for d in range(10):
            g = (P[d] - (d == y)) * 0.01    # blame
            W[d] = [w - g * p for w, p in zip(W[d], x)]

    # Evaluate the model after this epoch of training.
    correct = 0
    for x, y in test:
        probabilities = see(x)
        prediction = probabilities.index(max(probabilities))
        correct += prediction == y

    accuracy = correct / len(test)
    epoch_numbers.append(epoch + 1)
    accuracies.append(accuracy)

    # Display a training example using the updated weights.
    sample_x, sample_y = train[0]
    sample_probabilities = see(sample_x)
    image.set_data([sample_x[row * 16:(row + 1) * 16] for row in range(16)])
    image_ax.set_title(f"Example: {sample_y}")

    for bar, probability in zip(bars, sample_probabilities):
        bar.set_height(probability)

    accuracy_line.set_data(epoch_numbers, accuracies)
    fig.suptitle(f"Epoch {epoch + 1}/6 | Test accuracy: {accuracy:.0%}")
    fig.canvas.draw_idle()
    plt.pause(0.5)

plt.ioff()
plt.show()

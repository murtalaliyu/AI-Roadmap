import math
import matplotlib.pyplot as plt

# Input and target
x1 = 1.0
x2 = 0.5
target = 1.0

# Weights
w_hidden1_1 = 0.5
w_hidden1_2 = -0.4
w_hidden2_1 = 0.8
w_hidden2_2 = 0.3

w_output1 = 0.9
w_output2 = 0.46

learning_rate = 0.2

# Sigmoid activation function
# It squashes values into the range (0, 1).
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Forward pass
# This computes the hidden activations and the final prediction.
def forward_pass():
    z_hidden1 = w_hidden1_1 * x1 + w_hidden1_2 * x2
    hidden1 = sigmoid(z_hidden1)

    z_hidden2 = w_hidden2_1 * x1 + w_hidden2_2 * x2
    hidden2 = sigmoid(z_hidden2)

    z_output = w_output1 * hidden1 + w_output2 * hidden2
    prediction = sigmoid(z_output)
    return hidden1, hidden2, prediction

# Start live plotting so we can see the training process update over time.
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.set_title('Prediction vs target')
ax1.set_ylim(0, 1.2)
ax1.axhline(target, color='black', linestyle='--', label='target')
line_prediction, = ax1.plot([], [], color='blue', label='prediction')
ax1.legend()

ax2.set_title('Loss over training')
ax2.set_ylim(0, 1.0)
line_loss, = ax2.plot([], [], color='red', label='loss')
ax2.legend()

prediction_history = []
loss_history = []

# Training loop
# Each epoch does one forward pass, one backward pass, and one weight update.
for epoch in range(200):
    hidden1, hidden2, prediction = forward_pass()

    # Loss tells us how wrong the prediction is.
    loss = (prediction - target) ** 2

    # Backpropagation
    # This is the derivative of the loss with respect to the output.
    output_gradient = 2 * (prediction - target) * prediction * (1 - prediction)

    # These tell us how much each hidden neuron contributed to the error.
    hidden1_gradient = output_gradient * w_output1 * hidden1 * (1 - hidden1)
    hidden2_gradient = output_gradient * w_output2 * hidden2 * (1 - hidden2)

    # Gradients for weights
    # This tells us how much the output weights should change.
    grad_w_output1 = output_gradient * hidden1
    grad_w_output2 = output_gradient * hidden2

    # This tells us how much the hidden-layer weights should change.
    grad_w_hidden1_1 = hidden1_gradient * x1
    grad_w_hidden1_2 = hidden1_gradient * x2
    grad_w_hidden2_1 = hidden2_gradient * x1
    grad_w_hidden2_2 = hidden2_gradient * x2

    # Update weights using gradients
    # We move them in the direction that reduces the loss.
    w_output1 -= learning_rate * grad_w_output1
    w_output2 -= learning_rate * grad_w_output2
    w_hidden1_1 -= learning_rate * grad_w_hidden1_1
    w_hidden1_2 -= learning_rate * grad_w_hidden1_2
    w_hidden2_1 -= learning_rate * grad_w_hidden2_1
    w_hidden2_2 -= learning_rate * grad_w_hidden2_2

    prediction_history.append(prediction)
    loss_history.append(loss)

    line_prediction.set_data(range(len(prediction_history)), prediction_history)
    line_loss.set_data(range(len(loss_history)), loss_history)

    ax1.set_xlim(0, max(10, len(prediction_history) + 10))
    ax2.set_xlim(0, max(10, len(loss_history) + 10))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

# Print final results after training.
print('final prediction:', round(prediction_history[-1], 3))
print('final loss:', round(loss_history[-1], 6))

plt.ioff()
plt.show()

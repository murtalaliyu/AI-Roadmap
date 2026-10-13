import math

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

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Forward pass
# This gives the output of the first hidden neuron.
z_hidden1 = w_hidden1_1 * x1 + w_hidden1_2 * x2
hidden1 = sigmoid(z_hidden1)

z_hidden2 = w_hidden2_1 * x1 + w_hidden2_2 * x2
hidden2 = sigmoid(z_hidden2)

# Final Prediction
z_output = w_output1 * hidden1 + w_output2 * hidden2
prediction = sigmoid(z_output)

# Loss
# This measures how wrong the prediction is.
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

# This tells us how much the first hidden layer weights should change.
grad_w_hidden1_1 = hidden1_gradient * x1
grad_w_hidden1_2 = hidden1_gradient * x2
grad_w_hidden2_1 = hidden2_gradient * x1
grad_w_hidden2_2 = hidden2_gradient * x2

print("hidden1:", round(hidden1, 3))
print("hidden2:", round(hidden2, 3))
print("prediction:", round(prediction, 3))
print("loss:", round(loss, 3))
print("output gradient:", round(output_gradient, 3))
print("hidden gradients:", round(hidden1_gradient, 3), round(hidden2_gradient, 3))
print("weight gradients:", round(grad_w_output1, 3), round(grad_w_output2, 3))
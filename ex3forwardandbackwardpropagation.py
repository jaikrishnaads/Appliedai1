# Initial weights and biases (fixed values)
# Input to Hidden weights (3x5)
w_input_hidden = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7]
]

# Hidden to Output weights (5x1)
w_hidden_output = [0.1, 0.2, 0.3, 0.4, 0.5]

# Biases
b_hidden = [0.1, 0.2, 0.3, 0.4, 0.5]
b_output = 0.1

# Input values
x = [0.5, 0.6, 0.7]

# Target output
y_target = 0.7

# Learning rate
learning_rate = 0.5

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + 10**(-x))

# Forward pass
hidden_in = [sum(w_input_hidden[i][j] * x[i] for i in range(3)) + b_hidden[j] for j in range(5)]
hidden_out = [sigmoid(h) for h in hidden_in]

output_in = sum(w_hidden_output[j] * hidden_out[j] for j in range(5)) + b_output
output_out = sigmoid(output_in)

# Backward pass
# Output layer error
output_error = y_target - output_out
output_delta = output_error * output_out * (1 - output_out)

# Hidden layer error
hidden_error = [output_delta * w_hidden_output[j] for j in range(5)]
hidden_delta = [hidden_error[j] * hidden_out[j] * (1 - hidden_out[j]) for j in range(5)]

# Update weights and biases
# Hidden to Output
w_hidden_output = [w_hidden_output[j] + learning_rate * output_delta * hidden_out[j] for j in range(5)]
b_output += learning_rate * output_delta

# Input to Hidden
w_input_hidden = [
    [w_input_hidden[i][j] + learning_rate * hidden_delta[j] * x[i] for j in range(5)]
    for i in range(3)
]
b_hidden = [b_hidden[j] + learning_rate * hidden_delta[j] for j in range(5)]

# Round to 2 decimal places
w_input_hidden = [[round(w, 2) for w in row] for row in w_input_hidden]
w_hidden_output = [round(w, 2) for w in w_hidden_output]
b_hidden = [round(b, 2) for b in b_hidden]
b_output = round(b_output, 2)

# Print unupdated and updated values
print("Unupdated Weights and Biases:")
print("Input to Hidden Weights:", [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.6, 0.7]])
print("Hidden to Output Weights:", [0.1, 0.2, 0.3, 0.4, 0.5])
print("Hidden Biases:", [0.1, 0.2, 0.3, 0.4, 0.5])
print("Output Bias:", 0.1)

print("\nUpdated Weights and Biases:")
print("Input to Hidden Weights:", w_input_hidden)
print("Hidden to Output Weights:", w_hidden_output)
print("Hidden Biases:", b_hidden)
print("Output Bias:", b_output)

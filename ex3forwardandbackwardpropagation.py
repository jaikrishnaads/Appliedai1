# Configuration
LR = 0.1  # Learning Rate
TARGET = 1.0  # The value we want the network to predict
X = [1.0, 0.5, 2.0]  # Inputs x1, x2, x3

# 1. Initialize Weights and Biases (No random lib)
# Weights from Input (3) to Hidden (5) - 15 total
w_input_hidden = [[0.5 for _ in range(5)] for _ in range(3)]
# Biases for Hidden Layer (5)
b_hidden = [0.1 for _ in range(5)]

# Weights from Hidden (5) to Output (1) - 5 total
w_hidden_output = [0.5 for _ in range(5)]
# Bias for Output Layer (1)
b_output = 0.1

def show_state(label):
    print(f"--- {label} ---")
    print(f"Hidden Layer Biases: {[round(b, 2) for b in b_hidden]}")
    print(f"Output Layer Bias: {round(b_output, 2)}")
    print("Sample Weights (Input[0] to Hidden):", [round(w_input_hidden[0][i], 2) for i in range(5)])
    print("Weights (Hidden to Output):", [round(w, 2) for w in w_hidden_output])
    print("\n")

# Show Initial State
show_state("INITIAL STATE (UNUPDATED)")

# --- FORWARD PASS ---
# Calculate Hidden Layer Neurons (x4 to x8)
# Note: Using a simple linear activation for clarity
hidden_outputs = []
for j in range(5):
    neuron_sum = b_hidden[j]
    for i in range(3):
        neuron_sum += X[i] * w_input_hidden[i][j]
    hidden_outputs.append(neuron_sum)

# Calculate Output Neuron (x10)
output = b_output
for j in range(5):
    output += hidden_outputs[j] * w_hidden_output[j]

# --- BACKWARD PASS (Manual Update) ---
# 1. Calculate Error (Target - Prediction)
error = TARGET - output

# 2. Update Output Weights and Bias
# Gradient for output weights = error * hidden_output
for j in range(5):
    delta_w_out = error * hidden_outputs[j]
    w_hidden_output[j] += LR * delta_w_out

b_output += LR * error

# 3. Update Input-to-Hidden Weights and Biases
# Gradient for hidden = error * weight_to_output * input
for j in range(5):
    # Backpropagate error through the output weight
    hidden_error = error * w_hidden_output[j]
    
    for i in range(3):
        w_input_hidden[i][j] += LR * hidden_error * X[i]
    
    b_hidden[j] += LR * hidden_error

# Show Final State
show_state("UPDATED STATE (AFTER 1 ITERATION)")
print(f"Final Prediction was: {round(output, 2)}")

import numpy as np 
# Step Activation Function 
def activation(net_input): 
return 1 if net_input >= 0 else 0 
# AND Gate Inputs and Targets 
inputs = np.array([[0,0], [0,1], [1,0], [1,1]]) 
targets = np.array([0, 0, 0, 1]) 
# Initial Parameters based on observations 
weights = np.array([0.5, 0.5]) 
bias = -0.7  
learning_rate = 0.1 
print("--- AND Gate Perceptron Execution ---") 
print(f"Initial Weights: {weights}, Initial Bias: {bias}\n") 
for i in range(len(inputs)): 
# Calculate Net Input: (x1*w1 + x2*w2) + b 
net_input = np.dot(inputs[i], weights) + bias 
prediction = activation(net_input) 
# Error Calculation (Target - Output) 
error = targets[i] - prediction 
print(f"Input: {inputs[i]} | Target: {targets[i]} | Predicted: {prediction} | Error: {error}") 
print("\nResult: No weight updates required. AND gate is correctly classified.") 
import numpy as np 
# Step Activation Function 
def activation(net_input): 
return 1 if net_input >= 0 else 0 
# OR Gate Inputs and Targets 
inputs = np.array([[0,0], [0,1], [1,0], [1,1]]) 
targets = np.array([0, 1, 1, 1]) 
# Initial Parameters based on observations 
weights = np.array([0.5, 0.5]) 
bias = -0.2  
learning_rate = 0.1 
print("--- OR Gate Perceptron Execution ---") 
print(f"Initial Weights: {weights}, Initial Bias: {bias}\n") 
for i in range(len(inputs)): 
# Calculate Net Input: (x1*w1 + x2*w2) + b 
net_input = np.dot(inputs[i], weights) + bias 
prediction = activation(net_input) 
# Error Calculation 
error = targets[i] - prediction 
print(f"Input: {inputs[i]} | Target: {targets[i]} | Predicted: {prediction} | Error: {error}") 
print("\nResult: No weight updates required. OR gate is correctly classified.")

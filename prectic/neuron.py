import numpy as np

input1 = [1, 2, 3, 2.5]
input2 = [2.0, 5.0, -1.0, 2.0]
input3 = [-1.5, 2.7, 3.3, -0.8]
inputs = [input1, input2, input3]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

weights4 = [0.1, -0.14, 0.5]
weights5 = [-0.5, 0.12, -0.33]
weights6 = [-0.44, 0.73, -0.13]

weights01 = [weights1, weights2, weights3]
weights02 = [weights4, weights5, weights6]

biases1 = [2, 3, 0.5]
biases2 = [-1, 2, -0.5]
layer1_outputs = np.dot(inputs, np.array(weights01).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights02).T) + biases2

print(layer2_outputs)

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()


X, y = spiral_data(samples=100, classes=3)

class layer_dence:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class activation_relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def calculate_accuracy(self, predictions, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy


class activation_softmax_Loss_categorical_crossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates
            layer.weights += weight_updates

            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates
            layer.biases += bias_updates
        else:
            layer.weights -= self.current_learning_rate * layer.dweights
            layer.biases -= self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations += 1


dense1 = layer_dence(2, 128)
activation1 = activation_relu()
dense2 = layer_dence(128, 3)
activation2 = activation_softmax_Loss_categorical_crossentropy()
optimizer = optimizer_SGD(decay=1e-3, momentum=0.9)


for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = activation2.forward(dense2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y
    accuracy = np.mean(predictions == y_true)

    if not epoch % 100:
        print(
            f"epoch: {epoch}, loss: {loss}, accuracy: {accuracy}, lr: {optimizer.current_learning_rate}"
        )

    activation2.backward(activation2.output, y_true)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

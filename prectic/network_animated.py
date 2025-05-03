import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    def __init__(self, learning_rate=1.0, decay=1e-3, momentum=0.9):
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

epochs = 10000
history_loss = []
history_accuracy = []

# Grid for background prediction
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.5, 1.5, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Training Progress with Spiral Data")

ax_loss = axs[0, 0]
ax_accuracy = axs[1, 0]
ax_spiral = axs[0, 1]

ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_xlim(0, epochs)
ax_loss.set_ylim(0, 2)

ax_accuracy.set_title("Accuracy")
ax_accuracy.set_xlabel("Epoch")
ax_accuracy.set_ylabel("Accuracy")
ax_accuracy.set_xlim(0, epochs)
ax_accuracy.set_ylim(0, 1)

ax_spiral.set_title("Spiral Data Prediction")


(line_loss,) = ax_loss.plot([], [], "r-")
(line_accuracy,) = ax_accuracy.plot([], [], "b-")
spiral_background = None
scatter = None


def animate(epoch):
    global \
        dense1, \
        activation1, \
        dense2, \
        activation2, \
        optimizer, \
        history_loss, \
        history_accuracy

    # Training step
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = activation2.forward(dense2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    y_true = np.argmax(y, axis=1) if len(y.shape) == 2 else y
    accuracy = np.mean(predictions == y_true)

    activation2.backward(activation2.output, y_true)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    history_loss.append(loss)
    history_accuracy.append(accuracy)

    # Clear previous plots
    ax_loss.clear()
    ax_accuracy.clear()
    ax_spiral.clear()

    # Loss plot
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlim(0, epochs)
    ax_loss.set_ylim(0, 2)
    ax_loss.plot(history_loss, color="red")

    # Accuracy plot
    ax_accuracy.set_title("Accuracy")
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_xlim(0, epochs)
    ax_accuracy.set_ylim(0, 1)
    ax_accuracy.plot(history_accuracy, color="blue")

    # Now predict on grid (xx, yy)
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    dense1.forward(grid_input)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.activation.forward(dense2.output)

    grid_predictions = np.argmax(activation2.activation.output, axis=1)

    # Spiral background
    ax_spiral.contourf(
        xx, yy, grid_predictions.reshape(xx.shape), alpha=0.5, cmap="coolwarm"
    )

    # Scatter points
    ax_spiral.scatter(X[:, 0], X[:, 1], c=y_true, cmap="brg", edgecolor="k", s=20)
    ax_spiral.set_title(f"Spiral Data Prediction (Epoch {epoch})")

    return ax_loss, ax_accuracy, ax_spiral


fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 3)  # 2 rows, 3 columns

ax_loss = fig.add_subplot(gs[0, 0])
ax_accuracy = fig.add_subplot(gs[1, 0])
ax_spiral = fig.add_subplot(gs[:, 1:])  # Spiral takes 2 rows and 2 columns


ani = FuncAnimation(fig, animate, frames=epochs, interval=10)

plt.tight_layout()
plt.show()

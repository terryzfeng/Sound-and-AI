import numpy as np

# Multi-Layer Perceptron
class MLP:

    # Constructor 
    # Default 3 inputs, 2 hidden layers with 3 and 5 neurons, 2 outputs
    def __init__(self, num_inputs = 3, num_hidden = [3, 5], num_outputs = 2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Total number of layers: in, hidden, out
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # random weights for edges
        self.weights = []
        for i in range(len(layers) - 1):
            # 2d array, matrix
            # row = current layer
            # col = number of neurons in the layer
            # randomly initiates matrix values from zero to 1
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

    def forward_propagate(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate net inputs
            # dot product, matrix multiplication
            # inputs DOT weight
            net_inputs = np.dot(activations, w)

            # calculate activations
            activations = self._sigmoid(net_inputs)

        return activations

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

if __name__ == "__main__":
    # create MLP
    mlp = MLP()

    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    outputs = mlp.forward_propagate(inputs)

    # print results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))

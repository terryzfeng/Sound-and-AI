import numpy as np
from random import random

# save activationsa and derivatives
# implement back propagation
# implement gradient descent
# implement train method
# train our network with dummy dataset
# make predictions

# Multi-Layer Perceptron class
class MLP:

    # Constructor 
    # Default 3 inputs, 2 hidden layers with 3 and 5 neurons, 2 outputs
    def __init__(self, num_inputs = 3, num_hidden = [3, 5], num_outputs = 2):
        """
        Args:
            num_inputs (int): number of inputs
            num_layers (list): number of layers, how many neurons per layer
            num_outputs (int): number of outputs
        """

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Total number of layers: in, hidden, out
        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # random weights for edges
        weights = []
        for i in range(len(layers) - 1):
            # 2d array, matrix
            # row = current layer
            # col = number of neurons in the layer
            # randomly initiates matrix values from zero to 1
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

        # activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # derivatives
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives


    def forward_propagate(self, inputs):
        """ 
            Compute forward propagation through network
            from input signal
        """
        activations = inputs

        # for back propogation
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs
            # dot product, matrix multiplication
            # inputs DOT previous activation weight
            net_inputs = np.dot(activations, w)

            # calculate activations
            activations = self._sigmoid(net_inputs)

            # store next activation for back propagation
            self.activations[i+1] = activations 

        return activations


    def back_propagate(self, error, verbose = False):
        # Back propagation gradient
        # dE/dW_i = (y - a_[i+1] s'(h_[i+1])) a_i
        # s'(h_[i+1]) = s(h_[i+t])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # dE/dW_[i-1] = (y - a_[i+1] s'(h_[i+1])) W-i s'(h_1) a_[i-1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]

            delta = error * self._sigmoid_derivative(activations)
            # Reshape delta : n-array([0.1,0.2]) -> n-array([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            curr_activations = self.activations[i] 
            # Reshape activations: n-array([0.1,0.2]) -> n-array([[0.1], [0.2]])
            curr_activations_reshaped = curr_activations.reshape(curr_activations.shape[0], -1)

            # Calculate derivative
            self.derivatives[i] = np.dot(curr_activations_reshaped, delta_reshaped)

            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error


    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("Original W{} {}".format(i, weights))

            derivatives = self.derivatives[i]

            weights += derivatives * learning_rate
            #print("Updated W{} {}".format(i, weights))


    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                # forward propagation
                output = mlp.forward_propagate(input)

                # calculate error
                error = target - output
            
                # back propagation
                self.back_propagate(error, verbose=False)
            
                # apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error every iteration
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))


    def _mse(self, target, output):
        return np.average((target - output)**2)


    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


if __name__ == "__main__":
    # create dataset to train a network to perform sum operation
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    """
        inputs = array([[0.1,0.2], [0.3,0.4], ...)
    """
    targets = np.array([[i[0] + i[1]] for i in inputs])
    """
        targets = array([[0.3], [0.7], ...)
    """

    # create MLP
    mlp = MLP(2, [5], 1)

    # train our mlp
    mlp.train(inputs, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    output = mlp.forward_propagate(input)
    print("Our network believes that {} + {} = {}".format(input[0], input[1], output[0]))


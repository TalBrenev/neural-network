from random import uniform as randnum
from math import e
from ast import literal_eval


class Network():
    'A simple feedforward neural network.'

    def __init__(self, num_layers, neurons_in_layer):
        '''(Network, int, list of int) -> None
        Initializes the neural network with the specified number of layers,
        with random weights and biases. The first element of neurons_in_layer
        describes the number of inputs, and all subsequent elements describe
        the number of neurons in each layer. The number of neurons in the last
        layer is equal to the number of outputs of the network.
        REQ: num_layers >= 2
        REQ: len(neurons_in_layer) = num_layers
        REQ: for all x in neurons_in_layer, x >= 1
        '''
        # Store the number of layers, and the number of neurons in each layer
        self._num_layers = num_layers
        self._neurons_in_layer = neurons_in_layer
        # Create a weight matrix with random weights. The element at l, i, j
        # in the matrix refers to the weight connecting the ith neuron in the
        # lth layer to the jth neuron in the (l+1)th layer
        self._weights = []
        for l in range(num_layers - 1):
            self._weights.append([])
            for i in range(neurons_in_layer[l]):
                self._weights[l].append([])
                for j in range(neurons_in_layer[l+1]):
                    self._weights[l][i].append(randnum(-1, 1))
        # Create a bias matrix with random biases. The element at l, i in the
        # matrix refers to the bias of the ith neuron in the lth layer
        self._biases = [None]
        for l in range(1, num_layers):
            self._biases.append([])
            for i in range(neurons_in_layer[l]):
                self._biases[l].append(randnum(-1, 1))

    def _sigma(x):
        '''(float) -> float
        Calculates the value of the sigmoid (a.k.a. logisitic) function at x.
        '''
        try:
            tmp = e**(-x)
        except OverflowError:
            return 0
        return 1 / (1 + tmp)

    def _sigma_prime(x):
        '''(float) -> float
        Calculates the value of the derivative of the sigmoid function at x.
        '''
        try:
            tmp = e**(-x)
        except OverflowError:
            return 0
        try:
            tmp2 = (1 + tmp)**2
        except:
            return 0
        return tmp / tmp2

    def GetNumInputs(self):
        '''(self) -> int
        Returns the number of inputs to the network.
        '''
        return self._neurons_in_layer[0]

    def GetNumOutputs(self):
        '''(self) -> int
        Returns the number of outputs from the network.
        '''
        return self._neurons_in_layer[self._num_layers - 1]    

    def FeedForward(self, inputs):
        '''(Network, list of float) -> list of float
        Feeds the given inputs in to the neural network, and returns the output
        of the network. Raises an InputSizeError if the number of inputs does
        not match the number of inputs to the network.
        '''
        # Check that the number of inputs is correct
        if len(inputs) != self._neurons_in_layer[0]:
            raise InputSizeError('An incorrect number of input values was given.')
        # Calculate and store the activation of every neuron. The element l, i
        # refers to the ith neuron in the lth layer
        self._activations = [inputs]
        self._weighted_inputs = [None]
        for l in range(1, self._num_layers):
            self._activations.append([])
            self._weighted_inputs.append([])
            for i in range(self._neurons_in_layer[l]):
                weighted_input = 0
                for h in range(self._neurons_in_layer[l-1]):
                    weighted_input += (self._activations[l-1][h] *
                                       self._weights[l-1][h][i])
                self._weighted_inputs[l].append(weighted_input)
                weighted_input += self._biases[l][i]
                self._activations[l].append(Network._sigma(weighted_input))
        # Return the last layer of activations
        return self._activations[self._num_layers - 1]

    def TrainFromFile(self, file_name, learning_rate):
        '''(Network, str, float) -> None
        Trains the network with the data given in the file at file_name. The
        file must be formatted as follows:
        - First line contains number of training sets, t
        - The following 2*t lines are formatted as follows:
          > Space separated numbers representing the input values on one line
          > Space separated numbers representing the expected output values on
            the following line
        Raises an InputSizeError if the number of the inputs/outputs in the
        training file is incorrect.
        REQ: learning_rate > 0
        '''
        # Get all inputs and expected outputs from the file
        inputs = []
        expected = []
        file = open(file_name, 'r')
        for t in range(int(file.readline())):
            inputs.append(list(map(int, file.readline().split(' '))))
            expected.append(list(map(int, file.readline().split(' '))))
        file.close()
        # Train the neural network using the inputs and expected outputs
        self.Train(inputs, expected, learning_rate)

    def Train(self, inputs_set, expected_set, learning_rate):
        '''(Network, list, list, float) -> None
        Trains the network with the given inputs and expected outputs.
        Raises an InputSizeError if the number of the inputs/outputs in a
        training set is incorrect.
        REQ: len(inputs) = len(expected)
        REQ: learning_rate > 0
        '''
        # Back propogate for each training set
        for t in range(len(inputs_set)):
            # Get the inputs and expected outputs
            inputs = inputs_set[t]
            expected = expected_set[t]
            # Check that the input and expected output is of the correct size
            if len(inputs) != self._neurons_in_layer[0]:
                raise InputSizeError('An incorrect number of input values was given in the training file.')
            if len(expected) != self._neurons_in_layer[self._num_layers - 1]:
                raise InputSizeError('An incorrect number of output values was given in the training file.')
            # Propogate forward
            self.FeedForward(inputs)
            # Create error matrix. The element l, i in the matrix represents
            # the error of the ith neuron in the lth layer
            errors = [None]
            for l in range(1, self._num_layers):
                errors.append([])
            # Calculate the errors in the output layer
            for i in range(self._neurons_in_layer[self._num_layers - 1]):
                error = (self._activations[self._num_layers - 1][i] -
                         expected[i])
                error *= Network._sigma_prime(self._weighted_inputs[l][i])
                errors[self._num_layers - 1].append(error)
            # Calculate the errors in the hidden layers
            for l in range(self._num_layers - 2, 0, -1):
                for i in range(self._neurons_in_layer[l]):
                    error = 0
                    for j in range(self._neurons_in_layer[l+1]):
                        error += self._weights[l][i][j] * errors[l+1][j]
                    error *= Network._sigma_prime(self._weighted_inputs[l][i])
                    errors[l].append(error)
            # Update the biases for each neuron
            for l in range(1, self._num_layers):
                for i in range(self._neurons_in_layer[l]):
                    self._biases[l][i] -= learning_rate * errors[l][i]
            # Update the weights for each neuron
            for l in range(0, self._num_layers-1):
                for i in range(self._neurons_in_layer[l]):
                    for j in range(self._neurons_in_layer[l+1]):
                        self._weights[l][i][j] -= (learning_rate *
                                                   self._activations[l][i] *
                                                   errors[l+1][j])

    def SaveNetwork(self, file_name):
        '''(Network, str) -> None
        Saves the neural network to the specified file path.
        '''
        # Open the file handle
        file = open(file_name, 'w')
        # Write the number of layers to the first line
        file.write(str(self._num_layers) + '\n')
        # Write the neurons per layer list to the second line
        file.write(str(self._neurons_in_layer) + '\n')
        # Write the weights and biases lists to the third and fourth lines
        file.write(str(self._weights) + '\n')
        file.write(str(self._biases))
        # Close the file handle
        file.close()

    def LoadNetwork(file_name):
        '''(str) -> Network
        Loads and returns the neural network saved at the specified file path.
        '''
        # Open the file handle
        file = open(file_name, 'r')
        # Load all data from the file
        num_layers = int(file.readline())
        neurons_in_layer = literal_eval(file.readline())
        weights = literal_eval(file.readline())
        biases = literal_eval(file.readline())
        # Close the file handle
        file.close()
        # Create and return a neural network with the data that was read
        network = Network(num_layers, neurons_in_layer)
        network._weights = weights
        network._biases = biases
        return network


class InputSizeError(Exception):
    pass

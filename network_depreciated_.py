"""
Maze network and associated activation functions. **Deprecated code**

Author - Samuel Warner
"""
from Modules.matrix_operations import *
import math
import random
import copy
import pickle


class Sigmoid:
    def __init__(self):
        pass

    def activate(self, x):
        """
        Returns the Sigmoid of "x"

        :param x: (int/float)  Value of x in the sigmoid function
        :return:  (float)      Y value of x
        """
        return 1 / (1 + math.exp(-x))

        # Possibly more stable implementation of sigmoid
        # if x < 0:
        #     return 1 - 1 / (1 + math.exp(-x))
        # else:
        #     return 1 / (1 + math.exp(-x))

    def derivative(self, x):
        """
        Returns the sigmoid derivative of "x".

        :param x: (int/float) Value to find the derivative of
        :return:  (float)     Sigmoid derivative of x
        """
        return (math.exp(-x)) / (1 + math.exp(-x)) ** 2


class Relu:
    def __init__(self):
        self.a = 0.01

    def activate(self, x):
        return max(0, x)

    def derivative(self, x):
        if x < 0:
            return 0
        elif x >= 0:
            return 1.0


class ReluLeaky:
    def __init__(self):
        self.a = 0.01

    def activate(self, x):
        return max(self.a*x, x)

    def derivative(self, x):
        if x < 0:
            return self.a
        elif x >= 0:
            return 1.0


class Linear:
    def __init__(self):
        self.slope = 1.0

    def activate(self, x):
        return self.slope*x

    def derivative(self, x):
        return self.slope


class Maze:
    def __init__(self, network_layers, activation_functions, lr=None):
        # Check arguments for bad input
        if type(network_layers) != list:
            raise TypeError(" expected list for network_layers, "
                            "got %s" % type(network_layers))
        elif len(network_layers) < 2:
            raise IndexError("network_layers incorrect length; network must include at least 2 layers")

        elif type(activation_functions) != list:
            raise TypeError("Expected list for activation_functions, "
                            "got %s" % type(activation_functions))

        elif len(activation_functions) != (len(network_layers)-1):
            raise IndexError("activation_functions list incorrect length")

        # Initialize class variables
        self.layers = len(network_layers)      # number of layers in the network
        self.layout = network_layers           # network layout as a list
        self.weights = []                      # all network weights
        self.layer_sums = []                   # input sums for all layers in network(excluding input layer)
        self.layer_inputs = []                 # stores each layers "input" values during forward prop
        self.__bias = True                     # determines if network should include a bias node on every layer
        self.act = []                          # list of activation function classes used in each network layer
        self.avr_error = 0                     # average error of a training runs error calculations

        # learning constant, higher value = more aggressive weight adjustments
        if lr:
            self.__learning_constant = lr
        else:
            self.__learning_constant = 0.01

        # setup the correct activation function for each network layer
        for k in range(len(activation_functions)):
            act_type = activation_functions[k]
            if act_type == 'sig':
                self.act.append(Sigmoid())
            elif act_type == 'relu':
                self.act.append(Relu())
            elif act_type == 'lin':
                self.act.append(Linear())
            elif act_type == 'reluleak':
                self.act.append(ReluLeaky())

        # initialize with random weights
        self.randomize_weights()

    def randomize_weights(self):
        """
        Method creates a set of random weights for the current network layout

        :return: N/A
        """
        for layer in range(self.layers):

            # end before adding weights to output layer
            if layer == self.layers-1:
                break
            else:
                # Number of nodes in previous layer determines random weight initial values
                r = 1 / math.sqrt(self.layout[layer])

                # Create weight matrix array
                self.weights.append([])
                for node in range(self.layout[layer+1]):
                    self.weights[-1].append([])
                    for neuron in range(self.layout[layer]):
                        self.weights[-1][-1].append(random.uniform(-r, r))

                    # add weight for bias neuron if present
                    if self.__bias:
                        self.weights[-1][-1].append(random.uniform(-r, r))

    def forward(self, inputs_a):
        """
        Forward propagation function for the entire network. Takes in network input and outputs the final
        value of the last layer in the network.

        :param inputs_a: (list)  A 1xN Matrix of input values, where N is the number of neurons in the input layer
        :return: (list)          A 1xN Matrix of output values, where N is the number of neurons in the output layer
        """
        inputs = copy.deepcopy(inputs_a)

        # Check arguments for bad input
        if type(inputs) != list:
            raise TypeError("expected inputs argument as type list, got {}".format(type(inputs)))

        elif len(inputs[0]) != self.layout[0]:
            raise IndexError("Incorrect number of inputs for network, expected {} got {}".format(self.layout[0],
                                                                                                 len(inputs[0])))

        self.layer_sums = []    # Clean list for new sums
        self.layer_inputs = []  # Clean list for new inputs

        # Run inputs through entire network ending on the output layer which will return the result.
        for layer in range(self.layers-1):

            if self.__bias:
                inputs[0].append(1.0)

            # remember this layers inputs for back propagation later
            self.layer_inputs.append([n for n in inputs[0]])

            # transmit input through weights and store the sum of inputs to each following neuron
            self.layer_sums.append(multiply_2d(inputs, transpose_2d(self.weights[layer]))[0])


            # create input for the next layer weights by running sums through the activation function
            outputs = [[self.act[layer].activate(col) for col in self.layer_sums[layer]]]

            # use output as input for next layer
            inputs = outputs

        return outputs

    def backward(self, error):
        """
        Backward propagation function for the entire network. Takes in the error target output - network output and
        back propagates it through all layers adjusting weights as required.

        :param error: (list) One Demensional list of error values to be back propagated. Length must be equal to the
                             number of outputs in network
        :return: N/A
        """
        layer = self.layers-2  # Start on the first hidden layer before output neurons

        # setup up the derivative of layer sums for delta calculation below
        # self.layer_sums = [[self.act[layer].derivative(val) for val in layer] for layer in self.layer_sums]

        while layer >= 0:
            self.layer_sums[layer] = [self.act[layer].derivative(val) for val in self.layer_sums[layer]]

            # Multiply error by sum to get change needed in nodes
            delta = [[a*b for a,b in zip(self.layer_sums[layer], error)]]

            # Check for a layer delta containing all zeros, meaning network has a dead layer and will not update
            chk = False
            for j in error:
                if j != 0:
                    chk = True
                    break
            if chk:
                for v in delta[0]:
                    if v != 0:
                        chk = False
                        break
                if chk:
                    raise Exception("DELTA ERROR: A layer in the network has a delta of 0. Network layer has died. ")

            # Get weights for actual neurons not including bias
            if self.__bias:
                layer_weights = []
                for neuron in self.weights[layer]:
                    layer_weights.append(neuron[:-1])
            else:
                layer_weights = self.weights[layer]

            # Transmit error over weights to previous layer
            output = multiply_2d(delta, layer_weights)

            # set error for next layer
            error = output[0]

            # calculated needed adjustments
            adjustment = multiply_constant(self.__learning_constant, multiply_2d(transpose_2d(delta), [self.layer_inputs[layer]]))

            # Adjust weights for this layer
            self.weights[layer] = add_2d(adjustment, self.weights[layer])

            layer -= 1

    def train(self, inputs, target):
        """
        Runs one step of forward and backward propagation given appropriate data sets. Returns results and error after
        forward propagation and before adjusting network weights to reduce error.

        :param inputs: (list)   Matrix (list of lists) of inputs for the network, one value for each input in network.
        :param target: (list)   Matrix (list of lists) of target output values, one value for each output in network.
        :return: (list, list)   Matrices containing the result of forward propagation and the calculated network error
        """
        try:
            result = self.forward(inputs)
            e = subtract_2d(target, result)
            self.backward(e[0])

            sume = 0
            for err in e[0]:
                if err < 0:
                    sume += err*-1
                else:
                    sume += err

            return result, e, sume/len(e[0])

        except:
            # Crash logging code goes here
            raise

    def set_bias(self, value):
        """
        Method of activating or deactivating the bias inputs in the network layers. Removes associated weights when
        deactivated. When (re)activated bias weights are assigned a random value between -0.2 and 0.2 and do not use
        any old value that might have been used before deactivation.

        :param value: (bool)  True: activate bias, False: deactivate bias.
        :return: None
        """
        if type(value) == bool:
            if self.__bias == value:
                return

            # If disabling bias remove corresponding weights
            elif self.__bias and not value:
                for layer in self.weights:
                    for neuron in layer:
                        del neuron[-1]

            # If activating bias add corresponding weights
            elif value and not self.__bias:
                for layer in self.weights:
                    for neuron in layer:
                        neuron.append(random.uniform(-0.2, 0.2))

            self.__bias = value

        else:
            raise TypeError("Expected type bool, got {}".format(type(value)))

    def set_weights(self, weight_array):
        """
        Method of setting user defined weights for the network. Mostly useful for testing or debugging. Given weights
        must match network layout. If the weight array provided does not match the network layout errors will occur.

        :param weight_array: (list)  Formatted as a list of lists of lists representing the overall network, layers,
                                     and neuron weights, respectively.
        :return: N/A
        """
        if type(weight_array) != list:
            raise TypeError("Expected list, got {}".format(type(weight_array)))
        else:
            self.weights = weight_array

    def set_learning_constant(self, value):
        """
        Method sets network learning constant to given float value

        :param value: (float)  Value to set learning constant to
        :return: N/A
        """
        if type(value) == float:
            self.__learning_constant = value
        else:
            raise TypeError("Expected float for value, got {}".format(type(value)))

    def get_learning_constant(self):
        """
        Method of retrieving the networks learning constant value

        :return: (float) Value of networks learning constant
        """
        return self.__learning_constant

    def get_bias(self):
        """
        Method of accessing the bias value of the network

        :return: (float) Value of Bias
        """
        return self.__bias

    def save_weights(self, file_path):
        """
        Method of saving the weights of the network. Weight array is pickled to a given file path

        :param file_path: (directory) File path where weights should be saved
        :return: N/A
        """
        with open(file_path, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load_weights(self, file_path):
        """
        Method of loading weights for the network. Loaded weights must match network layout or exceptions will occur
        when network is run.

        :param file_path: (directory) File path where weights should loaded from
        :return: N/A
        """
        with open(file_path, 'rb') as fp:
            self.weights = pickle.load(fp)

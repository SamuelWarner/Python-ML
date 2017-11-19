"""
Contains Neural Network class MazeX

Author - Samuel Warner
"""
from Modules.mtrx import *
import math


class MazeX:
    def __init__(self, network_layers, activation_functions, learning_constant=None, bias=True):
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
        self.weights = []                      # stores all network weights
        self.layer_sums = []                   # stores sum of input signal for layers in network(excluding input layer)
        self.layer_inputs = []                 # stores each layers "input" values during forward prop
        self.__bias = bias                     # determines if network should include a bias node on every layer
        self.act = []                          # list of activation function class instances used in each network layer

        # learning constant, larger value = more aggressive weight adjustments
        if learning_constant:
            self.__learning_constant = learning_constant
        else:
            self.__learning_constant = 0.01

        # setup the correct activation function for each network layer
        for k in range(len(activation_functions)):
            act_type = activation_functions[k]
            if act_type == 'sig':
                self.act.append(MatSig())
            elif act_type == 'relu':
                self.act.append(MatRelu())
            elif act_type == 'lin':
                self.act.append(MatLin())
            elif act_type == 'reluleak':
                self.act.append(MatReluLeaky())
            else:
                raise TypeError(f"Unknown activation function \"{act_type}\" not supported.")

        # initialize with random weights
        self.randomize_weights()

    def randomize_weights(self):
        """
        Method creates a set of random weights for the current network layout

        :return: N/A
        """
        for layer in range(self.layers):

            if layer == self.layers-1:
                break  # break as last layer does not need outputs

            else:
                # Number of nodes in previous layer determines random weight initial values
                r = 1 / math.sqrt(self.layout[layer])

                if self.__bias:  # setup weights plus one for bias neuron
                    self.weights.append(np.random.uniform(-r, r, (self.layout[layer+1], (self.layout[layer] + 1))))
                else:  # setup weights without bias neuron
                    self.weights.append(np.random.uniform(-r, r, (self.layout[layer+1], self.layout[layer])))

    def forward(self, inputs):
        """
        Forward propagation function for the entire network. Takes in network input and outputs the final
        value of the last layer in the network.

        :param inputs: (ndarray)   Input values for network
        :return: (ndarray)         Output of last layer in network
        """

        # Check arguments for bad input
        if isinstance(inputs, np.ndarray):
            if inputs.shape != (1, self.layout[0]):
                raise IndexError("Incorrect number of inputs for network, expected {} got {}"
                                 .format((1, self.layout[0]), inputs.shape))
            else:
                pass
        else:
            raise TypeError(f"Expected inputs as ndarray, got {type(inputs)}")

        self.layer_sums = []    # Clean list for new sums
        self.layer_inputs = []  # Clean list for new inputs

        # Run inputs through entire network ending on the output layer which will return the result.
        for layer in range(self.layers-1):

            # If bias is enabled add bias input to inputs
            if self.__bias:
                inputs = np.append(inputs, [[1.0]], 1)

            # Store inputs for use in back propagation
            self.layer_inputs.append(inputs)

            # Run inputs through weights and store the sums of input for each neuron
            self.layer_sums.append(matmul(inputs, mattrans(self.weights[layer])))

            # Run the incoming sums of input for this layer through the activation function to get the layers output
            outputs = self.act[layer].activate(self.layer_sums[layer])

            # This layers output is used as the next layers input
            inputs = outputs

        # Return output once forward propagation has completed
        return outputs

    def backward(self, error):
        """
        Backward propagation function for the entire network. Takes in a set of error values for the output layer and
        back propagates it through all layers adjusting weights as required.

        :param error: (ndarray)  Error from most recent network output
        :return: N/A
        """

        layer = self.layers-2  # Start on the first hidden layer before output neurons

        while layer >= 0:
            # Calculate derivative of the sums of incoming signal for this layer which was stored in forward propagation
            self.layer_sums[layer] = self.act[layer].derivative(self.layer_sums[layer])

            # Multiply the error by the calculated derivative to get change needed in layer neurons
            delta = matmul_elements(error, self.layer_sums[layer])

            # Calculate weight adjustments based off needed change(delta) and a learning constant
            adjustment = matmul_scalar(self.__learning_constant, matmul(mattrans(delta), self.layer_inputs[layer]))

            # Multiply the needed change by the unadjusted weights to calculate error of the previous layer
            output = matmul(delta, self.weights[layer])

            # Create new weights for this layer using the calculated adjustment
            self.weights[layer] = matadd(adjustment, self.weights[layer])

            # Set output as error for next layer, accounting for bias neuron if they exist
            if self.__bias:
                error = output[..., :-1]
            else:
                error = output

            # Move back one layer
            layer -= 1

    def train(self, inputs, target):
        """
        Runs one step of forward and backward propagation given appropriate data sets.

        :param inputs: (ndarray)  Input for network to train on
        :param target: (ndarray)  Desired output of network for given inputs
        :return: N/A
        """
        try:
            result = self.forward(inputs)  # Forward propagation
            e = matsub(target, result)  # Calculate error from results and target values
            self.backward(e)  # Backpropagate the error through the network

        except:
            # Placeholder for crash logging code
            raise

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
        Method used to determine if bias is activate

        :return: (bool) Value of Bias
        """
        return self.__bias

    def save_weights(self, file_path):
        """
        Method of saving the weights of the network. Weight array is pickled to a given file path

        :param file_path: (directory) File path where weights should be saved
        :return: N/A
        """
        np.save(file_path, self.weights)

    def load_weights(self, file_path):
        """
        Method of loading weights for the network. Loaded weights must match network layout or exceptions will occur
        when network is run.

        :param file_path: (directory) File path where weights should loaded from
        :return: N/A
        """
        self.weights = np.load(file_path)
"""
Contains Neural Network class MazeE

Modified MazeX architecture designed to facilitate evolved neural net learning algorithms

Author - Samuel Warner
"""
from Modules.mtrx import *
import math
import random


class MazeE:
    def __init__(self, network_layers, activation_functions, bias=True, weight_min=-0.5, weight_max=0.5):
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
        self.layers = len(network_layers)       # number of layers in the network
        self.layout = network_layers            # network layout as a list
        self._weights = []                      # stores all network weights
        self.layer_sums = []                    # sums of input signal for layers in network(excluding input layer)
        self.layer_inputs = []                  # stores each layers "input" values during forward prop
        self.__bias = bias                      # determines if network should include a bias node on every layer
        self.act = []                           # list of activation function class instances used in each network layer
        self.act_layout = activation_functions  # stores activation list for breeding child networks

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
        self.randomize_weights(a=weight_min, b=weight_max)

    def randomize_weights(self, a=-0.5, b=0.5):
        """
        Method creates a set of random weights for the current network layout using a given range

        :return: N/A
        """
        self._weights.clear()
        for layer in range(self.layers):

            if layer == self.layers-1:
                break  # break as last layer does not need outputs

            else:
                # Number of nodes in previous layer determines random weight initial values
                r = 1 / math.sqrt(self.layout[layer])

                if self.__bias:  # setup weights plus one for bias neuron
                    self._weights.append(np.random.uniform(a, b, (self.layout[layer+1], (self.layout[layer] + 1))))

                else:  # setup weights without bias neuron
                    self._weights.append(np.random.uniform(a, b, (self.layout[layer+1], self.layout[layer])))

    def mutate(self, amount):
        """
        Method Mutates network weights by a given amount. Weights have 33% chances to increase by given amount, 33%
        chance to decrease, and 33% chance of staying the same.

        :param amount: (float) Amount to mutate each weight
        :return: N/A
        """
        for set in self._weights:
            for x in np.nditer(set, op_flags=['readwrite']):
                x[...] = x + (amount * random.randint(-1, 1))

    def breed(self, other, crossover):
        """
        Function breeds two networks of the same shape together and returns a new network containing their genes.
        The percentage of weights from this network that appear in the child is determined by the crossover value.
        The rest of the weights come from the "other" network.

        :param other: (MazeE) Network to breed this network with
        :param crossover: (float) 0 to 1.0 inclusive value determining the weight contribution of each parent network
        :return: (MazeE) Network with combined weights from this network and "other"
        """

        # Get weights from other network and use breeding bask to replace certain ones with this network's weights
        child_weights = other.get_weights()
        for i in range(len(child_weights)):
            np.copyto(child_weights[i], self._weights[i], where=np.random.choice([True, False],
                                                                                 self._weights[i].shape,
                                                                                 p=[crossover, 1-crossover]))
        # create the child network with the same shape as this one and add the new weights
        child = MazeE(self.layout, self.act_layout, self.__bias)
        child.set_weights(child_weights)

        return child

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
            self.layer_sums.append(matmul(inputs, mattrans(self._weights[layer])))

            # Run the incoming sums of input for this layer through the activation function to get the layers output
            outputs = self.act[layer].activate(self.layer_sums[layer])

            # This layers output is used as the next layers input
            inputs = outputs

        # Return output once forward propagation has completed
        return outputs

    def get_weights(self):
        """
        Returns network weights as a list of numpy arrays

        :return: (list) Network Weights
        """
        return self._weights

    def get_bias(self):
        """
        Method used to determine if bias is activate

        :return: (bool) Value of Bias
        """
        return self.__bias

    def set_weights(self, weights):
        """
        Method sets the weights of this network to the given weights

        :param weights: (list) List of numpy arrays containing weight values
        :return: N/A
        """
        # Check lists for matching length
        if len(weights) == len(self._weights):

            # Check each weight array for matching shape
            for i in range(len(self._weights)):
                if weights[i].shape != self._weights[i].shape:
                    raise ValueError("Weight array shape mismatched. For weight array {i} expected "
                                     "{self._weights[i].shape} got {weights[i].shape}")

            # Set weights to given values
            self._weights = weights

        else:
            raise ValueError(f"Weight list incorrect length. Expected {len(self._weights)} but got {len(weights)}")

    def save_weights(self, file_path):
        """
        Method of saving the weights of the network. Weight array is pickled to a given file path

        :param file_path: (directory) File path where weights should be saved
        :return: N/A
        """
        np.save(file_path, self._weights)

    def load_weights(self, file_path):
        """
        Method of loading weights for the network. Loaded weights must match network layout or exceptions will occur
        when network is run.

        :param file_path: (directory) File path where weights should loaded from
        :return: N/A
        """
        self._weights = np.load(file_path)
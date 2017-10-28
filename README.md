# Python Machine Learning

Welcome to a Repository of my personal machine learning code. Feel free to use it where ever you like include the lisence files as per the MIT lisence.



## Creating a Maze neural net

One of the main parts of this repository is the neural network class "Maze". It's interface is simple and intuitive and it works with matrices in the form of 2 demensional lists. To get started you only need to know what number of hidden layers you want, the number of neurons in each layer, and what activation function(see below for supported functions) you want to use. After that it is just a simple line of code:

```
mynetwork = Maze([4, 8, 2], 'sig')
```

And you're network is up and running and ready to train. Looking at the arguments a bit more we have:

```
[4, 8, 2]
```

Which tells the network we want 3 layers. The first should have 4 neurons(inputs), the second should have 8(hidden neurons), and the final layer should have 2(outputs). You can set any number of layers and neurons with this argument, however, the network must have at least 2 layers and at least 1 neuron per layer.

The next argument is a string used to tell the network what activation function should be used on the neurons(in this example we use the sigmoid function). 

```
'sig'
```

Supported functions currently are:

- Sigmoid
- Relu



## Using the network

Onces you have an instance of the class you can use the train method to train the network. Simply pass it a matrix of inputs and one of expected output like so:

```
mynetwork.train([[0.4, 0.2, 0.3, 0.7]], [[0.1, 0.5]])
```

The network will run a forward pass on the inputs, determine the error using the expected outputs, and backpropagate that error to update the weights for each layer.

You can run forward or backward propagation manually using the "forward(inputs)" or "backward(error)" methods.

As a final note the network includes a bias neuron on every layer by default. The value of each bias is set to 1.0. Bias should **only** be disabled(or re-enabled) through the "set_bias(bool)" method as weights need to be adjusted for the whole network.



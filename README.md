# Python Machine Learning

Welcome to a Repository of my personal machine learning code written almost entirely in python. 

## Brief History

This project spawned from my first attempt to create a pure python neural network architecture using only python standard library methods, which I called DOONN. DOONN was meant to be used as a way to learn about neural networks and how they are designed. After creating a basic framework I found that many machine learning problems(such as image or voice recognition) required more than standard python alone could provide. Thus this repository was created. The first version of a neural net in this project was Maze; A very similar network to Doonn. However, free from the constrants of python's standard library I was able to create a more advanced framework utilizing GPU power and the numpy library to accelerate learning. I will be adding more network types as I develop them and improving upon existing ones.

## Networks currently supported

MazeX - A general purpose Neural Network class that lets a user create a neural net of any size with a few of the more popular activation functions. Useful for data classification or as part of DQN reinforcement learning algorithms for example.

MazeE - A variation of MazeX with methods that allow for evolving network weights through genetic algorithms. Used in the Evolved driver example to have a network learn to drive a vehicle around a course.

MazeC - Work In Progress, Will eventually be a network for computer vision including convolve, max pool, and fully connected layers. Designed in a different way than other network classes here to be more user friendly


## Simple MazeX usage example

One of the main parts of this repository is the neural network class "MazeX". Its interface is simple and intuitive and it works with matrices in the form of 2 demensional numpy arrays. To get started you only need to know what number of hidden layers you want, the number of neurons in each layer, and what activation function(see below for supported functions) you want to use. After that it is just a simple line of code:

```
mynetwork = MazeX([4, 8, 2], ['relu', 'sig'])
```

And you're network is up and running and ready to train. Looking at the arguments a bit more we have:

```
[4, 8, 2]
```

Which tells the network we want 3 layers. The first should have 4 neurons(inputs), the second should have 8(hidden neurons), and the final layer should have 2(outputs). You can set any number of layers and neurons with this argument, however, the network must have at least 2 layers and at least 1 neuron per layer.

The next argument is a list used to tell the network what activation function should be used on the neurons of each layer

```
['relu', 'sig']
```

Only 2 activation functions are required for a 3 layer network as the first layer (the input layer) does no processing. Inputs of the first layer are simply used as is.

Supported functions currently are:

- Sigmoid    ('sig')
- Relu       ('relu')
- ReluLeaky*  ('reluleak')
- Linear     ('lin')

*Relu Leaky function has a < 0 slope of 0.01.

## Using the network

Once you have an instance of the MazeX class you can use the train method to train the network. Simply pass it a 2D numpy array of inputs and one of expected output like so:

```
X = np.array([[0.4, 0.2, 0.3, 0.7]])
Y = np.array([[0.1, 0.5]])
mynetwork.train(X, Y)
```

The network will run a forward pass on the inputs, determine the error using the expected outputs, and backpropagate that error to update the weights for each layer.

You can run forward or backward propagation manually using the "forward(inputs)" or "backward(error)" methods. It is important to note that forward propagation must be run at least once before back propagation can occur. If forward propagation is not run then the back propagation method will raise exceptions.

As a final note the network includes a bias neuron on every layer by default. The value of each bias is set to 1.0. Bias as well as the networks learning constant can be set during initalization like so:

```
mynetwork = MazeX([4, 8, 2], ['relu', 'sig'], learning_constant=0.3, bias=False)
```




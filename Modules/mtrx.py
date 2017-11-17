"""
Module contains matrix operations and activation function classes for neural network design.
Requires numba, and numpy libraries.

Author - Samuel Warner
"""
from numba import jit, jitclass, float64
import numpy as np


@jit()
def matmul(a, b):
    """
    Multiplies two arrays.

    :param a: (numpy array)
    :param b: (numpy array)
    :return: (numpy array) product of a * b
    """
    return np.dot(a, b)


@jit()
def matadd(a, b):
    """
    Adds one array to another.

    :param a: (numpy array)
    :param b: (numpy array)
    :return: (numpy array) product of a + b
    """
    return a + b


@jit()
def matsub(a, b):
    """
    Subtracts one array from another.

    :param a: (numpy array)
    :param b: (numpy array)
    :return: (numpy array) product of a - b
    """
    return a - b


@jit()
def mattrans(a):
    """
    Transposes a given numpy array

    :param a: (numpy array)
    :return: (numpy array) transposed version of a
    """
    return np.transpose(a)


@jit()
def matmul_scalar(c, b):
    """
    Multiplies a given numpy array (b) by a given scalar value (c)

    :param c: (numpy array)
    :param b: (numpy array)
    :return: (numpy array) same shape as b
    """
    return np.multiply(c, b)


@jitclass([])
class MatSig:
    def __init__(self):
        pass

    def activate(self, x):
        """
        Sigmoid function implementation.

        :param x: (numpy array)
        :return: (numpy array)  Same shape as x
        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        """
        Derivative method of Sigmoid function

        :param x: (numpy array)
        :return: (numpy array)  Same shape as x
        """
        return (np.exp(-x)) / (1.0 + np.exp(-x)) ** 2.0


@jitclass([])
class MatRelu:
    def __init__(self):
        pass

    def activate(self, x):
        """
        Relu function implementation. The method modifies "x" in place for speed of computation. Beware of the m
        odification to x when using method in code.

        :param x: (numpy array)
        :return: (numpy array)  Same shape as x
        """
        np.maximum(x, 0.0, x)
        return x

    def derivative(self, x):
        """
        Derivative method of Relu function

        :param x: (numpy array)
        :return: (numpy array)  Same shape as x
        """
        zeros = np.full(x.shape, 0.0, dtype=float64)
        ones = np.full(x.shape, 1.0, dtype=float64)

        return np.where(x < 0.0, zeros, ones)


@jitclass([('slope', float64)])
class MatReluLeaky:
    def __init__(self):
        self.slope = 0.01

    def activate(self, x):
        """
        Implementation of leaky Relu function

        :param x: (numpy array)
        :return: (numpy array)  Same shape as x
        """
        return np.maximum(self.slope * x, x)

    def derivative(self, x):
        """
        Derivative method of leaky Relu function

        :param x: (numpy array)
        :return: (numpy array) Same shape as x
        """
        leak = np.full(x.shape, self.slope, dtype=float64)
        lin = np.full(x.shape, 1.0, dtype=float64)

        return np.where(x < 0.0, leak, lin)


@jitclass([('slope', float64)])
class MatLin:
    def __init__(self):
        self.slope = 1.0

    def activate(self, x):
        """
        Implementation of linear function

        :param x: (numpy array)
        :return: (numpy array)  returns x unmodified
        """
        return x

    def derivative(self, x):
        """
        Derivative method of linear function

        :param x: (numpy array)
        :return: (numpy array) Same shape as x
        """
        return np.full(x.shape, self.slope, dtype=float64)
"""
Basic matrix operations package. All functions assume matrix format of a list of lists. Formally created as "xmath"
module for DOONN project. **Deprecated Code**

Author - Samuel Warner
"""


def add_2d(a, b):
    """
    Adds matrix 'a' with matrix 'b' to produce a matrix result
    that contains the added values.

    :param a:   2d matrix of type 'list' to be added to 'b'
    :param b:   2d matrix of type 'list' to be added with 'a'
    :return:    (list) 2d matrix result of a + b
    """

    # check if given matrices are of "list" type
    if type(a) != list or type(b) != list:
        raise TypeError('Error xm02:Incorrect type, matrices must be of type "list"')

    # check if given matrices are of matching row size
    if len(a) != len(b):
        raise IndexError("Error xm03: Mismatch in matrix row count")

    # compare matrix columns, all rows must have the same number of columns
    i = 0
    columns = len(a[i])

    while i < len(a):
        if len(a[i]) != columns or len(b[i]) != columns:
            raise Exception("Error xm04: Incorrect matrix dimensions in row: %s" % i)
        i += 1

    # add matrices and return matrix result
    zipped = list(zip(a, b))
    return [[sum(column) for column in zip(*group)] for group in zipped]


def multiply_2d(a, b):
    """
    Multiply two matrices (a*b) using zip functions. Matrices must have
    dimensions that allow multiplication (ie. M x N and N x L).

    :param a:  (list) 2D matrix
    :param b:  (list) 2D matrix
    :return:   (list) 2D matrix containing the result of a*b
    """

    # check if given matrices are of "list" type
    if type(a) != list or type(b) != list:
        raise TypeError('Error xm05: Incorrect type, matrices must be of type "list"')

    # check that row length is consistent in matrix a
    c = 0
    while c < len(a):
        if len(a[c]) != len(a[0]):
            raise ArithmeticError("Error xm06: Incorrect column dimensions in matrix a")
        c += 1

    # check that row length is consistent in matrix b
    c = 0
    while c < len(b):
        if len(b[c]) != len(b[0]):
            raise ArithmeticError("Error xm07: Incorrect column dimensions in matrix b")
        c += 1

    # check if dimensions allow for multiplication
    if len(a[0]) != len(b):
        raise ArithmeticError("Error xm08: Can not multiply a %sx%s matrix by a %sx%s matrix"
                        % (len(a), len(a[0]), len(b), len(b[0])))

    # Multiply Matrices and return resulting matrix
    zipped = zip(*b)
    zipped = list(zipped)
    return [[sum(ele_a * ele_b for ele_a, ele_b in zip(row_a, col_b))
             for col_b in zipped] for row_a in a]


def subtract_2d(a, b):
    """
    Function subtracts two matrices and returns a matrix result.

    :param a:    (list) 2D matrix
    :param b:    (list) 2D matrix
    :return:     (list) 2D matrix containing the result of a-b
    """

    # zip a and inverted b then add the two matrix together
    zipped = list(zip(a, [[b[r][i] * -1 for i in range(len(b[r]))] for r in range(len(b))]))
    return [[sum(column) for column in zip(*group)] for group in zipped]


def transpose_2d(a):
    """
    Transpose a given matrix using Zip. A 1x4 matrix becomes a 4x1 matrix

    :param a: (list) 2D Matrix to transpose
    :return:  (list) Transposed 2d matrix of a
    """

    # check if given matrix is list
    if type(a) != list:
        raise TypeError('Error xm10: Incorrect type, matrices must be of type "list"')

    # check that rows are of matching length
    l = len(a[0])
    for row in a:
        if len(row) != l:
            raise Exception("Error xm11: Row lengths do not match")

    # return transposed matrix
    return list(map(list, zip(*a)))


def multiply_constant(c, a):
    """
    Multiplies each element in matrix 'a' by a constant value 'c'.

    :param c:   (int/float) Value to multiply elements in a with.
    :param a:   (list) 2D matrix
    :return:    (list) 2D matrix result of c*a.
    """

    # check argument types
    if (type(c) != float and type(c) != int) or type(a) != list:
        raise TypeError('Error xm13: Incorrect argument type')

    # Return a matrix multiplied by the constant
    return [[a[r][i] * c for i in range(len(a[r]))] for r in range(len(a))]


def multiply_basic(a, b):
    """
    This function is NOT a standard matrix multiplication operation. It
    instead multiplies two matrix directly(first index with first index).
    Both Matrices must be: MxN where M=1 and N=any positive number.

    :param a:  (list) 2D matrix with only one row
    :param b:  (list) 2D matrix with only one row
    :return:   (list) 2D matrix containing the product of a*b
    """

    # Check if both matrix contain only one row.
    if len(a) != 1 or len(b) != 1:
        raise Exception("Error xm14: Basic multiplication only works on 1xN matrices")
    # Check for mismatched row lenght
    if len(a[0]) != len(b[0]):
        raise Exception("Error xm15: Row lengths do not match")

    # multiply elements together and return matrix
    return [[b[0][i] * a[0][i] for i in range(len(a[0]))]]


def sum_2d(a):
    """
    Sum all the values in a matrix together.

    :param a:  (list) 2D matrix containing values to be summed
    :return:   (int/float) sum value of the matrix.
    """

    # Check if given matrices are of "list" type
    if type(a) != list:
        raise TypeError('Error xm01: Incorrect type, matrices must be of type "list"')

    # add each value in each row to a total sum
    total_sum = 0
    for row in a:
        for column in row:
            total_sum += column

    # return the sum
    return total_sum


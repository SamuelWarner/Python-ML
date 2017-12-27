'''
!!WORK IN PROGRESS FILE!!
!!WORK IN PROGRESS FILE!!
!!WORK IN PROGRESS FILE!!
!!WORK IN PROGRESS FILE!!

Author Samuel Warner
'''

import matplotlib.image as mpimg
import os
from Modules.mtrx import *

# load test image
dur = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dur, "Examples", "ImageTestSet", "img_test.png")
img = mpimg.imread(path)


class MazeC:
    def __init__(self, ):
        self.layers = []
        self.relu = MatRelu()


    # filtersize, stride, input_shape, number_of_filters, padding=0
    def add_convolve_layer(self, stride=1, padding=0, filtersize=(2, 2), number_of_filters=1, input_shape=None):

        # check arguments
        if input_shape is None:
            raise ValueError("Must specify a 3 dimensional input shape for convolve layers")
        else:
            if type(input_shape) != tuple:
                raise ValueError(f"Input shape must be a tuple, got {type(input_shape)}")
            elif len(input_shape) != 3:
                raise ValueError(f"Input shape must contain 3 dimensions, got {len(input_shape)}")
            else:
                if filtersize[0] > input_shape[0] or filtersize[1] > input_shape[1]:
                    raise ValueError("Filter size too large for given input shape.")

        # create property dictionary for layer
        layer_info = {"type": "con",
                      "stride": stride,
                      "filters": [],
                      "padding": padding,
                      "filtersize": filtersize,
                      }

        for i in range(number_of_filters):
            layer_info["filters"].append(np.random.uniform(0.0, 1.0, (filtersize[0],
                                                                      filtersize[1],
                                                                      input_shape[2])))
        self.layers.append(layer_info)

    def add_maxpool_layer(self, filtersize=(2, 2), padding=0, stride=2):
        layer_info = {"type": "maxpool",
                      "filtersize": filtersize,
                      "padding": padding,
                      "stride": stride}

        self.layers.append(layer_info)

    def add_fullyconnected_layer(self, input_shape, number_of_nodes, activation_function):
        pass

    def forward(self, data):
        layer = 0
        while layer < len(self.layers):

            if self.layers[layer]["type"] == "con":
                result = self.convolve_forward(self.layers[layer], data)
            elif self.layers[layer]["type"] == "maxpool":
                result = self.maxpool_forward(self.layers[layer], data)
            else:
                raise Exception("Could not determine layer type")

            data = result
            layer += 1

        return data

    def maxpool_forward(self, info, input_array):
        output = np.zeros((int(((input_array.shape[0] - info["filtersize"][0] + info["padding"]) / info["stride"]) + 1),
                           int(((input_array.shape[1] - info["filtersize"][1] + info["padding"]) / info["stride"]) + 1),
                           input_array.shape[2]), dtype=float)

        # initial values for loops
        row = 0
        y1 = 0
        y2 = info["filtersize"][0]

        # move the pool area down the image till the bottom edge is reached
        while y2 <= input_array.shape[0]:
            x1 = 0
            x2 = info["filtersize"][1]
            column = 0

            # move the pool area to the right till the right edge is reached
            while x2 <= input_array.shape[1]:

                # process each index in layer's 3rd dimension
                for f in range(input_array.shape[2]):
                    print("FASDF:", )
                    # Find max value of this location for the 'f' index
                    val = np.max(input_array[y1:y2, x1:x2, f:f + 1])

                    # Store value for this index and location to max pooled array
                    output[[row], [column], [f]] = val

                # increment variables for next loop
                x1 += info["stride"]
                x2 += info["stride"]
                column += 1

            # increment variables for next loop
            y1 += info["stride"]
            y2 += info["stride"]
            row += 1

        return output

    def convolve_forward(self, info, input_array):
        output = np.zeros((int(((input_array.shape[0] - info["filtersize"][0] + info["padding"]) / info["stride"]) + 1),
                           int(((input_array.shape[1] - info["filtersize"][1] + info["padding"]) / info["stride"]) + 1),
                           len(info["filters"])), dtype=float)

        # initial values for loops
        row = 0
        y1 = 0
        y2 = info["filtersize"][1]

        # move the filter down the image till the bottom edge is reached
        while y2 <= input_array.shape[0]:
            x1 = 0
            x2 = info["filtersize"][0]
            column = 0

            # move the filter to the right till the right edge is reached
            while x2 <= input_array.shape[1]:

                # process each filter in layer
                for f in range(len(info["filters"])):
                    # Sum of element wise multiplication of filter weights by pixels in the current location
                    val = np.sum(matmul_elements(info["filters"][f], input_array[y1:y2, x1:x2, :]))

                    # Store value for this filter and location to activation map
                    output[[row], [column], [f]] = val

                # increment variables for next loop
                x1 += info["stride"]
                x2 += info["stride"]
                column += 1

            # increment variables for next loop
            y1 += info["stride"]
            y2 += info["stride"]
            row += 1

        return self.relu.activate(output)




def pad_image(image, pad):
    return np.lib.pad(image, [(pad, pad), (pad, pad), (0, 0)], 'constant', constant_values=(0.0))





# save activation map for review
#scipy.misc.imsave('outfile.png', new_img)
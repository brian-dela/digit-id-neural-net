import numpy as np

# initializing the network class

class Network(object):
    def __init__(self, sizes):
    # input "sizes" is a list containing the number of neurons in the respective layers
    self.num_layers = len(sizes)
    self.sizes = sizes 
    # initializig random weights and biases as list of np matrices
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
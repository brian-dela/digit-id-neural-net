import numpy as np

# initializing the network class

class Network(object):
    def __init__(self, sizes):
        # input "sizes" is a list containing the number of neurons in the respective layers
        #   for example net = Network([2,3,1]) creates a Network object with 2 neurons in the 
        #   first layer, 3 in the second, and 1 in the third. Layer 1 assumed to be input layer.
        self.num_layers = len(sizes)
        self.sizes = sizes 
        # initializig random weights and biases as list of np matrices
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if "a" is input.""" 
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network’s weights and biases by applying gradient descent using
backpropagation to a single mini batch. The "mini_batch" is a list of tuples "(x, y)", and "eta" is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b , delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b )]

def sigmoid(z):
    """INPUT: z = a vector or np.array
       OUTPUT: the elementwise computation of the sigmoid function on
               each elment of z"""
    return 1.0/(1.0+np.exp(-z))

def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None): 
    """Train the neural network using mini-batch stochastic gradient descent. The
    "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired outputs. The other non-optional parameters are self- explanatory. If "test_data" is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out. This is useful for tracking progress, but slows things down substantially. """
    if test_data:
        n_test = len(test_data) 
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n,
mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta) 
        if test_data:
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
        else: 
            print("Epoch {0} complete".format(j))

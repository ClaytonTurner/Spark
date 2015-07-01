'''
This file is based off of Google's Distbelief implementation (see README for details)
This is a python implementation of Distbelief built on top of 
	our adagrad_nn implementation
'''

import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class Node:
	
	def __init__(self,weight,machine_on,connections_to):
		self.weight = weight
		self.machine_on = machine_on # So we know whether to go to a new machine through connections
		self.connections_to = connections_to # This is a list of nodes
			# We will check their machine_on values to know when comm needs to happen
	def get_machine(self):
		return self.machine_on
	def get_connections(self):
		return self.connections_to

class DistBelief:

    def __init__(self, layers, activation='tanh', machines=[2,2]):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

	self.machines = machines # This is for representing the model parallelism
				 #	within DistBelief
				 # We will assume even numbers only
				 # The value is a shape of the model
	machines_x,machines_y = machines
	machine_nodes = [[] for i in range(sum(machines))] #list of lists. each top level list represents a machine
			   # each sublist represents nodes at that layer
	# Set up the parallelization nodes for the y
	y_section = []
	for i in range(machines_y):
		y_section.append(layers[i*len(layers)/machines_y:(i+1)*len(layers)/machines_y])
	y_flat = [item for sublist in y_section for item in sublist]
	new_rep = []
	for y in y_section:
		i = 0
		for mach in y:
			if i == 0:
				start = y_flat.index(mach)
			end = y_flat.index(mach)
			print i
			print "start",start
			print "end",end
			i += 1
		temp_rep = []
		for i in range(len(y_flat)):
			if i >= start and i <= end:
				temp_rep.append(y_flat[i])
			else:
				temp_rep.append(0)
		new_rep.append(temp_rep)
	# Set up the parallelization nodes for the x


        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
	    n = Node(r, , )
            self.weights.append(n)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def set_weights(self,params):
	self.weights = params

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        for k in range(epochs):
            if k % 10000 == 0: print 'epochs:', k
            #if k % 10000 == 0: print 'weights:',self.weights 
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
		    #print "a[l]:",a[l]
		    #print "self.weights[l]:",self.weights[l]
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
	    adagrad_cache = [0 for x in range(len(self.weights))]
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
		self.weights[i] = np.array(self.weights[i]).copy() # This prevents discontiguous memory errors
		grad = layer.T.dot(delta)
		adagrad_cache[i] += grad**2
               	self.weights[i] += learning_rate * grad / np.sqrt(adagrad_cache[i] + 1e-8)


    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)     
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    print "This is a test of the neural net on the XOR problem\n"

    #nn = NeuralNetwork([2,2,1])
    nn = NeuralNetwork([2,2,2])
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    #y = np.array([0, 1, 1, 0])
    y = np.array([[1,0],[0,1],[0,1],[1,0]])

    nn.fit(X, y, learning_rate=0.1)

    for e in X:
        print(e,nn.predict(e))

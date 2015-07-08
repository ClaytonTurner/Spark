'''
This file is based off of Google's Distbelief implementation (see README for details)
This is a python implementation of Distbelief built on top of 
	our adagrad_nn implementation
'''

import numpy as np
from copy import deepcopy

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class Node:
	
	def __init__(self,weight,machine_on):
		self.weight = weight
		self.machine_on = machine_on # So we know whether to go to a new machine through connections
	def get_machine(self):
		return self.machine_on
	def set_machine(self,mach_on):
		self.machine_on = mach_on
	def get_weight(self):
		return self.weight
	def set_weight(self,w):
		self.weight = w

class DistBelief:

    def __init__(self, layers, machine_index, activation='tanh', machines=[2,2]):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

	###################
	# Parallelization #
	###################

	self.machines = machines # This is for representing the model parallelism
				 #	within DistBelief
				 # We will assume even numbers only
				 # The value is a shape of the model
	machines_x,machines_y = machines
	machine_nodes = [[] for i in range(sum(machines))] #list of lists. each top level list represents a machine
			   # each sublist represents nodes at that layer

	##############################################
	# Set up the parallelization nodes for the y #
	##############################################
	y_flat_used = [False for x in layers]
	y_section = []
	for i in range(machines_y):
		y_section.append(layers[i*len(layers)/machines_y:(i+1)*len(layers)/machines_y])
	y_flat = [item for sublist in y_section for item in sublist]
	new_rep = []
	first_y = True
	for y in y_section:
		i = 0
		for mach in y: # Could be fixed by listifying like [-1,1] but not terrible as is
			if i == 0:
				index_found = y_flat.index(mach)
				prev_index_found = -8 # just an index we know won't get used
				increaser = 1
				while(y_flat_used[index_found] == True):
					# Let's find the next occurrence then
					index_found = y_flat[index_found+1:].index(mach) + index_found
					if index_found == prev_index_found and prev_index_found == increaser:
						index_found += 1
						increaser += 1
					prev_index_found = index_found
					
				y_flat_used[index_found] = True
				start = index_found
				# Initially this was needed but doesn't seem to be now
				#if not first_y and len(layers) % machines_y == 0: # Something here
				#	start += 1-1
			if i == len(y) - 1:
				end = y_flat.index(mach)
				while(y_flat_used[end] == True):
					end = y_flat[end+1:].index(mach) + end + 1
			y_flat_used[i] = True
			i += 1
		temp_rep = []
		for i in range(len(y_flat)):
			if i >= start and i <= end:
				temp_rep.append(y_flat[i])
			else:
				temp_rep.append(0)
		new_rep.append(temp_rep)
		first_y = False
	
	##############################################
	# Set up the parallelization nodes for the x #
	##############################################
	print "NN Layers:",layers
	final_rep = []
	for mach in new_rep:
		odd_added = False
		for i in range(machines_x):
			#final_rep.append([x/machines_x for x in mach])
			temp_final_rep = []
			for j in range(len(mach)):
				x = mach[j]
				if x % machines_x == 0 or odd_added: # if the amount of nodes is even relative to machines_x
					temp_final_rep.append(x/machines_x)
				else: # amount of nodes is odd
					temp_final_rep.append(x/machines_x + x%machines_x)
					odd_added = True
			final_rep.append(temp_final_rep)

	print "Distbelief Node Distribution:",final_rep 
	############################
	# Parallelization complete #
	############################
	dist_net = final_rep[machine_index]
	print "Current Distbelief Node:",dist_net
        # Set weights
        self.weights = []
        # range of weight values (-1,1)
        # input and hidden layers
	for i in range(1, len(layers) - 1):
		r = 2 * np.random.random(( layers[i-1] + 1, layers[i] + 1 )) - 1
		if dist_net[i] == 0: # This machine isn't responsible for maintaining this node
			for index,item in enumerate(r):
				r[index] = 0.0 # We know our weights will never randomize
					# to zero and this way flags and prevents NaNs
		print r
		n = Node(r,machine_index)
		self.weights.append(n)
	for node in self.weights:
		print "node.get_weight()",node.get_weight()
	#print self.weights	 
	'''
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
	    #n = Node(r, , )
	    print "r",r
            self.weights.append(r)#n)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
	'''
	
	# END __init__

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

    print "This is a test of distbelief as a local NN on the XOR problem"

    layers = [2,8,6,4,2] # [input, hidden layers..., output layer]
    #layers = [2,4,4,2]
    #layers = [2,3,3,2]
    #layers = [2,4,4,4,4,4,4,2]
    nn = DistBelief(layers,1)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    #y = np.array([0, 1, 1, 0])
    y = np.array([[1,0],[0,1],[0,1],[1,0]])

    nn.fit(X, y, learning_rate=0.1)

    for e in X:
        print(e,nn.predict(e))
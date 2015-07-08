import numpy as np

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1.0 - x**2


class NeuralNetwork:

	def __init__(self, layers, activation='tanh'):
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.activation_prime = sigmoid_prime
		elif activation == 'tanh':
			self.activation = tanh
			self.activation_prime = tanh_prime

		#Wij is the weight from node i in layer l-1 to node j in layer l
		#for a NN with layers = [2,2,2] :
			# input and hidden layers - (2+1, 2) : 3 x 2
			# output layer - (2+1, 2) : 3 x 2
		self.weights = []
		for i in range(1, len(layers) - 1):
			r = 2*np.random.random((layers[i-1] + 1, layers[i])) -1
			self.weights.append(r)
		# output layer - random((2+1, 1)) : 3 x 1
		r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
		self.weights.append(r)

	def set_weights(self,params):
		self.weights = params

	def feed_forward(self, x):
		a = [x]
		for l in range(len(self.weights)):
			inputs = np.hstack((1, a[l])) #add the biases node
			activation = self.activation(np.dot(inputs, self.weights[l]))
			a.append(activation)
		return a

	def fit(self, X, y, learning_rate=0.2, epochs=50000):
		for k in range(epochs):
			if k % 10000 == 0: print 'epochs:', k
			#if k % 10000 == 0: print 'weights:',self.weights 
			i = np.random.randint(X.shape[0])
			
			a = self.feed_forward(X[i])
			
			# output layer
			error = y[i] - a[-1]
			deltas = [error * self.activation_prime(a[-1])]

			# we need to begin at the second to last layer 
			# (a layer before the output layer)
			for l in range(len(self.weights) - 1, 0, -1): 
				weights_tmp = self.weights[l][1:] #remove biases node's weights
				deltas.append(deltas[-1].dot(weights_tmp.T)*self.activation_prime(a[l]))

			# reverse
			# [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
			deltas.reverse()

			# backpropagation
			# 1. Multiply its output delta and input activation 
			#    to get the gradient of the weight.
			# 2. Subtract a ratio (percentage) of the gradient from the weight.
			grad = np.array(())
			for i in range(len(self.weights)):
				a_tmp = np.hstack((1, a[i]))
				delta_l =  np.outer(deltas[i], a_tmp).T
				self.weights[i] += learning_rate * delta_l


	def predict(self, x): 
		a = self.feed_forward(x)
		return a[-1]

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

	nn.fit(X, y)

	for e in X:
		print(e,nn.predict(e))

import socket
import sys
from math import sqrt
import random
import xmlrpclib
from neural_net import NeuralNetwork
import numpy as np
import base64

learning_rate = 0.1 # tune later on
n_fetch = 1 # fixed in the paper, so let's leave it that way here
n_push = 1 # same as n_fetch

'''
Parameter and Gradient initialization will now happen on the param server
we can do an initial read from all replicas if needbe, but there is no
	reason to re-calculate that information for each replica
	if it will be the same from the start
'''

def compute_gradient(nn):
	# This function doesn't actually compute the gradient, but
	#	rather it accesses the updated weights as a result
	#	of the processed mini-batch and computed gradient
	# All computation takes place here anyways so this is merely
	#	a roundabout method of grabbing what we need
	return nn.weights

def slice_data(data,shape):
	# This function assumes np.array as the type for data
	# This function separates data into X (features) and Y (label) for the NN
	print data
	print data.shape 
	data = np.reshape(data,shape)
	print data
	print data.shape
	y = data[:,-1]
	x = data[:,:-1]
	return x,y

if __name__ == "__main__":
        #data_file = "/data/spark/Spark/iris_labelFirst.data"

	step = 0
	
	HOST = socket.gethostname()
	PARAM_SERVER,PORT = "192.168.137.50",8000
        
	proxy = xmlrpclib.ServerProxy("http://"+PARAM_SERVER+":"+str(PORT)+"/",allow_none=True)

	random.seed(8008) # should allow stabilization across machines
			  # Removes need for initial weight fetch
	layers = [2,2,1] # layers - 1 = hidden layers, I believe
	nn = NeuralNetwork(layers)#,activation='sigmoid')
	grad_push_errors = 0
        while(step < 5):# Going to change up later for minibatch counts
                if step%n_fetch == 0: # Always true in fixed case
			parameters = proxy.startAsynchronouslyFetchingParameters()
			nn.set_weights(parameters)
		data,shape = proxy.getNextMinibatch()
		if(data == "Done"):
			break
		data = np.frombuffer(base64.decodestring(data)) #.fromstring()
		x,y = slice_data(data,shape)
		nn.fit(x,y,learning_rate)
		accrued_gradients = compute_gradient(nn)
		if step%n_push == 0: # Always true in fixed case
			if proxy.startAsynchronouslyPushingGradients(accrued_gradients) is not True:
				grad_push_errors += 1
                step += 1
	print "Neural Network Replica trained with "+str(grad_push_errors)+" Gradient errors\n"
	'''
	At this point, our NN replica is done. We need to make sure all replicas have completed
	Then we can pull down a final set of weights for a trained model for prediction
	'''	
	
	x = np.array([[0,0],[0,1],[1,0],[1,1]])
	for e in x:
		print(e,nn.predict(e))
 ##gradient = sc.parallelize(data, numSlices=slices) \
 ##       .mapPartitions(lambda x: computeGradient(parameters,x) \
 ##       .reduce(lambda x, y: merge(x,y))
                

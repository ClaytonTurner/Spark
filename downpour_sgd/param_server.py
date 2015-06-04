'''
This file is for the param server which will host gradient descent updates
 to be read by and written to by the distbelief model replicas
'''

import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
from math import sqrt
import random
import socket
import numpy as np
import base64

HOST = socket.gethostname()
PORT = 8000

params = list() 

'''
# This shouldn't be needed as we use seeding in replica.py to create our models
def initialize():
	global params
	data_len = 10 # UPDATE THIS
	hidden_layers = [100,100,100]
        weight_dist = 1./sqrt(data_len)
        params = [[random.uniform(-weight_dist,weight_dist) for y in hidden_layers] for x in range(len(hidden_layers)+1)]
'''		
batches_processed = 0
batch_size = 2
data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
def getNextMinibatch():
	global batches_processed
	minibatch_start = batches_processed*batch_size
	minibatch_end = minibatch_start+batch_size
	if(minibatch_end > len(data)):
		return "Done","Done" # Gonna have to change this
	else:
		minibatch = data[minibatch_start:(minibatch_start+batch_size),:-1]
		print minibatch.shape
		batches_processed += 1 
		return base64.b64encode(minibatch.tostring()),minibatch.shape
		#return base64.b64encode(np.tostring(data[minibatch_start:(minibatch_start+batch_size),:-1]))

def startAsynchronouslyPushingGradients(grad):
	# Update the gradients on the server
	global params
	params = grad
	return True

def startAsynchronouslyFetchingParameters():
	global params
	return params

if __name__ == "__main__":
	print "Starting Param Server. Ctrl + C to quit\n"
	server = SimpleXMLRPCServer((HOST,PORT))
	print "Listening on port "+str(PORT)+"..."
	server.register_function(getNextMinibatch)
	server.register_function(startAsynchronouslyPushingGradients)
	server.register_function(startAsynchronouslyFetchingParameters)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print "Keyboard interrupt: exiting"


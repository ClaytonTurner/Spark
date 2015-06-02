'''
This file is for the param server which will host gradient descent updates
 to be read by and written to by the distbelief model replicas
'''

import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
from math import sqrt
import random
import socket

HOST = socket.gethostname()
PORT = 8000
SIZE = 4096 # Find out how big our messages will be then fix this
	    # Or send a size first and handle that then
	    # TODO

params = list() 

def initialize():
	global params
	data_len = 10 # UPDATE THIS
	hidden_layers = [100,100,100]
        weight_dist = 1./sqrt(data_len)
        weights = [[random.uniform(-weight_dist,weight_dist) for y in hidden_layers] for x in range(len(hidden_layers)+1)]
	params = weights
		

def getNextMinibatch():
	#TODO 
	return '' 

def startAsynchronouslyPushingGradients(grad):
	# Update the gradients on the server
	return True

def startAsynchronouslyFetchingParameters():
	global params
	return params

if __name__ == "__main__":
	print "Starting Param Server. Ctrl + C to quit\n"
	initialize()
	server = SimpleXMLRPCServer(("localhost",PORT))
	print "Listening on port "+str(PORT)+"..."
	server.register_function(getNextMinibatch)
	server.register_function(startAsynchronouslyPushingGradients)
	server.register_function(startAsynchronouslyFetchingParameters)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print "Keyboard interrupt: exiting"


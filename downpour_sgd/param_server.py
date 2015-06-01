'''
This file is for the param server which will host gradient descent updates
 to be read by and written to by the distbelief model replicas
'''

import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
from math import sqrt
import random

HOST = socket.gethostname()
PORT = 8000
SIZE = 4096 # Find out how big our messages will be then fix this
	    # Or send a size first and handle that then
	    # TODO

learning_rate = .2
params = list() 

def initialize():
	global params
	data_len = 10 # UPDATE THIS
	hidden_layers = [100,100,100]
        weight_dist = 1./sqrt(data_len)
        weights = [[random.uniform(-weight_dist,weight_dist) for y in hidden_layers] for x in range(len(hidden_layers)+1)]
	params = weights
		

def test_func(nums):
	return nums

if __name__ == "__main__":
	self.initialize()
	server = SimpleXMLRPCServer(("localhost",PORT))
	print "Listening on port "+str(PORT)+"..."
	server.register_function(test_func);
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print "Keyboard interrupt: exiting"


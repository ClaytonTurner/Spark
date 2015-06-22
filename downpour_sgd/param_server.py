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

HOST = "192.168.137.62" # Host by name switched to contents of /etc/hosts instead of IP. Hardcode is fine
#HOST = socket.gethostbyname(socket.getfqdn())
PORT = 49151

params = []

client_count = 5 # Needed for syncing and data sharding
shards_given = 0
clients_processed = 0
batch_size = 2 # 150 for iris dataset full batch
#data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=np.float64)
random.seed(8000)
data = np.loadtxt("iris.data",delimiter=",") # Labels must be floats
np.random.shuffle(data)
label_count = len(set(data[:,-1]))
feature_count = len(data[0])-1

def get_label_count(): # This is for setting up multiple labels for the replicas
        global label_count
        return label_count

def get_feature_count():
        global feature_count
        return feature_count

def get_minibatch_size():
        global batch_size
        return batch_size

def finish_client():
	global clients_processed
	clients_processed += 1
	return True

def finished_processing():
	global client_count
	global clients_processed
	if client_count == clients_processed:
		print client_count,"clients finished processing"
		return True
	else:
		return False

'''
def getNextMinibatch():
        global batches_processed
        minibatch_start = batches_processed*batch_size
        minibatch_end = minibatch_start+batch_size
        if(minibatch_end > len(data)):
                return "Done","Done" # Gonna have to change this
        else:
                minibatch = data[minibatch_start:(minibatch_start+batch_size),:]
                batches_processed += 1
                return base64.b64encode(minibatch.tostring()),minibatch.shape
'''

def get_data_shard():
        global shards_given
        shards_given += 1
        shard = data[(shards_given-1)*len(data)/client_count:(shards_given)*len(data)/client_count,:]
        return base64.b64encode(shard.tostring()),shard.shape


def startAsynchronouslyPushingGradients(grad,shape,out_nodes,out_shape):
        # Update the gradients on the server
        global params
        decode_grad = np.frombuffer(base64.decodestring(grad),dtype=np.float64)
        decode_grad = np.reshape(decode_grad,shape)
        out_grad = np.frombuffer(base64.decodestring(out_nodes),dtype=np.float64)
        out_grad = np.reshape(out_grad,out_shape)
        params = [decode_grad,out_grad]
        return True

def startAsynchronouslyFetchingParameters():
        global params
        shape = params[0].shape
        encode_params = base64.b64encode(params[0])
        out_shape = params[1].shape
        encode_out = base64.b64encode(params[1])
        return encode_params,shape,encode_out,out_shape

if __name__ == "__main__":
        print "Starting Param Server. Ctrl + C to quit\n"
        server = SimpleXMLRPCServer((HOST,PORT))
        print "Listening on port "+str(PORT)+" with "+str(client_count)+" assumed clients..."
        #server.register_function(getNextMinibatch)
        server.register_function(startAsynchronouslyPushingGradients)
        server.register_function(startAsynchronouslyFetchingParameters)
        server.register_function(get_feature_count)
        server.register_function(get_label_count)
        server.register_function(get_data_shard)
        server.register_function(get_minibatch_size)
	server.register_function(finished_processing)
	server.register_function(finish_client)
        try:
                server.serve_forever()
        except KeyboardInterrupt:
                print "Keyboard interrupt: exiting"



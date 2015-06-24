import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
from math import sqrt
import random
import socket
import numpy as np
import base64

HOST = socket.gethostname()
PORT = 8000

params = None #weights
accruedGradients = None
history_S = [] #s_k = x_kp1 - x_k
history_Y = [] #y_k = gf_kp1 - gf_k
rho = []  #rho_k = 1.0 / (s_k * y_k)

batches_processed = 0
batch_size = 5

#data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=np.float64)
data = np.loadtxt("iris.data",delimiter=",") # Labels must be floats
np.random.shuffle(data)
label_count = len(set(data[:,-1]))
feature_count = len(data[0])-1

def initializeParameters(weights):
	global params
	params = weights

def initializeGradients(len_params):
	global accruedGradients
	accruedGradients = np.zeros(len_params)

def get_label_count(): # This is for setting up multiple labels for the replicas
	return label_count

def get_feature_count():
	return feature_count

def didFinishBatches():
	return batches_processed*batch_size >= len(data)

def getAllData():
	return data

def getDataPortion():
	global batches_processed
	minibatch_start = batches_processed*batch_size
	minibatch_end = min((minibatch_start+batch_size), len(data))
	minibatch = data[minibatch_start:minibatch_end,:]
	batches_processed += 1 
	return minibatch

def sendGradients(localAccruedGrad):
	# Update the gradients on the server
	global accruedGradients
	accruedGradients += localAccruedGrad

def getAccruedGradients():
	return accruedGradients

def getHistory_S():
	return history_S

def getHistory_Y():
	return history_Y

def getRho():
	return rho

def getParameters():
	return params

def initializeParameters(initialParams):
	global params
	params = initialParams

def updateParameters(step_k, d_k, alpha_k, maxHistory, gf_kp1):
	global params, history_S, history_Y, rho

	newParams = params + alpha_k * d_k  #x_kp1 = x_k + alpha_K * d_k

	if(step_k > maxHistory):
		history_S.pop(0)
		history_Y.pop(0)
		rho.pop(0)

	#save new pair
	s_k = newParams - params
	history_S.append(s_k)
	y_k = gf_kp1 - accruedGradients
	history_Y.append(y_k)
	rho.append(1.0 / (np.dot(s_k, y_k)))

	params = newParams #update the weights


if __name__ == "__main__":
	print "Starting Param Server. Ctrl + C to quit\n"
	server = SimpleXMLRPCServer((HOST,PORT))
	print "Listening on port "+str(PORT)+"..."
	server.register_function(didFinishBatches)
	server.register_function(getDataPortion)
	server.register_function(sendGradients)
	server.register_function(getParametersFromParamServer)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print "Keyboard interrupt: exiting"


import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
from math import sqrt
import random
import socket
import numpy as np
import base64
import lbfgs
from neural_net import NeuralNetwork

HOST = socket.gethostname()
PORT = 8000

def sliceData(data):
	# This function assumes np.array as the type for data
	# This function separates data into X (features) and Y (label) for the NN
	x = data[:,:-1]
	labels = data[:,-1] # We don't know how many we have due to minibatch size 
	ys = []
	for l in labels: # This sets up probabilities as outputs | 1 per output class
		temp_y = [0 for i in range(label_count)]
		temp_y[int(l)] = 1 # we can cast this because we know labels are ints and not a weird float
		ys.append(temp_y)
	y = ys
	return x,y

#data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]],dtype=np.float64)
rawData = np.loadtxt("iris.data",delimiter=",") # Labels must be floats
np.random.shuffle(rawData)
label_count = len(set(rawData[:,-1]))
feature_count = len(rawData[0])-1
X, y = sliceData(rawData)
dataSetSize = len(X)

####### WARNING!!! It has to be changed######
NNlayers = [feature_count, 10, label_count]	#
nn = NeuralNetwork(NNlayers)				#
costFunction = nn.cost 						#
jacFunction = nn.jac 	 					#
#############################################

params = nn.get_weights() #weights
accruedGradients = np.zeros(sum(nn.sizes))
maxHistory = 10
history_S = [] #s_k = x_kp1 - x_k
history_Y = [] #y_k = gf_kp1 - gf_k
rho = []  #rho_k = 1.0 / (s_k * y_k)

batches_processed = 0
batch_size = 10

def getNeuralNetLayers():
	return NNlayers

def zeroOutGradients():
	global accruedGradients
	accruedGradients[:] = 0

def get_label_count(): # This is for setting up multiple labels for the replicas
	return label_count

def get_feature_count():
	return feature_count

def didFinishBatches():
	return batches_processed*batch_size >= dataSetSize

def getAllData():
	return X, y

def getDataPortion():
	global batches_processed

	minibatch_start = batches_processed*batch_size
	#Since models replicas run asynchronous, this condicional is needed
		#to do not run out of the bounds
	if(minibatch_start >= dataSetSize):
		return None
	else:
		minibatch_end = min((minibatch_start+batch_size), dataSetSize)
		minibatch = X[minibatch_start:minibatch_end], y[minibatch_start:minibatch_end]
		batches_processed += 1 
		return minibatch

def sendGradients(localAccruedGrad):
	# Update the gradients on the server
	global accruedGradients
	batches_number = dataSetSize / batch_size
	#Dividing the gradients by the number of batches is needed to normalize those values
	accruedGradients += (localAccruedGrad / batches_number)

def getAccruedGradients():
	return accruedGradients

def getParameters():
	return params

def computeLBFGSDirection(step):
	d_k = lbfgs.computeDirection(maxHistory, step, accruedGradients, history_S, history_Y, rho)
	return d_k

def lineSearch(direction_k, fval_x_k, fval_x_km1):
	"""
	returns:
		alpha_k: float or None
			alpha for which x_kp1 = x_k + alpha * d_k, or None if line search algorithm did not converge.
		new_fval : float or None
			New function value f(x_kp1), or None if the line search algorithm did not converge.
		old_fval : float
			Old function value f(fval_x_k).
		gf_kp1 : float or None
			myfprime(x_kp1), or None if the line search algorithm did not converge.
	"""
	alpha_k, new_fval, old_fval, gf_kp1 = lbfgs.lineSearch(costFunction, jacFunction, params, direction_k, accruedGradients, fval_x_k, fval_x_km1, args=(X,y))
	return (alpha_k, new_fval, old_fval, gf_kp1)

def updateParameters(step, d_k, alpha_k, gf_kp1):
	global params, history_S, history_Y, rho

	newParams = params + alpha_k * d_k  #x_kp1 = x_k + alpha_K * d_k

	if(step > maxHistory):
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


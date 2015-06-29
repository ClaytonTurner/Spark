import sys
from SimpleXMLRPCServer import SimpleXMLRPCServer
import random
import socket
import base64
import numpy as np
import lbfgs
from neural_net import NeuralNetwork

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
	y = np.asarray(ys)
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


def processedBatches():
	return batches_processed

def resetParamServer():
	global params, accruedGradients, history_S, history_Y, rho,batches_processed
	
	params = nn.get_weights()
	accruedGradients = np.zeros(sum(nn.sizes))
	history_S = []
	history_Y = []
	rho = []
	batches_processed = 0

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
	shape_x = X.shape
	shape_y = y.shape
	return base64.b64encode(X.tostring()), shape_x, base64.b64encode(y.tostring()), shape_y

def getDataPortion():
	global batches_processed
	minibatch_start = batches_processed*batch_size
	assert minibatch_start < dataSetSize, "minibatch starting out of the bound"
	
	minibatch_end = min((minibatch_start+batch_size), dataSetSize)
	minibatch_x = X[minibatch_start:minibatch_end]
	shape_x = minibatch_x.shape
	minibatch_y = y[minibatch_start:minibatch_end]
	shape_y = minibatch_y.shape
	batches_processed += 1
	return base64.b64encode(minibatch_x.tostring()), shape_x, base64.b64encode(minibatch_y.tostring()), shape_y

def getAccruedGradients():
	return base64.b64encode(accruedGradients)

def getParameters():
	return base64.b64encode(params)

def sendGradients(encodedLocalGrad):
	# Update the gradients on the server
	global accruedGradients
	localGrad = np.frombuffer(base64.decodestring(encodedLocalGrad),dtype=np.float64)
	batches_number = dataSetSize / batch_size
	#Dividing the gradients by the number of batches is needed to normalize those values
	accruedGradients += (localGrad / batches_number)

def computeLBFGSDirection(step):
	d_k = lbfgs.computeDirection(maxHistory, step, accruedGradients, history_S, history_Y, rho)
	return base64.b64encode(d_k)

def lineSearch(encoded_d_k, fval_x_k, fval_x_km1):
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
	d_k = np.frombuffer(base64.decodestring(encoded_d_k),dtype=np.float64)
	alpha_k, new_fval, old_fval, gf_kp1 = lbfgs.lineSearch(costFunction, jacFunction, params, d_k, accruedGradients, fval_x_k, fval_x_km1, args=(X,y))
	return (alpha_k, new_fval, float(old_fval), base64.b64encode(gf_kp1))

def updateParameters(step, encoded_d_k, alpha_k, encoded_gf_kp1):
	global params, history_S, history_Y, rho

	d_k = np.frombuffer(base64.decodestring(encoded_d_k),dtype=np.float64)
	gf_kp1 = np.frombuffer(base64.decodestring(encoded_gf_kp1),dtype=np.float64)
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
	HOST = socket.gethostbyname(socket.gethostname())
	PORT = 8000
	print "Starting Param Server. Ctrl + C to quit\n"
	server = SimpleXMLRPCServer((HOST,PORT), allow_none=True)
	print "Listening on port "+str(PORT)
	server.register_function(resetParamServer)
	server.register_function(getNeuralNetLayers)
	server.register_function(zeroOutGradients)
	server.register_function(get_feature_count)
	server.register_function(get_label_count)
	server.register_function(didFinishBatches)
	server.register_function(getAllData)
	server.register_function(getDataPortion)
	server.register_function(sendGradients)
	server.register_function(getAccruedGradients)
	server.register_function(getParameters)
	server.register_function(computeLBFGSDirection)
	server.register_function(lineSearch)
	server.register_function(updateParameters)
	server.register_function(processedBatches)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print "Keyboard interrupt: exiting"
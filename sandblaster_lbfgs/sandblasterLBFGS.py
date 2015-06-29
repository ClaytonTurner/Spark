import socket
import sys
import random
import xmlrpclib
import base64
import numpy as np
from modelReplica import ModelReplica
from neural_net import NeuralNetwork
from multiprocessing import Process, Queue, Lock

dataLock = Lock()
queue = Queue()

def processPortion(proxy, modelReplica, step):
	"""
	returns False when there isnt any portion to be processed,
	else returns True
	"""
	if(not modelReplica.hasParametersForStep(step)):
		encoded_params = proxy.getParameters()
		params = np.frombuffer(base64.decodestring(encoded_params), dtype=np.float64)
		modelReplica.setParams(params, step)

	#lock avoids more than one replica processing a same subset of data
	dataLock.acquire()
	if(not proxy.didFinishBatches()):
		print 'before', proxy.processedBatches()
		encoded_x, shape_x, encoded_y, shape_y = proxy.getDataPortion()
		print 'after', proxy.processedBatches()
		dataLock.release()
	
		x = np.frombuffer(base64.decodestring(encoded_x),dtype=np.float64).reshape(shape_x)
		y = np.frombuffer(base64.decodestring(encoded_y),dtype=np.float64).reshape(shape_y)

		gradients = modelReplica.computeGradient(x, y)
		modelReplica.updateAccruedGradients(gradients)
		return True
	else:
		return False

def runModelReplica(proxy, replica, step):
	wasPortionProcessed =  processPortion(proxy, replica, step)
	#lets start fixing timeToSendGradients always to True.
	#WARNING!!!! it should be changed later
	if(wasPortionProcessed):
		localGrad = replica.getLocalAccruedGrad()
		proxy.sendGradients(base64.b64encode(localGrad))
		replica.accruedGradients[:] = 0
	else:
		queue.close()
	
	queue.put(replica)
	return wasPortionProcessed



if (__name__ == "__main__"):
	HOST = socket.gethostbyname(socket.gethostname())
	PARAM_SERVER,PORT = HOST,8000
	   
	proxy = xmlrpclib.ServerProxy("http://"+PARAM_SERVER+":"+str(PORT)+"/",allow_none=True)

	proxy.resetParamServer()

	neuralNetLayers = proxy.getNeuralNetLayers()
	modelReplicas = [ModelReplica(neuralNetLayers), ModelReplica(neuralNetLayers), ModelReplica(neuralNetLayers), ModelReplica(neuralNetLayers)]

	old_fval = None
	old_old_fval = None
	
	step = 0
	gtol = 1e-5

	queueLock = Lock()
	for replica in modelReplicas:
		queue.put(replica)

	while(step < 500):
		proxy.zeroOutGradients()
		proxy.batches_processed = 0
	
		processes = []

		while(not proxy.didFinishBatches()):
		
			replica = queue.get()
			runModelReplica(proxy, replica, step)
			# process = Process(target=runModelReplica, args=(replica, step))
			# processes.append(process)
			# process.start()
		
		# for process in processes:
		# 	process.join()

		encoded_d_k = proxy.computeLBFGSDirection(step)
		
		alpha_k, old_fval, old_old_fval, encoded_gf_kp1 = \
						proxy.lineSearch(encoded_d_k, old_fval, old_old_fval)

		if(alpha_k is None): # Line search failed to find a better solution.
			print "Stopped because line search did not converge"
			break

		proxy.updateParameters(step, encoded_d_k, alpha_k, encoded_gf_kp1)


		encoded_grads = proxy.getAccruedGradients()
		accruedGradients = np.frombuffer(base64.decodestring(encoded_grads), dtype=np.float64)
		if(np.linalg.norm(accruedGradients, np.inf) < gtol):
			print "converged!!"
			break

		step += 1


	print step
	encoded_x, shape_x, encoded_y, shape_y = proxy.getAllData()
	X = np.frombuffer(base64.decodestring(encoded_x),dtype=np.float64).reshape(shape_x)
	y = np.frombuffer(base64.decodestring(encoded_y),dtype=np.float64).reshape(shape_y)

	nn = NeuralNetwork(neuralNetLayers)
	encoded_params = proxy.getParameters()
	params = np.frombuffer(base64.decodestring(encoded_params), dtype=np.float64)
	print nn.cost(params, X, y)

	nn.set_weights(params)
	correct = 0
	for i, e in enumerate(X):
		#print(e,nn.predict(e))
		prediction = list(nn.predict(e))
		#print "Label: ",y[i]," | Predictions: ",prediction
		if prediction.index(max(prediction)) == np.argmax(y[i]):
			correct += 1
	print "Correct: ",correct,"/",i,"(",float(correct)/float(i),"%)"
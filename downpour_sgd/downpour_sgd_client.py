import socket
import sys
from math import sqrt
import random
import xmlrpclib
from neural_net import NeuralNetwork
import numpy as np
import base64

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

def slice_data(data,shape,lbl_cnt):
	# This function assumes np.array as the type for data
	# This function separates data into X (features) and Y (label) for the NN
	# Check if c orientation is fine, if not switch to fortran (check numpy docs)
	data = np.reshape(data,shape)
	
	labels = data[:,-1] # We don't know how many we have due to minibatch size 
	ys = []
	for l in labels:
		temp_y = [0 for x in range(lbl_cnt)]#np.zeros(lbl_cnt)
		temp_y[int(l)] = 1 # we can cast this because we know labels are ints and not a weird float
		ys.append(temp_y)#[0])
	#y = np.zeros(lbl_cnt)
	#y = np.matrix(ys)
	y = ys
	print len(y)
	print y[0]
	#y[label] = 1
	
	x = data[:,:-1]
	#y = data[:,-1]
	return x,y

if __name__ == "__main__":
        #data_file = "/data/spark/Spark/iris_labelFirst.data"

	step = 0
	
	HOST = socket.gethostname()
	PARAM_SERVER,PORT = "192.168.137.50",8000
        
	proxy = xmlrpclib.ServerProxy("http://"+PARAM_SERVER+":"+str(PORT)+"/",allow_none=True)
	feature_count = proxy.get_feature_count()
	label_count = proxy.get_label_count()

	random.seed(8000) # should allow stabilization across machines
			  # Removes need for initial weight fetch
	layers = [feature_count,2,label_count] # layers - first is input layer. last is output layer. rest is hidden.
	nn = NeuralNetwork(layers)#,activation='sigmoid')
	grad_push_errors = 0
        while(True):# Going to change up later for minibatch counts
                if step%n_fetch == 0 and step > 0: # Always true in fixed case | step > 0 since init happens with the first nn.fit
			parameters,param_shape,out,out_shape = proxy.startAsynchronouslyFetchingParameters()
			parameters = np.reshape(np.frombuffer(base64.decodestring(parameters),dtype=np.float64),param_shape)
			out = np.reshape(np.frombuffer(base64.decodestring(out),dtype=np.float64),out_shape)
			#print nn.weights
			nn.set_weights([parameters,out])
			#print nn.weights
		data,shape = proxy.getNextMinibatch()
		if(data == "Done"):
			if step == 0:
				print "No data provided to replica. Exiting..."
				import sys
				sys.exit()
			break
		data = np.frombuffer(base64.decodestring(data),dtype=np.float64) #.fromstring()
		x,y = slice_data(data,shape,label_count)#,label_count)
		nn.fit(x,y,epochs=1000)
		accrued_gradients = compute_gradient(nn)
		if step%n_push == 0: # Always true in fixed case
			ag_shape = accrued_gradients[0].shape
			accrued_nodes = base64.b64encode(accrued_gradients[0].tostring())
			output_shape = accrued_gradients[1].shape
			output_nodes = base64.b64encode(accrued_gradients[1].tostring())# I THINK these are the output nodes
			if proxy.startAsynchronouslyPushingGradients(accrued_nodes,ag_shape,output_nodes,output_shape) is not True:
				grad_push_errors += 1
                step += 1
	print "Neural Network Replica trained with "+str(grad_push_errors)+" Gradient errors\n"
	'''
	At this point, our NN replica is done. We need to make sure all replicas have completed
	Then we can pull down a final set of weights for a trained model for prediction
	'''	
	
	#x = np.array([[0,0],[0,1],[1,0],[1,1]])
	x = np.loadtxt("iris.data",delimiter=",")[:,:-1]
	for e in x:
		print(e,nn.predict(e))
 ##gradient = sc.parallelize(data, numSlices=slices) \
 ##       .mapPartitions(lambda x: computeGradient(parameters,x) \
 ##       .reduce(lambda x, y: merge(x,y))
                

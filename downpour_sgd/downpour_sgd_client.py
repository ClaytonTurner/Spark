import socket
import sys
from math import sqrt
import random
import xmlrpclib
from adagrad_nn import NeuralNetwork
import numpy as np
import base64

n_fetch = 1 # fixed in the paper, so let's leave it that way here
n_push = 1 # same as n_fetch
batches_processed = 0 # just for this replica
FINISH_TIMEOUT = 120.0 # How many seconds to wait after a replica is finished 
		      #		before processing if all training isn't done

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
	for l in labels: # This sets up probabilities as outputs | 1 per output class
		temp_y = [0 for x in range(lbl_cnt)]
		temp_y[int(l)] = 1 # we can cast this because we know labels are ints and not a weird float
		ys.append(temp_y)#[0])
	y = ys
	
	x = data[:,:-1]
	#y = data[:,-1]
	return x,y

def fetch_and_set_weights(proxy,nn):
	parameters,param_shape,out,out_shape = proxy.startAsynchronouslyFetchingParameters()
        parameters = np.reshape(np.frombuffer(base64.decodestring(parameters),dtype=np.float64),param_shape)
        out = np.reshape(np.frombuffer(base64.decodestring(out),dtype=np.float64),out_shape)
        nn.set_weights([parameters,out])

def getNextMinibatch(data_shard):
        global batches_processed
        minibatch_start = batches_processed*batch_size
        minibatch_end = minibatch_start+batch_size
        if(minibatch_end > len(data_shard)):
                return "Done","Done"
        else:
                minibatch = data_shard[minibatch_start:(minibatch_start+batch_size),:]
                batches_processed += 1
                return minibatch,minibatch.shape



if __name__ == "__main__":
        #data_file = "/data/spark/Spark/iris_labelFirst.data"

	step = 0
	
	HOST = socket.gethostname()
	PARAM_SERVER,PORT = "192.168.137.62",49151
       
	proxy = xmlrpclib.ServerProxy("http://"+PARAM_SERVER+":"+str(PORT)+"/",allow_none=True)
        feature_count = proxy.get_feature_count()
        label_count = proxy.get_label_count()
        batch_size = proxy.get_minibatch_size()

        #random.seed(8000) # should allow stabilization across machines
                          # Removes need for initial weight fetch
        layers = [feature_count,10,label_count] # layers - first is input layer. last is output layer. rest is hidden.
        nn = NeuralNetwork(layers)#,activation='sigmoid')
        data_shard,shard_shape = proxy.get_data_shard()
        data_shard = np.frombuffer(base64.decodestring(data_shard),dtype=np.float64)
        data_shard = np.reshape(data_shard,shard_shape)

        grad_push_errors = 0
        while(True):# Going to change up later for minibatch counts
                if step%n_fetch == 0 and step > 0: # Always true in fixed case | step > 0 since init happens with the first nn.fit
                        '''parameters,param_shape,out,out_shape = proxy.startAsynchronouslyFetchingParameters()
                        parameters = np.reshape(np.frombuffer(base64.decodestring(parameters),dtype=np.float64),param_shape)
                        out = np.reshape(np.frombuffer(base64.decodestring(out),dtype=np.float64),out_shape)
                        #print nn.weights
                        nn.set_weights([parameters,out])
                        #print nn.weights'''
                        fetch_and_set_weights(proxy,nn)
                data,shape = getNextMinibatch(data_shard)
                if(data == "Done"):
                        if step == 0:
                                print "No data provided to replica. Exiting..."
                                import sys
                                sys.exit()
                        else:
                                print "Replica finished processing data shard minibatches. Waiting for testing."
			proxy.finish_client()
                        import time
			start_wait = time.time()
			while(not proxy.finished_processing() and time.time() - start_wait < FINISH_TIMEOUT):
				print "All replicas not complete. " \
					"Performing sleep(5) until timeout " \
					"or until all done"
                        	time.sleep(5) # When deployed, replace this with a wait from the param server
                        fetch_and_set_weights(proxy,nn) # Final weight fetch for each model
                        break
                #data = np.frombuffer(base64.decodestring(data),dtype=np.float64) #.fromstring()
                x,y = slice_data(data,shape,label_count)#,label_count)
                nn.fit(x,y,learning_rate=0.2,epochs=100000)
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
	f = np.loadtxt("iris.data",delimiter=",")
	x = f[:,:-1]
	y = f[:,-1]
	correct = 0
	i = 0
	for e in x:
		#print(e,nn.predict(e))
		prediction = list(nn.predict(e))
		#print "Label: ",y[i]," | Predictions: ",prediction
		if prediction.index(max(prediction)) == y[i]:
			correct += 1
		i += 1
	print "Correct: ",correct,"/",i,"(",float(correct)/float(i),"%)"
 ##gradient = sc.parallelize(data, numSlices=slices) \
 ##       .mapPartitions(lambda x: computeGradient(parameters,x) \
 ##       .reduce(lambda x, y: merge(x,y))
                

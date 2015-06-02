import socket
import sys
from math import sqrt
import random
import xmlrpclib

learning_rate = 0.1 # tune later on
n_fetch = 1 # fixed in the paper, so let's leave it that way here
n_push = 1 # same as n_fetch

'''
Parameter and Gradient initialization will now happen on the param server
we can do an initial read from all replicas if needbe, but there is no
	reason to re-calculate that information for each replica
	if it will be the same from the start
'''

def compute_gradient(params,alpha):
	return None

if __name__ == "__main__":
        slices = 10 # Arbitrary - Empirically tune for performance

        #data_file = "/data/spark/Spark/iris_labelFirst.data"

        step = 0
	
	HOST = socket.gethostname()
	PORT, SIZE = 8000, 1024
        
	proxy = xmlrpclib.ServerProxy("http://localhost:"+str(PORT)+"/",allow_none=True)

	#TODO NN initialization
	grad_push_errors = 0
        while(step < 5):
                if step%n_fetch == 0: # Always true in fixed case
                        parameters = proxy.startAsynchronouslyFetchingParameters()
                data = proxy.getNextMinibatch()
		accrued_gradients = compute_gradient(parameters,learning_rate)
                #parameters -= alpha*gradient #TODO as parameters is currently a dictionary
                if step%n_push == 0: # Always true in fixed case
			if proxy.startAsynchronouslyPushingGradients(accrued_gradients) is not True:
				grad_push_errors += 1
                step += 1
	print "Gradient Pushing Errors: "+str(grad_push_errors)+"\n"

 ##gradient = sc.parallelize(data, numSlices=slices) \
 ##       .mapPartitions(lambda x: computeGradient(parameters,x) \
 ##       .reduce(lambda x, y: merge(x,y))
                

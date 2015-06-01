import downpour_sgd as d_sgd
import socket
import sys
from math import sqrt
import random
import xmlrpclib

learning_rate = 0.1 # tune later on
n_fetch = 1 # fixed in the paper, so let's leave it that way here
n_push = 1 # same as n_fetch

if __name__ == "__main__":
        slices = 10 # Arbitrary - Empirically tune for performance

        #data_file = "/data/spark/Spark/iris_labelFirst.data"
        #data_file = str(sys.argv[1])

        step = 0
	###sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP socket

        accrued_gradients = init_accrued_gradients()
	
	HOST = socket.gethostname()
	PORT, SIZE = 45001, 1024
        
	#TODO NN initialization

        while(step < 5):
		## Move to different file later
		# Need here later, but not ready to test
                
		if step == 0:
                        parameters = init_parameters()

		proxy = xmlrpclib.ServerProxy("http://localhost:8000/")
		print(proxy.test_func([1,2,3]))	

                #if step%n_fetch == 0: # Always true in fixed case
                #        startAsynchronouslyFetchingParameters(parameters)
                #data = d_sgd.getNextMinibatch(data)
                ##gradient = sc.parallelize(data, numSlices=slices) \
                ##       .mapPartitions(lambda x: computeGradient(parameters,x) \
                ##       .reduce(lambda x, y: merge(x,y))
                #gradient = d_sgd.avg_model(gradient, slices)
                #d_sgd.set_accrued_gradients(gradient)
                #parameters -= alpha*gradient #TODO as parameters is currently a dictionary
                #if step%n_push == 0: # Always true in fixed case
                #        startAsynchronouslyPushingGradients(accrued_gradients)
                step += 1
	#sock.close()





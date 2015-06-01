import downpour_sgd as d_sgd
import socket
import sys
from math import sqrt
import random
import xmlrpclib

learning_rate = 0.1 # tune later on
n_fetch = 1 # fixed in the paper, so let's leave it that way here
n_push = 1 # same as n_fetch

def init_parameters(data_len=5): # tweek that later. just trying to pass interpretation error-less
        hidden_layers = [100,100,100]
        weight_dist = 1./sqrt(data_len)
        weights = [[random.uniform(-weight_dist,weight_dist) for y in hidden_layers] for x in range(len(hidden_layers)+1)]
        # so we have 4 sets of weights since we need to set up the output layer
        # Now we push the initialized updates to the param server
        return {"hidden_layers":hidden_layers,"weights":weights}

def init_accrued_gradients():
	#TODO
	return 0.

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
		print("Output should be 6\n")
		print(proxy.test_func([1,2,3])	

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
	sock.close()





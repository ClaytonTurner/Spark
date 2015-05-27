import sys
from pyspark import SparkContext
from sklearn import linear_model as lm # for gradient descent; adapted from https://gist.github.com/MLnick/4707012
from sklearn.base import copy
import random
from math import sqrt
import mlp
import numpy
import my_communication as comm

#TODO
# Add adagrad

accrued_gradients = 0. # initialize the same way we reset later
SERVER_IP = "192.168.137.50"
PARAM_PORT = 45001
GRAD_PORT = 45002

def get_accrued_gradients():
	data = comm.receive_data(GRAD_PORT)
	return data

def set_accrued_gradients(grad):
	# Modify accrued_gradients will values local to just one replica
	#TODO

def set_parameters(params):
	comm.send_data(params,PARAM_PORT)

def getParametersFromParamServer():
	data = comm.receive_data(PARAM_PORT)
	return data

def startAsynchronouslyFetchingParameters(parameters):
	params = getParametersFromParamServer()
	set_parameters(parameters)

def startAsynchronouslyPushingGradients(accrued_gradients):
	sendGradientsToParamServer(accrued_gradients)
	accrued_gradients = 0 # reset accrued_gradients

def sendGradientsToParamsServer(accrued_gradients):
	#set_accrued_gradients(accrued_gradients)
	comm.send_data(accrued_gradients,GRAD_PORT)

def getNextMinibatch(data):
	#TODO
	yield data

def computeGradient(parameters,data):
	#TODO
	sgd = lm.SGDClassifier(loss='log') # initialize SGD
	
def merge(left, right): # Part of aforementioned adaptation
	new = copy.deepcopy(left)
	new.coef_ += right.coef_
	new.intercept_ += right.intercept_
	return new

def avg_model(sgd, slices): # Part of aforementioned adaptation
	sgd.coef_ /= slices
	sgd.intercept_ /= slices
	return sgd

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print >> sys.stderr, "Usage: spark_sgd.py <data_file> <alpha> <n_fetch> <n_push>"
		exit(-1)

	slices = 10 # Arbitrary - Empirically tune for performance

	#data_file = "/data/spark/Spark/iris_labelFirst.data" 	
	data_file = str(sys.argv[1])
	alpha = float(sys.argv[2])
	n_fetch = 1 #int(sys.argv[3]) # Google fixed to 1 in paper
	n_push = 1 #int(sys.argv[4]) # Google fixed to 1 in paper

	step = 0
	accrued_gradients = get_accrued_gradients()

	#TODO NN initialization

	while(True):
		if step == 0:
			parameters = init_parameters()
		if step > 1000: # This can be tweaked
			break 
		if step%n_fetch == 0: # Always true in fixed case
			startAsynchronouslyFetchingParameters(parameters)
		data = getNextMinibatch(data)
		#gradient = sc.parallelize(data, numSlices=slices) \
		#	.mapPartitions(lambda x: computeGradient(parameters,x) \
		#	.reduce(lambda x, y: merge(x,y))
		gradient = avg_model(gradient, slices)
		set_accrued_gradients(gradient)
		parameters -= alpha*gradient #TODO as parameters is currently a dictionary
		if step%n_push == 0: # Always true in fixed case
			startAsynchronouslyPushingGradients(accrued_gradients)
		step += 1



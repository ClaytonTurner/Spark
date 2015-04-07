import sys
from pyspark import SparkContext
from sklearn import linear_model as lm # for gradient descent; adapted from https://gist.github.com/MLnick/4707012
from sklearn.base import copy

parameters = None #TODO
accrued_gradients = 0.

def get_accrued_gradients():
	global accrued_gradients # Best to be sure of what I'm returning
	return accrued_gradients

def set_accrued_gradients(grad):
	global accrued_gradients # Needed to modify accrued_gradients globally
	accrued_gradients += grad

def set_parameters(params):
	global parameters # Needed to modify parameters globally
	parameters = params

#def getParametersFromParamServer():
	# Google hosted a separate server for hosting parameters
	# We use broadcast variables instead so we don't need
	# 	this function

def startAsynchronouslyFetchingParameters(parameters):
	#params = getParametersFromParamServer()
	set_parameters(parameters)

def startAsynchronouslyPushingGradients(accrued_gradients):
	sendGradientsToParamServer(accrued_gradients)
	accrued_gradients = 0

def sendGradientsToParamsServer(accrued_gradients):
	#TODO

def getNextMinibatch():
	#TODO

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

	worker_count = 4 # Not sure what to do about this

	#data_file = "/data/spark/Spark/iris_labelFirst.data" 	
	data_file = str(sys.argv[1])
	alpha = float(sys.argv[2])
	n_fetch = int(sys.argv[3]) # Google fixed to 1 in paper
	n_push = int(sys.argv[4]) # Google fixed to 1 in paper

	sc = SparkContext(appName="Spark SGD")
	cached_data = sc.textFile(logFile).cache()
	broadcast_parameters = sc.broadcast(parameters)		

	step = 0
	accrued_gradients = get_accrued_gradients()
	while(True):
		#TODO - stop after X steps?
		if step%n_fetch == 0: # Always true in fixed case
			startAsynchronouslyFetchingParameters(broadcast_parameters)
		data = getNextMinibatch()
		gradient = sc.parallelize(data) \
			.mapPartitions(lambda x: computeGradient(broadcast_parameters,x) \
			.reduce(lambda x, y: merge(x,y))
		gradient = avg_model(gradient, worker_count)
		set_accrued_gradients(gradient)
		broadcast_parameters -= alpha*gradient #TODO
		if step%n_push == 0: # Always true in fixed case
			startAsynchronouslyPushingGradients(accrued_gradients)
		step += 1

	sc.stop()

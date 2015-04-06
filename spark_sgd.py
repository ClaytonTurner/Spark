import sys
from pyspark import SparkContext

parameters = None #TODO
accrued_gradients = 0.

def get_accrued_gradients():
	return accrued_gradients

def set_accrued_gradients(grad):
	global accrued_gradients # Needed to modify accrued_gradients globally
	accrued_gradients += grad

def set_parameters(params):
	global parameters
	parameters = params

def getParametersFromParamServer():
	#TODO

def startAsynchronouslyFetchingParameters(parameters):
	#TODO - probably needs signature change somewhere
	params = getParametersFromParamServer()
	set_parameters(params)

def startAsynchronouslyPushingGradients(accrued_gradients):
	sendGradientsToParamServer(accrued_gradients)
	accrued_gradients = 0

def sendGradientsToParamsServer(accrued_gradients):
	#TODO

def getNextMinibatch():
	#TODO

def computeGradient(parameters,data):
	#TODO

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print >> sys.stderr, "Usage: spark_sgd.py <data_file> <alpha> <n_fetch> <n_push>"
		exit(-1)

	#data_file = "/data/spark/Spark/iris_labelFirst.data" 	
	data_file = str(sys.argv[1])
	alpha = float(sys.argv[2])
	n_fetch = int(sys.argv[3]) # Google fixed to 1 in paper
	n_push = int(sys.argv[4]) # Google fixed to 1 in paper

	sc = SparkContext(appName="Spark SGD")
	cached_data = sc.textFile(logFile).cache()
		
	step = 0
	accrued_gradients = get_accrued_gradients()
	while(True):
		#TODO - stop after X steps?
		if step%n_fetch == 0:
			startAsynchronouslyFetchingParameters(parameters)
		data = getNextMinibatch()
		gradient = computeGradient(parameters,data)
		set_accrued_gradients(gradient)
		parameters -= alpha*gradient #TODO
		if step%n_push == 0:
			startAsynchronouslyPushingGradients(accrued_gradients)
		step += 1

	sc.stop()

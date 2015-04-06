import sys
from pyspark import SparkContext

parameters = None #TODO
accrued_gradients = 0

def get_accrued_gradients():
	return accrued_gradients

def set_accrued_gradients(grad):
	global accrued_gradients # Needed to modify accrued_gradients globally
	accrued_gradients += grad

def startAsynchronouslyFetchingParameters(parameters):
	#TODO

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
	data_file = sys.argv[1]
	alpha = sys.argv[2]
	n_fetch = sys.argv[3]
	n_push = sys.argv[4]

	sc = SparkContext(appName="Spark SGD")
	cached_data = sc.textFile(logFile).cache()
		
	step = 0
	accrued_gradients = get_accrued_gradients()
	while(True):
		#TODO
		if step%n_fetch == 0:
			startAsynchronouslyFetchingParameters(parameters)
		data = getNextMinibatch()
		gradient = computeGradient(parameters,data)
		set_accrued_gradients(gradient)
		parameters = 

	sc.stop()

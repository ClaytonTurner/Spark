from pyspark import SparkContext

parameters = None 
accrued_gradients = 0

def get_accrued_gradients():
	return accrued_gradients
def set_accrued_gradients(grad):
	global accrued_gradients # Needed to modify accrued_gradients globally
	accrued_gradients += grad

def startAsyncFetchParams(parameters):

if __name__ == "__main__":
	dataFile = "/data/spark/Spark/iris_labelFirst.data" 	
	sc = SparkContext(appName="Spark SGD")
	data = sc.textFile(logFile).cache()
		
	step = 0
	accrued_gradients = 0
	while(True):
		if step%n_fetch == 0:
			startAsyncFetchParams(parameters)

	sc.stop()

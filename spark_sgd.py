from pyspark import SparkContext

dataFile = "/data/spark/Spark/iris_labelFirst.data" #this is located on spark6a (the master node)
sc = SparkContext(appName="Spark SGD")
data = sc.textFile(logFile).cache()

#if __name__ == "__main__":
#	def myFunc(s):
#		words = s.split(",")
#		return len(words)

from pyspark import SparkContext, SparkConf


def main():
	appname = "spark_file_io"
	master = "spark://10.0.3.114:7077"
	#conf = SparkConf().setAppName(appname).setMaster(master)
	conf = SparkConf().setAppName(appname)
	sc = SparkContext(conf=conf)
	
	distFile = sc.textFile("../example_datasets/iris.data")
	lineLengths = distFile.map(lambda s: len(s))
	totalLength = lineLengths.reduce(lambda a, b: a + b)
	print("Total Length of lines: " + str(totalLength))
main()

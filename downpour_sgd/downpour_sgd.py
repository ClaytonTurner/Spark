import sys
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
	comm.send_data(params,PARAM_PORT,SERVER_IP)

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
	comm.send_data(accrued_gradients,GRAD_PORT,SERVER_IP)

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


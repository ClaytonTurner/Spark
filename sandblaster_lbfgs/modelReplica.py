import numpy as np
from neural_net import NeuralNetwork


class ModelReplica:

	def __init__(self, neuralNetLayers):
		self.neuralNet = NeuralNetwork(neuralNetLayers)
		self.currentParamsStep = None
		self.accruedGradients = np.zeros(sum(self.neuralNet.sizes))

	def hasParametersForStep(self, step):
		return step == self.currentParamsStep

	def setParams(self, params, step):
		self.params = params
		self.currentParamsStep = step

	def getParams(self, step):
		assert self.currentParamsStep == step, "params are not up to date"
		return self.params

	def updateAccruedGradients(self, newGrad):
		self.accruedGradients += newGrad

	def getLocalAccruedGrad(self):
		return self.accruedGradients

	def computeGradient(self, data):
		X, y = data 
		gradients = self.neuralNet.jac(self.params, X, y)
		return gradients

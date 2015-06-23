import numpy as np


class ModelReplica:

	def __init__(self, len_params):
		self.currentParamsStep = None
		self.accruedGradients = np.zeros(len_params)

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


import numpy as np
import paramServer as PS
from modelReplica import ModelReplica
from neural_net import NeuralNetwork

def processPortion(modelReplica, step):
	if(not modelReplica.hasParametersForStep(step)):
		params = PS.getParameters()
		modelReplica.setParams(params, step)

	data = PS.getDataPortion()
	if(data is None):
		return False
	else:
		gradients = modelReplica.computeGradient(data)
		modelReplica.updateAccruedGradients(gradients)
		return True


if (__name__ == "__main__"):

	neuralNetLayers = PS.getNeuralNetLayers()
	modelReplicas = [ModelReplica(neuralNetLayers), ModelReplica(neuralNetLayers)]

	old_fval = None
	old_old_fval = None
	
	step = 0
	gtol = 1e-5
	
	while(step < 500):
		PS.zeroOutGradients()
		PS.batches_processed = 0

		while(not PS.didFinishBatches()):
			for replica in modelReplicas:

				#should verify if a portion was processed properly
				processPortion(replica, step)
					
				#workDone
				localGrad = replica.getLocalAccruedGrad()
				PS.sendGradients(localGrad)
				replica.accruedGradients[:] = 0

		direction_k = PS.computeLBFGSDirection(step)
		alpha_k, old_fval, old_old_fval, gf_kp1 = \
						PS.lineSearch(direction_k, old_fval, old_old_fval)

		if(alpha_k is None): # Line search failed to find a better solution.
			print "Stopped because line search did not converge"
			break

		PS.updateParameters(step, direction_k, alpha_k, gf_kp1)

		if(np.linalg.norm(PS.getAccruedGradients(), np.inf) < gtol):
			print "converged!!"
			break

		step += 1


	print step
	X, y = PS.getAllData()
	nn = NeuralNetwork(neuralNetLayers)
	print nn.cost(PS.getParameters(), X, y)

	nn.set_weights(PS.getParameters())
	correct = 0
	for i, e in enumerate(X):
		#print(e,nn.predict(e))
		prediction = list(nn.predict(e))
		#print "Label: ",y[i]," | Predictions: ",prediction
		if prediction.index(max(prediction)) == y[i].index(max(y[i])):
			correct += 1
	print "Correct: ",correct,"/",i,"(",float(correct)/float(i),"%)"
import numpy as np
import paramServer as PS
from modelReplica import ModelReplica
from neural_net import NeuralNetwork
import lbfgs



# def sliceData(data):
# 	# This function assumes np.array as the type for data
# 	# This function separates data into X (features) and Y (label) for the NN
# 	x = data[:,:-1]
# 	label_count = PS.get_label_count()
# 	labels = data[:,-1] # We don't know how many we have due to minibatch size 
# 	ys = []
# 	for l in labels: # This sets up probabilities as outputs | 1 per output class
# 		temp_y = [0 for i in range(label_count)]
# 		temp_y[int(l)] = 1 # we can cast this because we know labels are ints and not a weird float
# 		ys.append(temp_y)
# 	y = ys
# 	return x,y

def computeGradient(nn, weights, data):
	X, y = data 
	gradients = nn.jac(weights, X, y)
	return gradients

def processPortion(modelReplica, step, nn):
	if(not modelReplica.hasParametersForStep(step)):
		params = PS.getParameters()
		modelReplica.setParams(params, step)
	else:
		params = modelReplica.getParams(step)

	data = PS.getDataPortion()
	if(data is None):
		return False
	else:
		gradients = computeGradient(nn, params, data)
		modelReplica.updateAccruedGradients(gradients)
		return True




if (__name__ == "__main__"):

	feature_count = PS.get_feature_count()
	label_count = PS.get_label_count()
	layers = [feature_count, 10, label_count] # layers - 1 = hidden layers
	
	nn = NeuralNetwork(layers)
	costFunction = nn.cost
	jacFunction = nn.jac
	len_params = sum(nn.sizes)
	modelReplicas = [ModelReplica(len_params), ModelReplica(len_params)]


	X, y = PS.getAllData()
	old_fval = costFunction(PS.getParameters(), X, y)
	old_old_fval = None
	
	step = 0
	gtol = 1e-5
	
	while(step < 500):
		PS.zeroOutGradients()
		PS.batches_processed = 0

		while(not PS.didFinishBatches()):
			for replica in modelReplicas:
				if(processPortion(replica, step, nn)):
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
	print costFunction(PS.getParameters(), X, y)

	nn.set_weights(PS.getParameters())
	correct = 0
	for i, e in enumerate(X):
		#print(e,nn.predict(e))
		prediction = list(nn.predict(e))
		#print "Label: ",y[i]," | Predictions: ",prediction
		if prediction.index(max(prediction)) == y[i].index(max(y[i])):
			correct += 1
	print "Correct: ",correct,"/",i,"(",float(correct)/float(i),"%)"
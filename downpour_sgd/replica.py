import downpour_sgd as d_sgd


learning_rate = 0.1 # tune later on
n_fetch = 1 # fixed in the paper, so let's leave it that way here
n_push = 1 # same as n_fetch

def init_parameters(data_len):
        hidden_layers = [100,100,100]
        weight_dist = 1./sqrt(data_len)
        weights = [[random.uniform(-weight_dist,weight_dist) for y in hidden_layers] for x in range(len(hidden_layers)+1)]
        # so we have 4 sets of weights since we need to set up the output layer
        # Now we push the initialized updates to the param server
        return {"hidden_layers":hidden_layers,"weights":weights}

def init_accrued_gradients():
	#TODO
	return 0.

if __name__ == "__main__":
        slices = 10 # Arbitrary - Empirically tune for performance

        #data_file = "/data/spark/Spark/iris_labelFirst.data"
        data_file = str(sys.argv[1])

        step = 0
        accrued_gradients = init_accrued_gradients()

        #TODO NN initialization

        while(True):
                if step == 0:
                        parameters = init_parameters()
                if step > 1000: # This can be tweaked
                        break
                if step%n_fetch == 0: # Always true in fixed case
                        startAsynchronouslyFetchingParameters(parameters)
                data = getNextMinibatch(data)
                #gradient = sc.parallelize(data, numSlices=slices) \
                #       .mapPartitions(lambda x: computeGradient(parameters,x) \
                #       .reduce(lambda x, y: merge(x,y))
                gradient = avg_model(gradient, slices)
                set_accrued_gradients(gradient)
                parameters -= alpha*gradient #TODO as parameters is currently a dictionary
                if step%n_push == 0: # Always true in fixed case
                        startAsynchronouslyPushingGradients(accrued_gradients)
                step += 1




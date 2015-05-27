

def init_parameters(data_len):
        hidden_layers = [100,100,100]
        weight_dist = 1./sqrt(data_len)
        weights = [[random.uniform(-weight_dist,weight_dist) for y in hidden_layers] for x in range(len(hidden_layers)+1)]
        # so we have 4 sets of weights since we need to set up the output layer
        # Now we push the initialized updates to the param server
        return {"hidden_layers":hidden_layers,"weights":weights}


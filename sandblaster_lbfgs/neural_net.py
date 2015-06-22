import numpy as np
import scipy.optimize as opti
import lbfgs


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers):
        self.activation = tanh
        self.activation_prime = tanh_prime

        # compute weight layer sizes and shapes
        self.sizes = [] # store weight layer divisions in the flattened weight array
        self.shapes = [] # store weight layer shapes for reshaping
        # input and hidden layers - (2+1, 2+1) : 3 x 3
        for i in range(1, len(layers)-1):
            self.sizes.append((layers[i-1] + 1) * (layers[i] + 1))
            self.shapes.append((layers[i-1] + 1, layers[i] + 1))
        
        # output layer - (2+1, 1) : 3 x 1
        self.sizes.append((layers[i] + 1) * layers[i+1])
        self.shapes.append((layers[i] + 1, layers[i+1]))

        self.weights = 2 * np.random.rand(sum(self.sizes)) - 1

    def get_weights(self):
        return self.weights

    def set_weights(self,params):
        self.weights = params

    def unpack_parameters(self, param):
        """
        Extracts and returns the parameter array for each layer in the network.
        """
        params = []
        i = 0 # store flattened theta array index value from previous iteration
        for j,s in zip(self.sizes, self.shapes):
            params.append(param[i:i+j].reshape(s[0], s[1])) # get current layers theta matrix
            i += j # record the flattened array index for the end of current layer
        return params

    def forward_prop(self, x_, weights):

        # iterate through each layer in the network computing and forward propagating activation values
        x = np.hstack((1, x_)) # add bias unit with value 1.0
        a = [x] # store activation values for each hidden and output layer node

        for i in range(len(weights)):
            a_ = self.activation(np.dot(a[i], weights[i]))
            a.append(a_) # record current layer activation values
        return a, a_

    def backward_prop(self, y, activations, weights):
        
        # delta_L (prediction error in output layer)
        error = activations[-1] - y
        deltas = [error * self.activation_prime(activations[-1])] 

        # iterate through layer activation values in reverse order computing delta_l
        for i in reversed(range(1, len(weights))):
            deltas.append(deltas[-1].dot(weights[i].T)*self.activation_prime(activations[i]))

        deltas.reverse()
        
        # backpropagation
        # 1. Multiply its output delta and input activation 
        #    to get the gradient of the weight.
        grad = np.array(())
        for i in range(len(weights)):
            layer = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            dot_value  = layer.T.dot(delta)
            grad = np.hstack((grad, dot_value.flatten()))

        return grad

    def jac(self, weights_flattened, X, y):
        m = X.shape[0] # number of instances
        acc_grad = np.zeros(sum(self.sizes), dtype=np.float64)
        weights = self.unpack_parameters(weights_flattened)

        # iterate through instances and accumulate deltas
        for i, x in enumerate(X):
            # calculate the activation values
            a, h_x = self.forward_prop(x, weights)

            # back propagate the error
            acc_grad += self.backward_prop(y[i], a, weights)
      
        return (1.0 / m) * acc_grad

    def cost(self, weights_flattened, X, y):

        m = X.shape[0]
        weights = self.unpack_parameters(weights_flattened)

        # compute the sum of the error
        cost_sum = 0
        for i, x in enumerate(X):
            a, h_x = self.forward_prop(x, weights)
            for k in range(len(h_x)):
                #cost_sum += (y[i]*np.log(h_x[0]) + (1- y[i])*np.log(1-h_x))
                cost_sum += (h_x[k] - y[i][k])**2

        return (1.0 / m * cost_sum)  

    def fit(self, X, y, epochs=1):

        for k in range(epochs):
            self.weights = lbfgs.fmin_LBFGS(self.cost, self.weights, self.jac, args=(X,y))
            #self.weights = opti.fmin_bfgs(self.cost, self.weights, fprime=self.jac, args=(X, y), gtol=1e-50, maxiter=200)

            #gradients = self.jac(self.weights, X, y)
            #self.weights -= 0.2 * gradients

    def predict(self, x):
        weights = self.unpack_parameters(self.weights)
        a, h_x = self.forward_prop(x, weights)
        return h_x


def trainXORProblem():
    print "This is a test of the neural net on the XOR problem\n"
    #nn = NeuralNetwork([2,2,1])
    #y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork([2,2,2])
    y = np.array([[1,0],[0,1],[0,1],[1,0]])
    
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    nn.fit(X, y)
    for i, e in enumerate(X):
        print(e,nn.predict(e))

def trainIrisDataset():
    feature_count = 4
    label_count = 3

    nn = NeuralNetwork([feature_count,10,label_count])
    data = np.loadtxt("iris.data",delimiter=",") # Labels must be floats
    np.random.shuffle(data)
    X = data[:,0:-1]
    flowers = data[:,-1]
    y = []
    for f in flowers:
        ys = [0,0,0]
        ys[int(f)] = 1
        y.append(ys)
    y = np.asarray(y)


    nn.fit(X, y)

    correct = 0
    for i, e in enumerate(X):
        #print(e,nn.predict(e))
        prediction = list(nn.predict(e))
        #print "Label: ",y[i]," | Predictions: ",prediction
        if prediction.index(max(prediction)) == flowers[i]:
            correct += 1
    print "Correct: ",correct,"/",i,"(",float(correct)/float(i),"%)"


if __name__ == '__main__':
    #trainXORProblem()
    trainIrisDataset()
    
    
    
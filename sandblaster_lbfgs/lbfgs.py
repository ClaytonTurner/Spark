import numpy as np
from scipy.optimize.optimize import vecnorm
from scipy.optimize.linesearch import line_search_BFGS as lineSearch

def f(x):   # The rosenbrock function
	return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
def fprime(x):
	return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))

def fmin_LBFGS(func, x0, fprime, maxIter=None):

	x0 = np.asarray(x0).flatten()
	if(x0.ndim == 0):
		x0.shape = (1,)

	x_k = x0
	old_fval = f(x0)
	old_old_fval = None
	gf_k = fprime(x0)
	invHessian_B = np.identity(len(x0))

	step_K = 0

	if(maxIter is None):
		maxIter = 1000

	maxHistory = 10
	history_S = np.empty(shape=(maxHistory,2))
	history_Y = np.empty(shape=(maxHistory,2))
	rho = np.empty(maxHistory)

	maxiter = 100
	norm = np.Inf
	gtol = 1e-5
	gnorm = vecnorm(gf_k, ord=norm)

	while (gnorm > gtol) and (step_K < maxiter):

		d_k = computeDirection(maxHistory, step_K, gf_k, invHessian_B, history_S, history_Y, rho)

		#lineSearch
		alpha_K = getAlphaLineSearch(x_k, d_k, gf_k, old_fval, old_old_fval)
		x_kp1 = np.add(x_k, alpha_K * d_k)

		#define the index to update the arrays S and Y
		#the history arrays, S and Y, are reused for eficiency
		indexHistory = step_K % maxHistory
		indexHistory = indexHistory if(indexHistory < (maxHistory - 1)) else 0

		#save new pair
		new_S = np.subtract(x_kp1, x_k)
		history_S[indexHistory] = new_S

		gf_kp1 = fprime(x_kp1) #gradient of f(x_kp1)
		new_Y = gf_kp1 - gf_k
		history_Y[indexHistory] = new_Y
		gf_k = gf_kp1
		
		rho[indexHistory] = 1.0 / (np.dot(new_S, new_Y))

		#updates the inverve Hessian matrix
		factor = np.dot(new_S, new_Y) / np.dot(new_Y, new_Y)
		invHessian_B = np.multiply(factor, invHessian_B)
		
		old_old_fval = old_fval
		old_fval = func(x_k)
		x_k = x_kp1
		gnorm = vecnorm(gf_k, ord=norm)
		step_K += 1
	
	x_opt = x_k
	fval = old_fval
	print("***Current function value: %f" % fval)
	print("***Iterations: %d" % step_K)
	return x_opt


def computeDirection(maxHistory, step_K, gf_k, B_k, history_S, history_Y, rho):
	
	upperBound = maxHistory if(step_K > maxHistory) else step_K
	alpha = np.empty(upperBound)

	for i in range(upperBound-1, -1, -1):
		alpha[i] = rho[i] * np.dot(history_S[i], gf_k)
		gf_k = np.subtract(gf_k, alpha[i] * history_Y[i])

	r = np.dot(B_k, gf_k)

	for i in range(0, upperBound):
		beta = rho[i] * np.dot(history_Y[i], r)
		r =  np.add(r, history_S[i] * (alpha[i] - beta))

	return -r

def getAlphaLineSearch(xK, direction_k, gf_k, old_fval, old_old_fval):
	""" returns alpha that satisfies x_new = x0 + alpha * pk"""
	res = lineSearch(f, xK, direction_k, gf_k, old_fval)
	(alpha_K), others = res[0], res[1:]
	return alpha_K


print fmin_LBFGS(f, [2.,2.], fprime)

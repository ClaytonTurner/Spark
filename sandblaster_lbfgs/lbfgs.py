import numpy as np
from scipy.optimize.linesearch import line_search_wolfe1

def fmin_LBFGS(func, x0, funcprime, maxIter=1000):
	"""
	func: callable f(x)
		Objective function to be minimized.
	x0: ndarray
		Initial guess.
	funcprime: callable f'(x)
		Gradient of f.
	maxIter: int, optional
		Maximum number of Iterations to perform
	"""
	x0 = np.asarray(x0).flatten()
	if(x0.ndim == 0):
		x0.shape = (1,)

	len_x = x0.size
	x_k = x0
	old_fval = func(x0)
	old_old_fval = None
	gf_k = funcprime(x0)
	invHessian_B = np.identity(len_x)

	maxHistory = 10
	history_S = np.empty(shape=(maxHistory, len_x))
	history_Y = np.empty(shape=(maxHistory, len_x))
	rho = np.empty(maxHistory)
	warnFlag = 0
	step_K = 0

	gtol = 1e-5
	gnorm = np.linalg.norm(gf_k, np.inf)

	while (gnorm > gtol) and (step_K < maxIter):

		#direction d_k = - B_k * g_k
		d_k = computeDirection(maxHistory, step_K, gf_k, invHessian_B, history_S, history_Y, rho)

		#lineSearch 
		alpha_K, fc, gc, old_fval, old_old_fval, gf_kp1 = \
							line_search_wolfe1(func, funcprime, x_k, d_k, gf_k, old_fval, old_old_fval)
		if(alpha_K is None):
			# Line search failed to find a better solution.
			warnFlag = 1
			break

		x_kp1 = x_k + alpha_K * d_k

		#define the index to update the arrays S and Y
		#the history arrays, S and Y, are reused for eficiency
		indexHistory = step_K % maxHistory
		indexHistory = indexHistory if(indexHistory < (maxHistory - 1)) else 0

		#save new pair
		history_S[indexHistory] = s_k = np.subtract(x_kp1, x_k)
		history_Y[indexHistory] = y_k = gf_kp1 - gf_k
		rho[indexHistory] = 1.0 / (np.dot(s_k, y_k))
		
		#updates the inverve Hessian matrix
		factor = np.dot(s_k, y_k) / np.dot(y_k, y_k)
		invHessian_B = np.multiply(factor, invHessian_B)

		gnorm = np.linalg.norm(gf_kp1, np.inf)
		x_k = x_kp1
		gf_k = gf_kp1

		if(not np.isfinite(old_fval)):
			#optimal value is +-Inf
			warnFlag = 1
			break

		step_K += 1
	
	if(warnFlag == 1):
		print "Stopped due to precission loss"

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


def f(x):   # The rosenbrock function
	return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
def fprime(x):
	return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))

def BoothFunc(x):
	return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def BoothFprime(x):
	return np.array((10*x[0] + 8*x[1] - 34, 10*x[1] + 8*x[0] - 38))

def MatyasFunc(x):
	return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

def MatyasFprime(x):
	return np.array((0.52*x[0] - 0.48*x[1], 0.52*x[1] + 0.48*x[0]))

print fmin_LBFGS(f, [2.,2.], fprime)

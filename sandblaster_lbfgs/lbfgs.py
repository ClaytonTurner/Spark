import numpy as np
from scipy.optimize.linesearch import line_search_wolfe1

def computeDirection(maxHistory, step_k, gf_k, history_S, history_Y, rho):
	"""
	returns d_k = B_k * g_k
	"""
	upperBound = min(step_k, maxHistory)
	alpha = np.empty(upperBound, dtype=np.float64)

	for i in range(upperBound-1, -1, -1):
		alpha[i] = rho[i] * np.dot(history_S[i], gf_k)
		gf_k = np.subtract(gf_k, alpha[i] * history_Y[i])

	if(step_k == 0):
		r = np.ones(len(gf_k)) * gf_k
	else:
		r = (np.dot(history_S[-1], history_Y[-1]) / np.dot(history_Y[-1], history_Y[-1])) * gf_k

	for i in range(0, upperBound):
		beta = rho[i] * np.dot(history_Y[i], r)
		r =  np.add(r, history_S[i] * (alpha[i] - beta))

	return -r

def fmin_LBFGS(func, x0, funcprime, args=(), maxIter=1000):
	"""
	func: callable f(x)
		Objective function to be minimized.
	x0: ndarray
		Initial guess.
	funcprime: callable f'(x)
		Gradient of f.
	maxIter: int, optional
		Maximum number of Iterations to perform

	return:
		x_opt: ndarray
	"""
	x0 = np.asarray(x0, dtype=np.float64).flatten()
	if(x0.ndim == 0):
		x0.shape = (1,)

	len_x = x0.size
	x_k = x0
	old_fval = func(x0, *args)
	old_old_fval = None
	gf_k = funcprime(x0, *args)

	maxHistory = 10

	history_S = []
	history_Y = []
	rho = []
	warnFlag = 0
	step_k = 0

	gtol = 1e-5
	gnorm = np.linalg.norm(gf_k, np.inf)
	lineSearchConverged = True
	while (not lineSearchConverged or gnorm > gtol) and (step_k < maxIter):
		lineSearchConverged = True

		#direction d_k = - B_k * g_k
		d_k = computeDirection(maxHistory, step_k, gf_k, history_S, history_Y, rho)

		#lineSearch 
		alpha_k, fc, gc, old_fval, old_old_fval, gf_kp1 = \
			line_search_wolfe1(func, funcprime, x_k, d_k, gf_k, old_fval, old_old_fval, args=args)

		if(alpha_k is None):
			# Line search failed to find a better solution.
			print 'Step:', step_k, "Line search did not converge. Set alpha_k for a small value"
			lineSearchConverged = False
			alpha_k = 0.01

		x_kp1 = x_k + alpha_k * d_k

		if(not lineSearchConverged):
			gf_kp1 = funcprime(x_kp1, *args)

		if(step_k > maxHistory):
			history_S.pop(0)
			history_Y.pop(0)
			rho.pop(0)

		#save new pair
		s_k = x_kp1 - x_k
		history_S.append(s_k)
		y_k = gf_kp1 - gf_k
		history_Y.append(y_k)
		try:
			dem = float(np.dot(s_k, y_k))
			rhok = 1.0 / dem
		except ZeroDivisionError:
			rhok = 1000.0
			print("Divide-by-zero encountered: rhok assumed large")
		if np.isinf(rhok):
			rhok = 1000.0
		rho.append(rhok)

		gnorm = np.linalg.norm(gf_kp1, np.inf)
		x_k = x_kp1
		gf_k = gf_kp1

		if(not np.isfinite(old_fval)):
			#optimal value is +-Inf
			warnFlag = 2
			break

		step_k += 1
	
	if(warnFlag == 2):
		print "Stopped due to precission loss"
	print "Steps:", step_k
	print "function value", old_fval
	return x_k


def f(x):
	return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2		
def fprime(x):		
	return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 
						2*(x[1] - x[0]**2)))
if __name__ == '__main__':
	print fmin_LBFGS(f, [2.,2.], fprime)

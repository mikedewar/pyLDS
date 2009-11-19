import pylab as pb
from numpy import matlib
import sys
import logging

logging.basicConfig(stream=sys.stdout,level=logging.INFO)
log = logging.getLogger('LDS')

class LDS:
	"""class defining a linear, gaussian, discrete-time Linear Dynamic System.
	
	Arguments
	----------
	A : matrix or list of matrix
		State transition matrix. Supply a list for non-stationary systems.
	B : matrix
		State transition matrix.
	C : matrix
		Observation matrix.
	Sw : matrix
		State covariance matrix.
	Sv : matrix
		Observation noise covariance matrix
	x0: matrix
		Initial state vector
	
	If you supply a list of A matrices, then the observations and inputs you
	use in each method will need to be the same length as the list of A matrices
	
	Attributes
	----------
	A : matrix or list of matrix
		State transition matrix.
	B : matrix
		State transition matrix.
	C : matrix
		Observation matrix.
	Sw : matrix
		State covariance matrix.
	Sv : matrix
		Observation noise covariance matrix
	nx: int
		number of states
	ny: int 
		number of outputs
	x0: matrix
		intial state
	X: list of matrix
		state sequence
	K: list of matrix 
		Kalman gain sequence
	M: list of matrix
		Cross covariance matrix sequence
	P: list of matrix
		Covariance matrix sequence
		
	Notes
	----------
	The Kalman Filter code is based on code originally written by Dr Sean
	Anderson.
	"""
	
	def __init__(self,A,B,C,Sw,Sv,x0):
		
		ny, nx = C.shape
		nu = B.shape[1]	
		
		if type(A) is list:
			for Ai in A:
				assert Ai.shape == (nx, nx)
		else:
			assert A.shape == (nx, nx)
		assert B.shape == (nx, nu)
		assert Sw.shape == (nx, nx)
		assert Sv.shape == (ny, ny)
		assert x0.shape == (nx,1)
		
		self.A = A
		self.B = B
		self.C = C
		self.Sw = Sw
		self.Sv = Sv
		self.ny = ny
		self.nx = nx
		self.nu = nu
		self.x0 = x0
		
		# initial condition
		# TODO - not sure about this as a prior covariance...
		self.P0 = 40000 * pb.matrix(pb.ones((self.nx,self.nx)))
		
		log.info('initialised state space model')
	
	def gen_A(self):
		"""
		generator function for the state transition matrix A. 
		
		Arguments
		----------
		A : matrix or list of matrix
			The state transition matrix.
			
		Notes
		----------
		This is written so as to incorporate a possibly time varying transition 
		matrix. If you supply a list of matrices for A, then this generator 
		yields the next item in that list. If you have a stationary system, 
		and hence only provide a single matrix for A, this generator will 
		always yield that matrix.
		"""
		if type(self.A) is list:
			for At in self.A:
				yield At
		else:
			while 1:
				yield self.A
				
	def gen_A_reverse(self):
		"""
		generator function for the state transition matrix A. 
		
		Arguments
		----------
		A : matrix or list of matrix
			The state transition matrix.
			
		Notes
		----------
		This method is essentially the same as LDS.gen_A() except that the A
		matrices are returned in the reverse order. Again, if the system is
		stationary, then this will always yield the stationary A matrix. This
		reversed generator is useful for a backwards pass.
		"""
		if type(self.A) is list:
			for At in reversed(self.A):
				yield At
		else:
			while 1:
				yield self.A
	
	def gaussian(self, mean, covariance):
		"""
		wraps pylab.multivariate_normal so it accepts a vector argument
		
		Arguments
		----------
		mean : matrix
			mean of a multivariate normal distribution written as a nx1 matrix
		covariance: matrix
			covariance matrix of a multivariate normal distribution
		"""
		return pb.multivariate_normal(mean.A.flatten(),covariance,[1]).T
	
	def transition_dist(self, x, u):
		mean = self.A_gen.next()*x + self.B*u
		return self.gaussian(mean,self.Sw)
	
	def observation_dist(self, x):
		mean = self.C*x
		return self.gaussian(mean,self.Sv)
		
	def gen(self,U):
		"""
		generator for the state space model
		
		Arguments
		----------
		U : list of matrix
			input sequence

		Yields
		----------
		x : matrix
			next state vector
		y : matrix
			next observation
			
		See Also:
		----------
		LDS.simulate() : returns a sequence of states and observations, rather
			than a generator
		"""	
		# intialise A matrix generator
		self.A_gen = self.gen_A()
		# intial state
		x = self.x0
		for u in U:
			y = self.observation_dist(x)
			yield x,y
			x = self.transition_dist(x, u)

	def simulate(self,U):
		"""
		simulates the state space model

		Arguments
		----------
		U : list of matrix
			input sequence

		Returns
		----------
		X : list of matrix
			array of state vectors
		Y : list of matrix
			array of observation vectors
		
		See Also:
		----------
		LDS.gen() : returns a generator for the linear dynamic system rather
			than a sequence of states and observations
		"""
		
		log.info('sampling from the state space model')
		
		X = []
		Y = []
		for (x,y) in self.gen(U):
			X.append(x)
			Y.append(y)
		return X,Y
	
	
	def kfilter(self, Y, U):
		"""Vanilla implementation of the Kalman Filter
		
		Arguments
		----------
		Y : list of matrix
			A list of observation vectors
		U : list of matrix
			A list of observation vectors
			
		The lengths of Y and U must be equal
			
		Returns
		----------	
		X : list of matrix
			A list of state estimates
		P : list of matrix
			A list of state covariance matrices
		K : list of matrix
			A list of Kalman gains
		XPred : list of matrix
			A list of uncorrected state predictions
		PPred : list of matrix
			A list of un-corrected state covariance matrices
		
		XPred is A*x[t-1] + B*u[t-1] and PPred[t] is A*P[t-1]*A + Sw. These
		are used in the RTS Smoother.
		
		See Also:
		----------
		LDS.rtssmooth() : an implementation of the RTS Smoother
		"""
		
		log.info('running the Kalman filter')
		
		# Predictor
		def Kpred(P, x, u):
			A = self.A_gen.next()
			x = A*x + self.B*u
			P = A*P*A.T + self.Sw
			return x,P

		# Corrector
		def Kupdate(P, x, y):
			K = (P*self.C.T) * (self.C*P*self.C.T + self.Sv).I
			x = x + K*(y-(self.C*x))
			P = (pb.eye(self.nx)-K*self.C)*P;
			return x, P, K

		## initialise
		self.A_gen = self.gen_A()
		xhat = self.x0
		nx = self.nx
		ny = self.ny
		# filter quantities
		xhatPredStore = []
		PPredStore = []
		xhatStore = []
		PStore = []
		KStore = []
		# initialise the filter
		xhat, P = Kpred(self.P0, xhat, U[0])
		## filter
		for y, u in zip(Y[:-1],U[1:]):
			# store
			xhatPredStore.append(xhat)
			PPredStore.append(P)
			# correct
			xhat, P, K = Kupdate(P, xhat, y)
			# store
			KStore.append(K)
			xhatStore.append(xhat) 
			PStore.append(P)
			# predict
			xhat, P = Kpred(P,xhat, u);
		
		return xhatStore, PStore, KStore, xhatPredStore, PPredStore
	
	def rtssmooth(self, Y, U):
		"""Vanilla implementation of the Rauch Tung Streibel(RTS) smoother
		
		Arguments
		----------
		Y : list of matrix
			A list of observation vectors
		U : list of matrix
			A list of observation vectors
		
		The lengths of Y and U must be equal
		
		Returns
		----------	
		X : list of matrix
			A list of state estimates
		P : list of matrix
			A list of state covariance matrices
		K : list of matrix
			A list of Kalman gains
		M : list of matrix
			A list of cross covariance matrices
			
			
		See Also:
		----------	
		LDS.kfilter() - an implementation of the Kalman Filter
		"""
		
		log.info('running the RTS Smoother')
		
		# run the Kalman filter
		xhatStore, PStore, KStore, xhatPredStore, PPredStore = self.kfilter(Y, U)
		# initialise the smoother
		A = self.gen_A_reverse()
		T = len(Y)-1
		xb = [None]*T
		Pb = [None]*T
		S = [None]*T
		xb[-1], Pb[-1] = xhatStore[-1], PStore[-1]
		# step through A a couple of times to prepare for the backwards pass
		for t in range(T,T-2,-1):
			A.next()
		## smooth
		for t in range(T-2,0,-1):
			S[t] = PStore[t] * A.next().T * PPredStore[t+1].I
			xb[t] = xhatStore[t] + S[t]*(xb[t+1] - xhatPredStore[t])
			Pb[t] = PStore[t] + S[t] * (Pb[t+1] - PPredStore[t+1]) * S[t].T
		# finalise
		xb[0] = xhatStore[0]
		Pb[0] = PStore[0]
		# iterate a final time to calucate the cross covariance matrices
		A = self.gen_A_reverse()
		# we need to step through the final A matrix
		A.next()
 		M = [None]*T
		M[-1]=(pb.eye(self.nx)-KStore[-1]*self.C) * A.next()*PStore[-2]
		for t in range(T-2,1,-1):
		    M[t]=PStore[t]*S[t-1].T + S[t]*(M[t+1] - A.next()*PStore[t])*S[t-1].T
		M[1] = matlib.eye(self.nx)
		M[0] = matlib.eye(self.nx)
		
		return xb, Pb, KStore, M


if __name__ == "__main__":
	import os
	os.system('py.test')
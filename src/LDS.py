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
	A : matrix
		State transition matrix.
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
	
	Attributes
	----------
	A : matrix
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
		nu = B.shape[0]
		
		assert A.shape == (nx, nx)
		assert B.shape == (nx, nu)
		assert Sw.shape == (nx, nx)
		assert Sv.shape == (ny, ny)
		assert x0.shape == (nx,1)
		
		self.A = pb.matrix(A)
		self.A = pb.matrix(B)
		self.C = pb.matrix(C)
		self.Sw = pb.matrix(Sw)
		self.Sv = pb.matrix(Sv)		
		self.ny = ny
		self.nx = nx
		self.nu = nu
		self.x0 = x0
		
		# initial condition
		# TODO - not sure about this as a prior...
		self.P0 = 40000* pb.matrix(pb.ones((self.nx,self.nx)))
		
		log.info('initialised state space model')
	
	def transition_dist(self, x, u):
		mean = self.A*x + self.B*x
		return pb.multivariate_normal(mean.A.flatten(),self.Sw,[1]).T
	
	def observation_dist(self, x):
		mean = self.C*x
		return pb.multivariate_normal(mean.A.flatten(),self.Sv,[1]).T
		
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
		x = self.x0
		for u in range(U):
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
		PPred : list of matrix
			A list of un-corrected state covariance matrices
		
		PPred[t] is A*P[t-1]*A + Sw. This is used in the RTS Smoother
		
		See Also:
		----------
		LDS.rtssmooth() : an implementation of the RTS Smoother
		"""
		
		log.info('running the Kalman filter')
		
		# Predictor
		def Kpred(P, x, u):
			x = self.A*x + self.B*u
			P = self.A*P*self.A.T + self.Sw
			return x,P

		# Corrector
		def Kupdate(P, x, y):
			K = (P*self.C.T) * (self.C*P*self.C.T + self.Sv).I
			x = x + K*(y-(self.C*x))
			P = (pb.eye(self.nx)-K*self.C)*P;
			return x, P, K

		## initialise	
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
		for y, u in zip(Y,U[1:]):
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
		
		return xhatStore, PStore, KStore, PPredStore
	
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
		xhatStore, PStore, KStore, PPredStore = self.kfilter(Y, U)
		# initialise the smoother
		xb = [None]*T
		Pb = [None]*T
		S = [None]*T
		xb[-1], Pb[-1] = xhatStore[-1], PStore[-1]
		## smooth
		for t in range(T-2,0,-1):
			S[t] = PStore[t]*self.A.T * PPredStore[t+1].I
			xb[t] = xhatStore[t] + S[t]*(xb[t+1] - xhatPredStore[t])
			Pb[t] = PStore[t] + S[t] * (Pb[t+1] - PPredStore[t+1]) * S[t].T
		# finalise
		xb[0] = xhatStore[0]
		Pb[0] = PStore[0]
		# iterate a final time to calucate the cross covariance matrices
 		M = [None]*T
		M[-1]=(pb.eye(nx)-KStore[-1]*self.C) * self.A*PStore[-2]
		for t in range(T-2,1,-1):
		    M[t]=PStore[t]*S[t-1].T + S[t]*(M[t+1] - self.A*PStore[t])*S[t-1].T
		M[1] = matlib.eye(self.nx)
		M[0] = matlib.eye(self.nx)
		
		return xb, Pb, KStore, M


if __name__ == "__main__":
	import os
	os.system('py.test')
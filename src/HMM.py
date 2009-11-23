import numpy
import os
import logging
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stdout,level=logging.INFO)
log = logging.getLogger('HMM')

class HMM():
	"""
	Class defining a hidden markov model
	"""
	def __init__(self,Z,b,pi):
		"""
		arguments:
		Z: state transition matrix
		b: emmision matrix
		pi: initial distribution
		"""
		log.debug('creating HMM instance')
		# assertions
		assert round(sum(pi),5) == 1
		assert type(Z) is numpy.ndarray, type(Z)
		assert type(b) is numpy.ndarray, type(b)
		assert type(pi) is numpy.ndarray, type(pi)
		assert Z.shape[0] == Z.shape[1]
		assert Z.shape[0] == b.shape[0]
		assert Z.shape[0] == pi.shape[0]
		# assignment
		self.Z = Z
		self.b = b
		self.pi = pi
		self.M = len(pi)
	
	def gen(self,T,state_dict=None,obs_dict=None):
		"""
		HMM generator. Yields the next state.
		
		arguments:
		T: number of samples to be generated
		
		yields (s,o):
		s: next state symbol
		o: next observation symbol
		
		"""
		log.debug('initialised HMM generator')
		# make a quick function that draws from a multinomial defined by the
		# sth row of an array
		draw = lambda s,A: numpy.where(
			numpy.random.multinomial(1, A[s])
		)[0][0]	
		# draw the first state from self.pi
		s = numpy.where(numpy.random.multinomial(1, self.pi))[0][0]
		log.debug('initial state: %s'%s)
		# draw the first observation
		o = draw(s,self.b)
		log.debug('initial observation: %s'%o)
		if state_dict:
			log.info('drawing initial state from pi\n')
			yield state_dict[s]
			log.info('drawing first observation\n')
			yield obs_dict[o]
		else:
			yield s, o
		for t in range(T-1):
			
			# draw the next state based on the current state
			if state_dict:
				log.info('drawing next STATE\n')
			s = draw(s, self.Z)
			if state_dict:
				yield state_dict[s]
			
			# draw the next observation based on the next state
			if obs_dict:
				log.info('drawing next OBSERVATION\n')
			o = draw(s, self.b)
			if obs_dict:
				yield obs_dict[o]
			if not state_dict:
				yield s, o
			
	def forward_backward(self,S,scale_flag=True):
		"""
		Forwards-Backwards algorithm. Returns the alpha and beta variables as
		a list of arrays.
		
		arguments:
		S: observed symbol sequence
		
		returns:
		alpha: where alpha[t][i]=P(O_1...O_t,q_t=i|model)
		beta: where beta[t][i]=P(O_t+1...O_T|q_t=i,model)
		"""
		log.info('running forward-backward algorithm')
		# initialise alpha
		alpha = [numpy.empty(self.M) for s in S]
		alpha[0] = self.pi[:] * self.b[:,S[0]]
		if scale_flag:
			# initialise the scale array
			scale = numpy.empty(len(S))
			scale[0] = sum(alpha[0])
			# scale the initial alpha
			alpha[0] /= scale[0]
		# recursion
		for t,s in enumerate(S[1:]):
			for j in range(self.M):
				# calculate alpha
				alpha[t+1][j] = sum(alpha[t] * self.Z[:,j])*self.b[j,s]
			if (alpha[t+1]==0.0).all():
				raise ValueError
			if scale_flag:
				# calculate the scale
				scale[t+1] = sum(alpha[t+1])
				# calculate the scaled alpha
				alpha[t+1] /= scale[t+1]
		# initialise beta
		beta = [numpy.empty(self.M) for s in S]
		for i in range(self.M):
			beta[-1][i] = 1
		# recursion
		for t in range(len(S)-1,0,-1):
			for i in range(self.M):
				# calculate beta
				beta[t-1][i] = sum(self.Z[i,:] * self.b[:,S[t]] * beta[t][:])
				if numpy.isnan(beta[t-1][i]):
					raise ValueError
			if scale_flag:
				# scale beta
				beta[t-1] /= scale[t-1]		
		return alpha, beta
		
	def viterbi(self,S):
		# intialise variables
		log.info('running viterbi algorithm')
		psi = [numpy.empty(self.M) for s in S]
		phi = [numpy.empty(self.M) for s in S]
		q = [0 for s in S]
		#initialisation
		psi[0] = numpy.zeros(self.M)
		phi[0] = numpy.log(self.pi[:]) + numpy.log(self.b[:,S[0]])
		# recursion
		for t,s in enumerate(S[1:]):
			for j in range(self.M):
				logsum = phi[t]+numpy.log(self.Z[:,j])
				phi[t+1][j] = max(
					phi[t]+numpy.log(self.Z[:,j])
				) + numpy.log(self.b[j,s])
				log.debug("phi[%s] + log(Z[:,%s]) = %s"%(t,j,logsum))
				psi[t+1][j] = numpy.argmax(logsum)
		# termination
		q[-1] = numpy.argmax(phi[-1])
		# backtracking
		for t in range(len(S)-2,-1,-1):
			q[t] = psi[t+1][q[t+1]]
		return numpy.array(q)
		
	def draw_chain(self, states, obs, state_dict, obs_dict, 
		state_colors=None, obs_colors=None):
		
		if state_colors == None:
			state_colors = {}
			for key in state_dict:
				state_colors[key] = 'none'

		if obs_colors == None:
			obs_colors = {}
			for key in obs_dict:
				obs_colors[key] = 'none'
		
		T = len(states)
		
		fig1 = plt.figure()
		ax = fig1.add_axes([0, 0, 1, 1], 
			frameon=False, 
			aspect=1.,
			xlim=[0,T],
			ylim=[0,2]
		)
		
		for i,(s,o) in enumerate(zip(states,obs)):
			state = mpatches.Circle(
				(i+0.5, 1), 
				0.2, 
				fc=state_colors[s]
			)
			obs = mpatches.Circle(
				(i+0.5, 0.25), 
				0.2, 
				fc=obs_colors[o]
			)
			hidden_arr = mpatches.FancyArrow(
				i+0.75, 
				1, 
				0.5, 
				0,
				length_includes_head=True,
				head_width=0.1,
				fc = 'black'
			)
			obs_arr = mpatches.FancyArrow(
				i+0.5, 
				1-0.25, 
				0, 
				-0.25,
				length_includes_head=True,
				head_width=0.1,
				fc = 'black'
			)
			ax.add_patch(state)
			ax.add_patch(obs)
			ax.add_patch(obs_arr)
			if i<T-1:
				ax.add_patch(hidden_arr)
			
			"""
			fontsize = 0.2 * 70

					arrowstyle='-|>',
					shrinkA=5,
					shrinkB=5,
					fc="w", ec="k",
					connectionstyle="arc3,rad=-0.05",
				),
				bbox=dict(boxstyle="square", fc="w")
			)"""

		
		
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		plt.draw()
		plt.show()
		
	
	

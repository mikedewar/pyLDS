import numpy
from hmm import HMM

class TestHMM():
	
	def __init__(self):
		self.Z = numpy.array([
			[0.8,  0.09, 0.01],
			[0.09, 0.8,  0.01],
			[0.1,  0,    0.8]
		])
		self.b = numpy.array([
			[0.1, 0.1, 0.8],
			[0.05, 0.9, 0.05],
			[0.8, 0.1, 0.1]
		])
		self.pi = numpy.array([0.9,0.05,0.05])
		self.T = 2000
		# we want the errors to be less than 20%
		self.error_threshold = 0.2
	
	def setup(self):	
		self.model = HMM(self.Z,self.b,self.pi)
	
	def gen_states_obs(self):
		states = []
		obsvns = []
		for (s,o) in self.model.gen(self.T):
			states.append(s)
			obsvns.append(o)
		return states, obsvns
	
	def test_init(self):	
		self.model = HMM(self.Z,self.b,self.pi)
		
	def test_gen(self):
		self.setup()
		states = []
		obsvns = []
		for (s,o) in self.model.gen(10):
			states.append(s)
			obsvns.append(o)
		assert len(states) == 10
		assert len(obsvns) == 10
	
	def test_forward_backward(self):
		self.setup()
		states, obsvns = self.gen_states_obs()
		alpha,beta = self.model.forward_backward(obsvns)	
		
		gamma = [a*b/sum(a*b) for a,b in zip(alpha,beta)]
		state_est = numpy.array([numpy.where(g==max(g))[0][0] for g in gamma])
		err = sum(state_est != numpy.array(states))/float(len(states))
		assert err < self.error_threshold
	
	def test_viterbi(self):
		self.setup()
		states, obsvns = self.gen_states_obs()
		state_est = self.model.viterbi(obsvns)		
		err = sum(state_est != numpy.array(states))/float(len(states))
		assert err < self.error_threshold

if __name__ == "__main__":
	import os
	os.system('py.test -s')
import numpy as np
import pdb
from utils import normalize

class Environment:
	"""
	SINR foromulas
	(Environment works with numpy objects !)
	
	"""
	def __init__(self,data, nb_AP=10, nb_Users=5, transmission_power=5.01, seed=0):
		self.M = nb_AP                  # number of Access Points
		self.K = nb_Users				# number of Users
		self.Pd = transmission_power
		self.state_dim = self.K
		self.action_dim = self.M * self.K
		self.observation_shape = (self.state_dim,)
		self.action_space = (self.action_dim, )
		# self.X0 = np.array(data['X0'])
		self.Gammaa = np.array(data['Gammaa'])
		self.BETAA = np.array(data['BETAA'])
		self.Phii_cf = np.array(data['Phii_cf'])
		np.random.seed(seed)
		ot=np.random.uniform(high=1,size=(self.M, self.K)).astype('float32')
		ot = normalize(ot)
		self.ot = ot
		
		
		
	
	def sinr(self):
		"""
		Calculates the sinr (state)
		
		"""
#		pdb.set_trace()

		SINR = np.zeros(self.K, dtype='float32')
		R = np.zeros(self.K, dtype='float32')
		PC = np.zeros((self.K, self.K), dtype='float32')
		Othernoise = np.zeros((self.K, self.K), dtype='float32')
		for ii in range(self.K):
			for k in range(self.K):
				PC[ii, k] = sum((np.sqrt(self.ot[:, ii]) * np.sqrt(self.Gammaa[:, ii]) / self.BETAA[:, ii] * self.BETAA[:, k])) * np.dot(
					self.Phii_cf[:, ii], self.Phii_cf[:, k])
				Othernoise[ii, k] = sum((self.ot[:, ii] * self.BETAA[:, k]))
		PC1 = PC * PC
		Othernoise1 = abs(Othernoise)

		for k in range(self.K):
			num = 0
			for m in range(self.M):
				num = num + np.sqrt(self.ot[m, k]) * np.sqrt(self.Gammaa[m, k])
			SINR[k] = self.Pd * num ** 2 / (1 + self.Pd * sum(Othernoise1[:, k]) + self.Pd * sum(PC1[:, k]) - self.Pd * PC1[k, k])
			R[k] = SINR[k]


		return R.astype(np.float32)
	
	def reset(self):				
		ot = np.random.uniform(low=0,high=1,size=(self.M, self.K)).astype('float32')
		self.ot = ot

		return self.sinr()


	def step(self, action_t):
		# action_t is of dimension K*1
		self.ot = action_t.reshape(self.M, self.K)


		state_t_pls_1 = self.sinr()
		# rwd_t = (np.log2(1 + np.min(state_t_pls_1))-k).astype(np.float32)
		rwd_t = (np.log2(1 + np.min(state_t_pls_1))).astype(np.float32)
		done_t = 0.0
		return state_t_pls_1, rwd_t, np.float32(done_t), self.ot.reshape(1,-1)

if __name__ == "__main__":
	import scipy.io as sio
	import numpy as np
	data0 = sio.loadmat('datag.mat')
	nb_AP = int(data0['M'])
	nb_Users = int(data0['K'])
	Pd = np.array(data0['Pd'])
	obj = Environment(data=data0,nb_AP= nb_AP, nb_Users=nb_Users, transmission_power=Pd[0], seed=0)
	print("test sinr() {}".format(np.log2(1 + obj.sinr())))
	print((data0['a3']))

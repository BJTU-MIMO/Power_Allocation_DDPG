import numpy as np
import tensorflow as tf
import pdb
import time
import matplotlib.pyplot as plt
import scipy.io

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, Flatten
from tensorflow.keras.models import Model


class Actor():
	""" 
	Policy(State) = action

	NEURAL NETWORK MODEL:
	Input -> hidden-layer with 256 units (relu) -> hidden-layer with 128 units (sigmoid) -> output (sigmoid)
	
	Sigmoid is chosen because of the constraint:  0<=Wij<=1
	"""
	
	def __init__(self, state_dim, action_dim, lr, action_bound_range):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound_range = action_bound_range
		self.lr = lr
		self.model = self.build_model()
		self.target = self.build_model()
		self.adam_optimizer = Adam(learning_rate=lr)
	
	def build_model(self):
		state = Input(shape=self.state_dim)
		x = Dense(400, activation='relu')(state)
		x = Dense(300, activation='relu')(x)
		out = Dense(self.action_dim, activation='sigmoid')(x)


		return Model(inputs=state, outputs=out)




if __name__ == '__main__':
	# inference calcul


 """
	APs = np.linspace(15,150,20).astype(np.int)
	res = []
	for param in APs:
		print("AP: ",param)
		with tf.device('/cpu:0'):
			actor = Actor(state_dim=int(param/3), action_dim=int(param) * int(param/3), lr=0.01, action_bound_range=1)
			state = np.random.uniform(high=100,size=(1,int(param/3))).astype('float32')
			s = time.time()
			action = actor.model(state)
		res.append(time.time()-s)


	plt.plot(APs, res)
	plt.show()
	res = np.asarray(res, dtype=np.float)
	obj = np.concatenate((np.expand_dims(APs, axis=1), np.expand_dims(res, axis=1)), axis=1)
	scipy.io.savemat('inference.mat', mdict={'inference_drl':obj})
 """
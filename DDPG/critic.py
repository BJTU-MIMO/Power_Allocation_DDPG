import numpy as np

from tensorflow.keras.layers import Input, Dense, concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K


class Critic:
	"""
	Q(state, action) -> expected_reward
	NEURAL NETWORK MODEL:
	Input_state------> hidden-layer with 256 units (relu) ---> concat with Input_action ----> hidden-layer 
			with 256 units (relu) -----> hidden-layer with 128 units (relu) ----> output (linear)
	
	"""

	def __init__(self, state_dim, action_dim, lr):
		K.set_floatx('float32')

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.model = self.build_model()
		self.target = self.build_model()
		self.adam_optimizer = Adam(learning_rate=lr)
		
#		self.model.compile(Adam(lr), 'mse')
#		self.target.compile(Adam(lr), 'mse')

	
	def build_model(self):
		state = Input(shape=(self.state_dim,))
		action = Input(shape=(self.action_dim,))

		x = concatenate([state, action], axis=1)

		x = Dense(400, activation='relu')(x)
		x = Dense(300, activation='relu')(x)
		out = Dense(1, activation='linear')(x)

		return Model(inputs=[state, action], outputs=out)



if __name__ == '__main__':
	 test = Critic(2, 5, 5, 0.001)
	 #May be the code append a extra term
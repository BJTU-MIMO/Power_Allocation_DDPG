import random
import pdb
import numpy as np
from collections import deque
import gym


class Replay_Buffer:
	"""
	Buffer for Training the Critic and Actor Models
	It is a deque
	Each element in the deque is a list of numpy vectors [state, action, reward, next_state, done]
	"""
	def __init__(self, max_buffer_size, batch_size, dflt_dtype='float32'):
		random.seed(0)
		self.batch_size = batch_size
		self.buffer = deque(maxlen=max_buffer_size)
		self.dflt_dtype = dflt_dtype
		self.count = 0
	
	def add_experience(self, state, action, reward, next_state, done):
		self.count += 1
		self.buffer.append([state, action, reward, next_state, done])
	
	def sample_batch(self, state_dim, action_dim):
		if (self.count < self.batch_size):
			replay_buffer = np.array(random.sample(self.buffer, self.count))
		else:
			replay_buffer = np.array(random.sample(self.buffer, self.batch_size))
			
		states_batch = np.empty(shape=(0, state_dim), dtype='float32')
		actions_batch = np.empty(shape=(0, action_dim), dtype='float32')
		rewards_batch = np.empty(shape=(0, 1), dtype='float32')
		next_states_batch = np.empty(shape=(0, state_dim), dtype='float32')
		done_batch = np.empty(shape=(0, 1), dtype='float32')


		for x in replay_buffer:
#            pdb.set_trace()
			states_batch = np.vstack((states_batch, x[0]))
			actions_batch = np.vstack((actions_batch, x[1]))
			rewards_batch = np.vstack((rewards_batch, x[2]))
			next_states_batch = np.vstack((next_states_batch, x[3]))
			done_batch = np.vstack((done_batch, x[4]))
		
		return states_batch, actions_batch, rewards_batch, next_states_batch, done_batch


if __name__=="__main__":
	buffer = get_replay_buffer(int(1e6), 0.99, gym.make("Pendulum-v0"), True, True, True)
    #Maybe the author forget it
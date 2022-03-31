import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import pdb
import gym
import cProfile
import pstats
import random
import time
import scipy.io as sio
from collections import deque
from tensorflow.keras.losses import MeanSquaredError
from actor import Actor
from critic import Critic
from buffer import Replay_Buffer
from utils import huber_loss, update_target_variables, normalize

if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
#        tf.config.experimental.set_memory_growth(cur_device, enable=True)



class DDPG:
	def __init__(self, env,
				actor_weights = None,
				critic_weights = None,
				max_action=1,
				max_buffer_size = 1000000,
				batch_size = 200,
				tow = 0.005,
				discount_factor = 0.99,
				actor_learning_rate = 0.001,
				critic_learning_rate = 0.001,
				dtype='float32',
				timestamp = 100000,
				max_epsiode_steps = 1000,
				n_warmup = 200,
				sigma=0.1):


		self.max_epsiode_steps = max_epsiode_steps
		self.timestamp = timestamp

		self.n_warmup = n_warmup

		# 80 of time is for exploration
		self.dflt_dtype = dtype
		
		self.max_buffer_size = max_buffer_size
		self.batch_size = batch_size

		self.sigma = sigma
		self.tow = tow		
		self.gamma = discount_factor
		self.max_action = max_action
		np.random.seed(0)
		random.seed(0)
		
		self.env = env

		if isinstance(self.env, gym.Env):
			self.state_dim = env.observation_space.shape[0] if env.observation_space.shape != tuple() else 1
			self.action_dim = env.action_space.shape[0] if env.action_space.shape != tuple() else 1
		else:
			self.state_dim = env.state_dim
			self.action_dim = env.action_dim

		self.actor = Actor(self.state_dim, self.action_dim, actor_learning_rate, max_action)

		self.critic = Critic(self.state_dim, self.action_dim, critic_learning_rate)

#		if actor_weights:
#			update_target_variables(
#				self.actor.model.weights, actor_weights, tau=1.0)

#		if critic_weights:
#			update_target_variables(
#				self.critic.model.weights, critic_weights, tau=1.0)	

		update_target_variables(
			self.actor.target.weights, self.actor.model.weights, tau=1.0)

		update_target_variables(
			self.critic.target.weights, self.critic.model.weights, tau=1.0)

		
		# Setting the buffer
		self.buffer = Replay_Buffer(self.max_buffer_size, self.batch_size, self.dflt_dtype)

		self.device = '/gpu:1'

		self.return1 = np.zeros(timestamp)
		
	def get_action(self, state):
		"""
		Predicting the action with the actor model from the state
		C2: ||Wk||<=1 so we always divide by its norm so ||Wk||==1
		"""
		is_single_state = len(state.shape) == 1

		state = np.expand_dims(state, axis=0).astype(
			np.float32) if is_single_state else state

		action = self._get_action_body(
			tf.constant(state), self.sigma)
		return action.numpy()[0] if is_single_state else action.numpy()


	@tf.function
	def _get_action_body(self, state, sigma):
		with tf.device(self.device):
			action = self.actor.model(state)
			# action += tf.random.normal(shape=action.shape,
			# 	mean=0, stddev=sigma, dtype=tf.float32)
			
			return tf.clip_by_value(action, 0, self.max_action)


	def train_step(self, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch):
		"""
		Performing the update of the models (Actor and Critic) using the gradients on batches with size
		by default == 64
		"""
		actor_loss, critic_loss, td_error = self._train_step_body(
			states_batch, actions_batch, next_states_batch, rewards_batch, done_batch)


		if actor_loss is not None:
			tf.summary.scalar(name='Training/actor_loss', data=actor_loss)

		tf.summary.scalar(name='Training/critic_loss', data=critic_loss)

		return td_error

	@tf.function
	def _train_step_body(self, states, actions, next_states, rewards, dones):
		with tf.device(self.device):
			with tf.GradientTape() as tape:
				td_errors = self._compute_td_error_body(
					states, actions, next_states, rewards, dones)
				critic_loss = tf.reduce_mean(
					tf.square(td_errors))

			critic_grad = tape.gradient(
				critic_loss, self.critic.model.trainable_weights)
			self.critic.adam_optimizer.apply_gradients(
				zip(critic_grad, self.critic.model.trainable_weights))

			with tf.GradientTape() as tape:
				next_action = self.actor.model(states)
				actor_loss = -tf.reduce_mean(
					self.critic.model([states, next_action]))

			actor_grad = tape.gradient(
				actor_loss, self.actor.model.trainable_variables)
			self.actor.adam_optimizer.apply_gradients(
				zip(actor_grad, self.actor.model.trainable_variables))
			
			update_target_variables(
				self.actor.target.weights, self.actor.model.weights, tau=self.tow)

			update_target_variables(
				self.critic.target.weights, self.critic.model.weights, tau=self.tow)

			return actor_loss, critic_loss, td_errors

	@tf.function
	def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
		with tf.device(self.device):
			not_dones = 1. - dones
			target_Q = self.critic.target(
				[next_states, self.actor.target(next_states)])
			target_Q = rewards + (not_dones * self.gamma * target_Q)
			target_Q = tf.stop_gradient(target_Q)
			current_Q = self.critic.model([states, actions])
			td_errors = target_Q - current_Q
		return td_errors


	def train(self):

		total_steps = 0
		# tf.summary.experimental.set_step(total_steps)
		episode_steps = 0
		episode_return = 0
		episode_start_time = time.perf_counter()
		n_episode = 0

# 		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# 		train_log_dir = 'logs/' + current_time
#		train_log_dir = 'logs/gradient_tape/' + "AP="+str(self.env.M)+";Us="+str(self.env.K)+";Pr="+str(self.env.transmission_power)
# 		writer = tf.summary.create_file_writer(train_log_dir)
# 		writer.set_as_default()

		# tf.summary.experimental.set_step(total_steps)


		old_state = self.env.reset()
		max_reward_episode = 0
		start_state = old_state

		last_start = old_state

		fps = 0
		while total_steps < self.timestamp:
			if total_steps < self.n_warmup:
				action = np.random.uniform(low=0, high=1, size=self.action_dim)

			else:
				action = self.get_action(old_state)
			
#			action = normalize(action.reshape(self.env.M, self.env.K)).reshape(self.action_dim)
			new_state, reward, done, _ = self.env.step(action)
			episode_steps += 1
			episode_return += reward
			total_steps += 1
			# tf.summary.experimental.set_step(total_steps)

			self.buffer.add_experience(old_state, action, reward, new_state, done)

			done_flag = done

			old_state = new_state

			# tf.summary.scalar(name="Episode/Reward", data=reward)
			

			if done or episode_steps == self.max_epsiode_steps:
				fps = (time.perf_counter() - episode_start_time)/episode_steps
				# print("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
	            #         n_episode, total_steps, episode_steps, episode_return+k*1000, fps))
				print("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
					n_episode, total_steps, episode_steps, episode_return, fps))
				# self.return1[n_episode] = episode_return+k*1000
				self.return1[n_episode] = episode_return

				if episode_return > max_reward_episode:
					print("Start From Old")
					max_reward_episode = episode_return
					start_state = old_state
				else:
					old_state = last_start

				last_start = old_state

#				old_state = self.env.reset()
				episode_start_time = time.perf_counter()

				n_episode+=1

				episode_steps = 0
				episode_return = 0

				

			if total_steps < self.n_warmup:
				continue

			states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = self.buffer.sample_batch(self.state_dim, self.action_dim)
			
			self.train_step(states_batch, actions_batch, rewards_batch, next_states_batch, done_batch)


		state = self.env.reset()
		state = np.expand_dims(state, axis=0).astype(np.float32)
		step = 0
		max_reward = 0
		s = time.time()
		while step < 1000:
			step+=1
			action = self.actor.model(state)
			state, reward, _, _ = self.env.step(action.numpy())
			if reward>max_reward:
				max_reward = reward
				fps = time.time() - s 
			state = np.expand_dims(state, axis=0).astype(np.float32)
			# print("Steps: {1: 7} Return: {3: 5.4f} time to converge: {4: 3.7f}".format(
	        #             n_episode, step, episode_steps, reward+k*1000, fps))
			print("Steps: {1: 7} Return: {3: 5.4f} time to converge: {4: 3.7f}".format(
				n_episode, step, episode_steps, reward, fps))
			print(np.log2(1 + state).astype(np.float32))
			return1=self.return1/1000
			sio.savemat('return1.mat', {'array': return1})


		return self.actor.model.weights, self.critic.model.weights



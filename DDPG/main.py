import pdb
import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from environment import Environment
from ddpg import DDPG
import gym
import torch
import safety_gym


if __name__=="__main__":
	SINRall = np.empty(shape=0)
	for n in range(130000):
		data0 = sio.loadmat('data/datag' + str(n + 1) + '.mat')
		nb_AP = int(data0['M'])
		nb_Users = int(data0['K'])
		Pd = np.array(data0['Pd'])
		p = np.float32(10 ** 0.7)
		epidsode_step = 1000
		epidsode_number = 2000
		timestamp = epidsode_step * epidsode_number
		#	env = gym.make("Pendulum-v0")
		env = Environment(data=data0, nb_AP=nb_AP, nb_Users=nb_Users, transmission_power=Pd, seed=0)
		big_boss = DDPG(env, timestamp=timestamp, actor_weights=None, critic_weights=None)
		# actor_weights, critic_weights = big_boss.train()
		SINR = big_boss.train()
		SINRall = np.append(SINRall,SINR)
		sio.savemat('SINRall.mat', {'array': SINRall})




"""	for agentId in range(1,2):
		env = Environment(agentId = agentId, nb_AP = nb_AP, nb_Users = nb_Users)
		agent = DDPG(env, discount_factor=0)
		agent.train()
"""
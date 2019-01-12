import sys, math
import gym
import random
import numpy as np
import pandas as pd
import Box2D
import gym

from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from gym import spaces
from gym.utils import seeding
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

episodes = 3000
input_dim = 8
output_dim = 4
lr = 0.00025
memory_size = 100000
epsilon = 1
epsilon_decay = 0.99995
epsilon_min = 0.05
batch_size = 64
gamma = 0.99

memory = deque(maxlen=memory_size)

model_one = Sequential()
model_one.add(Dense(64, activation='relu', input_dim=input_dim))
model_one.add(Dense(64, activation='relu'))
model_one.add(Dense(output_dim, activation='linear'))
model_one.compile(loss='mse', optimizer=Adam(lr=lr))

model_two = Sequential()
model_two.add(Dense(64, activation='relu', input_dim=input_dim))
model_two.add(Dense(64, activation='relu'))
model_two.add(Dense(output_dim, activation='linear'))
model_two.compile(loss='mse', optimizer=Adam(lr=lr))

env = gym.make('LunarLander-v2')
recent_scores = deque(maxlen=100)
all_scores = []


def pick_action(state):
	state = np.array([state])
	if np.random.rand() < epsilon:
		action = random.randint(0, 3)
	else:
		q_values = model_one.predict(state)
		action = np.argmax(q_values)
	return action


def train():
	batch = np.array(random.sample(memory, batch_size))
	states, actions, rewards, next_states, dones = np.split(batch, 5, axis=1)
	
	states = np.array(states.flatten().flatten().tolist())
	actions = np.array(actions.flatten().flatten().tolist())
	rewards = rewards.flatten()
	next_states = np.array(next_states.flatten().flatten().tolist())
	dones = dones.flatten()

	state_qs = model_one.predict(states)
	next_state_qs = model_two.predict(next_states)

	inverse_actions = np.logical_not(actions)
	a = state_qs * actions
	next_state_intrinsic_value = np.amax(next_state_qs, axis=1).reshape(batch_size,1)
	dones = np.logical_not(dones).reshape(batch_size, 1)
	next_state_intrinsic_value = gamma*next_state_intrinsic_value * dones
	rewards = rewards.reshape(batch_size,1)
	next_state_value = next_state_intrinsic_value + rewards

	b = actions * next_state_value
	c = state_qs * inverse_actions
	state_qs = b+c
	
	model_one.fit(states, state_qs, batch_size=batch_size, epochs=1, verbose=0)

recent_scores = deque(maxlen=100)
all_scores = []

for e in range(episodes):
	done = False
	score = 0
	state = env.reset()
	while done is False:
		epsilon = epsilon*epsilon_decay
		if epsilon < epsilon_min:
			epsilon = epsilon_min

		action = pick_action(state)
		
		next_state, reward, done, info = env.step(action)

		score +=reward

		if action == 0:
			action = np.array([1,0,0,0])
		elif action == 1:
		 	action = np.array([0,1,0,0])
		elif action == 2:
		 	action = np.array([0,0,1,0])
		else:
		 	action = np.array([0,0,0,1])

		memory.append([state, action, reward, next_state, done])

		if len(memory) > batch_size:
			train()

		state = next_state

	model_two.set_weights(model_one.get_weights())			
	recent_scores.append(score)
	all_scores.append([score, epsilon, len(memory)])


	if np.mean(recent_scores) >= 200:
		model_one.save('model_success.h5')
		model_one.save_weights('weights_success.h5')
		break
	if e%100 == 0:
		print('episode:', e, ' current epsilon:',epsilon, ' current rolling score', np.mean(recent_scores))


all_scores_df = pd.DataFrame(all_scores)
all_scores_df.to_csv('scores_success.csv')



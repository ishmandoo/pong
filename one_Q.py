import gym
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
import numpy as np
import random


env = gym.make("Pong-v0")
print env.unwrapped.get_action_meanings()
random.seed(9)
def makeNetwork(layers=2):
	model = Sequential()
	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Conv2D(10, (3, 3), activation='relu', input_shape=(210, 160, 1)))
	#model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	#model.add(Conv2D(10, (3, 3), activation='relu'))
	#model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(10, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(5))

	sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mse', optimizer=sgd)
	return model


modelUp = makeNetwork()
modelDown = makeNetwork()

def toGrayscale(frame):
	return np.mean(frame, axis=2, keepdims=True)

def shredder(N, show=False, random_play=False):

	observations = []
	actions = []
	rewards = []


	for i in range(N):
		print "step %d"%i
		done = False

		obs = toGrayscale(env.reset())
		last_obs = obs
		reward = 0
		while not done:
			if random_play:
				action = random.choice([2,3])
			else:
				qs = model.predict(obs.reshape(1,210,160,1))[0]
				action = np.argmax(qs)
			observations.append(obs)
			actions.append(action)
			if show:
				env.render()
			obs,reward,done,info = env.step(int(action))
			obs = toGrayscale(obs) - last_obs
			rewards.append(reward)
			last_obs = obs

	next_observations = observations[1:]
	observations = observations[:-1]
	actions = actions[:-1]
	rewards = rewards[:-1]
	return actions, observations,next_observations, rewards

def shredder2(actions,observations,next_observations,rewards):
	gamma = .9

	print("labels")
	qs = model.predict(next_observations)
	maxQ = np.max(qs, axis=1)
	labels = rewards+gamma*maxQ

	print("fitting")
	model.fit(observations,labels,epochs=1)

def shredderWon(N, random_play=False):
	a, o, no, r = shredder(N, show=False, random_play=True)
	shredder2(a, o, no, r)
	shredder(1, show=True)

shredderWon(1)
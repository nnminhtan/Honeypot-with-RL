import numpy as np
import random
import tensorflow as tf
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Input

class DDQNAgent:
    def __init__(self, state_shape, action_shape, config):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.memory = []  # Initialize the memory attribute
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]        
        # self.config = config
        self.epsilon = config.get("epsilon", 1.0)  # Default value if not in config
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_shape,)))  # Add the Input layer here
        model.add(Dense(64, activation='relu'))       # Add your hidden layers
        model.add(Dense(self.action_shape, activation='linear'))  # Output layer
        model.compile(optimizer='adam', loss='mse')  # Compile the model
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_shape)
        return np.argmax(self.model.predict(np.array([state]), verbose=0))

    def train_step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        
        for s, a, r, s_next, d in minibatch:
            target = r
            if not d:
                target = r + self.gamma * np.amax(self.target_model.predict(np.array([s_next]), verbose=0))
            target_f = self.model.predict(np.array([s]), verbose=0)
            target_f[0][a] = target
            states.append(s)
            targets.append(target_f[0])
        
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        self.update_target_model()

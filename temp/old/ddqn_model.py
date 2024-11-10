import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, layers
import random
from collections import deque

class DDQNAgent:
    def __init__(self, state_shape, action_shape, config):
        self.state_shape = (state_shape,) if isinstance(state_shape, int) else state_shape
        self.action_shape = action_shape
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.min_epsilon = config["min_epsilon"]
        self.batch_size = config["batch_size"]
        self.memory = deque(maxlen=config["memory_size"])
        
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()

    def create_model(self):
        model = Sequential([
            layers.Dense(64, activation='relu', input_shape=self.state_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_shape, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # def choose_action(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return np.random.choice(self.action_shape)
    #     state = np.reshape(state, (1, -1))
    #     return np.argmax(self.model.predict(state, verbose=0))

    def choose_action(self, state):
        state = np.reshape(state, (1, -1))  # Reshape state to (1, 5)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_shape)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    # def train_step(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    #     if len(self.memory) < self.batch_size:
    #         return

    #     batch = random.sample(self.memory, self.batch_size)
    #     for state, action, reward, next_state, done in batch:
    #         target = reward
    #         if not done:
    #             target += self.gamma * np.amax(self.target_model.predict(next_state, verbose=0))

    #         target_f = self.model.predict(state, verbose=0)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)

    #     if self.epsilon > self.min_epsilon:
    #         self.epsilon *= self.epsilon_decay
    def train_step(self, state, action, reward, next_state, done):
        state = np.reshape(state, (1, -1))
        next_state = np.reshape(next_state, (1, -1))
        
        # Your remaining code for train_step
        target = reward
        if not done:
            target += self.gamma * np.amax(self.model.predict(next_state)[0])

        target_q_values = self.model.predict(state)
        target_q_values[0][action] = target

        self.model.fit(state, target_q_values, epochs=1, verbose=0)

    def save_weights(self, weights_file_path):
        self.model.save_weights(weights_file_path)  # Save weights for the Q-network

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers

class SARSAAgent:
    def __init__(self, state_shape, action_shape, config):
        self.state_shape = (state_shape,) if isinstance(state_shape, int) else state_shape
        self.action_shape = action_shape
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.min_epsilon = config["min_epsilon"]
        
        self.model = self.create_model()
        

    def create_model(self):
        model = Sequential([
          layers.Dense(64, activation='relu', input_shape=self.state_shape),
          layers.Dense(64, activation='relu'),
          layers.Dense(self.action_shape, activation='linear')
        ])  
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_shape)
        
        # Reshape state for prediction
        state = state.reshape(1, -1)  # Ensure the state is reshaped to (1, input_shape)
        return np.argmax(self.model.predict(state, verbose=0))

    def train_step(self, state, action, reward, next_state, next_action, done):
        # Reshape state and next_state for model input
        state = state.reshape(1, -1)  # Reshape to (1, input_shape)
        next_state = next_state.reshape(1, -1)  # Reshape to (1, input_shape)
        
        q_update = reward
        if not done:
            next_q_value = self.model.predict(next_state, verbose=0)[0][next_action]
            q_update += self.gamma * next_q_value
            
        q_values = self.model.predict(state, verbose=0)
        q_values[0][action] = (1 - self.learning_rate) * q_values[0][action] + self.learning_rate * q_update
        
        # Fit the model with reshaped state
        self.model.fit(state, q_values, epochs=1, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save_weights(self, weights_file_path):
        self.model.save_weights(weights_file_path)  # Save weights for the Q-network

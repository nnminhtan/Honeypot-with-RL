import tensorflow as tf
import numpy as np

class A2CAgent:
    def __init__(self, state_shape, action_shape, config):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.config = config
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_shape,)))  # Add Input layer here
        model.add(Dense(64, activation='relu'))       # Hidden layer
        model.add(Dense(self.action_shape, activation='linear'))  # Output layer
        model.compile(optimizer='adam', loss='mse')   # Compile the model
        return model


    def build_critic(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, activation='relu', input_shape=self.state_shape),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        state = np.array([state])
        probabilities = self.actor_model.predict(state, verbose=0)
        action = np.random.choice(self.action_shape, p=probabilities[0])
        return action

    def train_step(self, state, action, reward, next_state, done):
        state, next_state = np.array([state]), np.array([next_state])
        target = reward + (1 - done) * self.gamma * self.critic_model.predict(next_state, verbose=0)
        td_error = target - self.critic_model.predict(state, verbose=0)
        
        # Update critic model
        self.critic_model.fit(state, target, verbose=0)

        # Update actor model
        action_onehot = np.zeros(self.action_shape)
        action_onehot[action] = 1
        actor_loss = -np.log(self.actor_model.predict(state)[0][action]) * td_error
        self.actor_model.fit(state, action_onehot[np.newaxis, :], sample_weight=td_error.numpy(), verbose=0)

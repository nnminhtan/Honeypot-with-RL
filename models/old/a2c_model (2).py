import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError


class A2CAgent:
    def __init__(self, state_shape, action_shape, config):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        
        # Initialize actor and critic models
        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()

    def build_actor(self):
        model = Sequential([
            Input(shape=self.state_shape),
            Dense(64, activation='relu'),
            Dense(self.action_shape, activation='softmax')  # Output probabilities
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        return model

    def build_critic(self):
        model = Sequential([
            Input(shape=self.state_shape),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')  # Output single value
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def choose_action(self, state):
        state = np.array([state])
        probabilities = self.actor_model.predict(state, verbose=0)
        action = np.random.choice(self.action_shape, p=probabilities[0])
        return action
        
    @tf.function
    def train_step(self, state, action, reward, next_state, done):
        # Ensure inputs are in the correct shape and format
        state = tf.convert_to_tensor(np.reshape(state, (1, -1)), dtype=tf.float32)
        next_state = tf.convert_to_tensor(np.reshape(next_state, (1, -1)), dtype=tf.float32)
        reward = tf.convert_to_tensor([[reward]], dtype=tf.float32)  # Shape: (1, 1)
        done = tf.convert_to_tensor([[float(done)]], dtype=tf.float32)  # Shape: (1, 1)

        # Calculate target for critic model
        next_value = self.critic_model(next_state, training=False)
        target = reward + (1.0 - done) * self.gamma * next_value

        # Train critic model
        with tf.GradientTape() as tape:
            value = self.critic_model(state, training=True)
            critic_loss = tf.keras.losses.mean_squared_error(target, value)  # Ensure the loss function is correct
        grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_model.optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

        # Calculate TD error (Advantage)
        td_error = target - value

        # Train actor model
        action_onehot = tf.one_hot([action], self.action_shape)
        with tf.GradientTape() as tape:
            probs = self.actor_model(state, training=True)
            action_prob = tf.reduce_sum(action_onehot * probs, axis=1)
            actor_loss = -tf.math.log(action_prob + 1e-10) * tf.stop_gradient(td_error)  # Use TD error as advantage
            actor_loss = tf.reduce_mean(actor_loss)
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_model.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

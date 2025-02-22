import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, layers

class A2CAgent:
    def __init__(self, state_shape, action_shape, config):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        
        self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()

    def create_actor_model(self):
        model = Sequential([
            layers.Input(shape=self.state_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_shape, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy')
        return model

    def create_critic_model(self):
        model = Sequential([
            layers.Input(shape=self.state_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model

    def choose_action(self, state):
        state = np.reshape(state, (1, -1))
        probs = self.actor_model(state).numpy()
        return np.random.choice(self.action_shape, p=probs[0])

    def train_step(self, state, action, reward, next_state, done):
        state = tf.reshape(state, (1, -1))
        next_state = tf.reshape(next_state, (1, -1))
        
        with tf.GradientTape(persistent=True) as tape:
            value = self.critic_model(state)
            next_value = self.critic_model(next_state)
            target = reward + self.gamma * next_value * (1 - int(done))
            advantage = target - value
            
            action_probs = self.actor_model(state)
            action_onehot = tf.one_hot([action], self.action_shape)
            actor_loss = -tf.reduce_sum(action_onehot * tf.math.log(action_probs)) * advantage

            critic_loss = advantage ** 2

        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        
        self.actor_model.optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_model.optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

    def save_weights(self, actor_file_path, critic_file_path):
        self.actor_model.save_weights(actor_file_path)
        self.critic_model.save_weights(critic_file_path)
        print(f"Actor weights saved to {actor_file_path}")
        print(f"Critic weights saved to {critic_file_path}")

    def load_weights(self, actor_file_path, critic_file_path):
        self.actor_model.load_weights(actor_file_path)
        self.critic_model.load_weights(critic_file_path)
        print(f"Actor weights loaded from {actor_file_path}")
        print(f"Critic weights loaded from {critic_file_path}")
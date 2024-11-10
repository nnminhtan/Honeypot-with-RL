import numpy as np

class SARSAAgent:
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


    def choose_action(self, state):
      # Controlled test to check discretization
      test_state = np.array([0.5, 0.2, 0.8, 0.6, 0.4])  # Example state
      discrete_test_state = self.discretize_state(test_state)
      print(f"Test state: {test_state}, Discrete state: {discrete_test_state}")
      
      # Continue with the normal process
      print(f"Original state: {state}")  # Debug print
      discrete_state = self.discretize_state(state)
      print(f"Choosing action from discrete state: {discrete_state}")

      if np.random.rand() <= self.epsilon:
          return np.random.randint(self.action_shape)
      
      return np.argmax(self.q_table[discrete_state])

    def train_step(self, state, action, reward, next_state, next_action, done):
        discrete_state = self.discretize_state(state)

        # Log discrete state and its validity
        print(f"Discrete state: {discrete_state}, valid indices: {all(0 <= x < b for x, b in zip(discrete_state, self.bins))}")

        # Get the current Q value for the state-action pair
        try:
            current_q = self.q_table[discrete_state][action]
        except IndexError as e:
            print(f"IndexError encountered: {e}")
            print(f"Discrete state: {discrete_state}, Q-table shape: {self.q_table.shape}, Action: {action}")
            raise 
            print(f"IndexError with discrete state: {discrete_state}, Q-table shape: {self.q_table.shape}")
            discrete_state = tuple(int(x) for x in discrete_state)  # Convert elements to int
            current_q = self.q_table[discrete_state][action]
            
        try:
            # Calculate the target Q value
            target = reward + self.gamma * self.q_table[discrete_next_state][next_action] * (1 - done)
        except IndexError:
            print(f"IndexError with next discrete state: {discrete_next_state}")
            discrete_next_state = tuple(int(x) for x in discrete_next_state)  # Convert elements to int
            target = reward + self.gamma * self.q_table[discrete_next_state][next_action] * (1 - done)

        # Update the Q-table
        self.q_table[discrete_state][action] += self.learning_rate * (target - current_q)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

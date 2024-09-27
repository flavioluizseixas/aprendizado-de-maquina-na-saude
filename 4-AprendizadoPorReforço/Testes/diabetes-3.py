import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Assuming preprocessed OhioT1DM dataset is loaded
data = pd.read_csv('ohioT1DM.csv')

# Define environment variables
states = data[['glucose', 'carbs', 'insulin_dose']]  # Multi-dimensional state space
actions = np.linspace(0, 10, 21)  # Continuous insulin doses

# Build DQN model
def build_dqn_model():
    model = Sequential([
        Dense(24, input_dim=states.shape[1], activation='relu'),
        Dense(24, activation='relu'),
        Dense(len(actions), activation='linear')  # Output for each action
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Hyperparameters
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate
batch_size = 32
epochs = 1000

# Initialize the model
model = build_dqn_model()

# Experience replay buffer (to store past experiences)
experience_buffer = []

# Reward function
def reward(glucose):
    if 80 <= glucose <= 120:  # Target glucose range
        return 1
    else:
        return -1

def simulate_glucose_level(current_state, insulin_dose):
    # Unpack the current state variables
    glucose = current_state['glucose']  # Current glucose level in mg/dL
    carbs = current_state['carbs']  # Carbohydrate intake in grams
    time = current_state['time']  # Time since last insulin dose in hours
    
    # Parameters (you can adjust these based on your understanding of the system)
    carb_to_glucose_factor = 3  # How much glucose 1g of carbs increases glucose level (example)
    insulin_sensitivity = 15  # How much 1 unit of insulin decreases glucose (example)
    insulin_decay = np.exp(-0.1 * time)  # Exponential decay of insulin effect over time
    carb_metabolism_rate = np.exp(-0.05 * time)  # Decay rate of glucose increase over time
    
    # Calculate the effect of carbs on glucose level
    glucose_increase_from_carbs = carbs * carb_to_glucose_factor * carb_metabolism_rate
    
    # Calculate the effect of insulin on glucose level
    glucose_decrease_from_insulin = insulin_dose * insulin_sensitivity * insulin_decay
    
    # Update glucose level
    new_glucose = glucose + glucose_increase_from_carbs - glucose_decrease_from_insulin
    
    # Ensure glucose level is non-negative (physiologically realistic)
    new_glucose = max(new_glucose, 0)
    
    return new_glucose

# Training loop
for episode in range(epochs):
    current_state = np.random.choice(states.values)  # Initial state

    for step in range(100):  # Steps per episode
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)  # Random action
        else:
            q_values = model.predict(current_state.reshape(1, -1))  # Predict Q-values
            action = actions[np.argmax(q_values)]  # Best action

        # Simulate next state and reward
        glucose = simulate_glucose_level(current_state, action)
        
        new_state = pd.DataFrame({
            'glucose': glucose,
            'carbs': 2,
            'time': 1,
        })
        reward_value = reward(new_state)

        # Store experience
        experience_buffer.append((current_state, action, reward_value, new_state))

        # Update state
        current_state = new_state

        # Train DQN using a mini-batch from the experience buffer
        if len(experience_buffer) > batch_size:
            mini_batch = np.random.sample(experience_buffer, batch_size)
            for state, action, reward_value, next_state in mini_batch:
                target = reward_value + gamma * np.max(model.predict(next_state.reshape(1, -1)))
                q_values = model.predict(state.reshape(1, -1))
                q_values[0, actions == action] = target

                model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)

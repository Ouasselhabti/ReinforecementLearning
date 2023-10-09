import gym
from dqn import DQNAgent
from uav_env import UAVDataCollectionEnv
import numpy as np
import matplotlib.pyplot as plt

# Create the UAVDataCollectionEnv environment
n_rows = 5
m_cols = 5
env = UAVDataCollectionEnv(n_rows, m_cols)

# Set the state and action size for the DQN agent
state_size = 2  # Rows and columns of the grid (agent's position)
action_size = 4  # Up, down, left, right

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Training the DQN agent
batch_size = 32
n_episodes = 1000

# List to store cumulative rewards for each episode
cumulative_rewards = []

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0  # Initialize total reward for each episode

    for t in range(100):  # Maximum of 100 steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action,t)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward  # Accumulate the reward for each time step

        if done:
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Append the cumulative reward for this episode to the list
    cumulative_rewards.append(total_reward)

    # Print episode summary
    print(f"Episode {e + 1}/{n_episodes} - Total Reward: {total_reward:.2f}")

# Plot the cumulative reward as a function of episodes
plt.plot(range(1, n_episodes + 1), cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward vs. Episode')
plt.show()

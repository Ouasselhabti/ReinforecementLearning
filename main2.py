import tensorflow
import gym
import numpy as np
import matplotlib.pyplot as plt
from uav_env import UAVDataCollectionEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


# Create the UAVDataCollectionEnv environment
n_rows = 5
m_cols = 5
env = UAVDataCollectionEnv(n_rows, m_cols)

#model

model = Sequential()
model.add(Flatten(input_shape=(1,2)))
model.add(Dense(24,activation="relu"))
model.add(Dense(24,activation="relu"))
model.add(Dense(4,activation="linear"))


#agent

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy= GreedyQPolicy(),
    nb_actions=4,
    nb_steps_warmup=10,
    target_model_update=0.01
    )

agent.compile(tensorflow.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=["mae"])
agent.fit(env, nb_steps=10000, visualize=True, verbose=1)
# Testing
num_episodes = 100
episode_rewards = []  # To store episode rewards
for _ in range(num_episodes):
    result = agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=1000)
    episode_rewards.append(result.history["episode_reward"][0])  # Append reward of the episode

# Draw the rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Episode Rewards during Testing")
plt.show()

env.close()
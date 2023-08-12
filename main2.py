import tensorflow
import gym
import numpy as np
import matplotlib.pyplot as plt
from uav_env import UAVDataCollectionEnv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy  
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
    policy=BoltzmannQPolicy(),
    nb_actions=4,
    nb_steps_warmup=10,
    target_model_update=0.01
    )

agent.compile(tensorflow.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=["mae"])
agent.fit(env, nb_steps=1000, visualize=True, verbose=1)


results = agent.test(env, nb_episodes=1, visualize=True)
#print(np.mean(results.history["episode_reward"]))
env.close()
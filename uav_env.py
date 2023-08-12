import gym
from gym import spaces
import numpy as np
import pygame
import time
import random
import math


class UAV:
    ALPHA_S = 0.5
    BETA_S = 0.5
    START_ENERGY_S = 20000
    gamma_0_square = 1
    DEFAULT_ENLOSS = 20
    def __init__(self, x_uav, y_uav, H, alpha=ALPHA_S, beta=BETA_S, energy=START_ENERGY_S):
        self.x_uav = x_uav
        self.y_uav = y_uav
        self.altitude = H
        self.alpha = alpha
        self.beta = beta
        self.energy = energy
    @staticmethod
    def power(t):
        return random.random()

    @staticmethod
    def bandwidth(t):
        return random.randint(1500, 2000)

    @staticmethod
    def h(t):
        return random.random()

    def d(self, x, y):
        return math.sqrt((x - self.x_uav) ** 2 + (y - self.y_uav) ** 2 + self.altitude)

    def gamma(self, t, x, y):
        return (UAV.power(t) * UAV.h(t) * UAV.h(t)) / (UAV.gamma_0_square * self.d(x, y) * self.d(x, y))

    def r(self, t, x, y):
        return UAV.bandwidth(t) * math.log2(1 + self.gamma(t, x, y))

    
class UAVDataCollectionEnv(gym.Env):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.grid = np.zeros((n, m))
        self.data_collection_points = [(n - 1, m - 1)]  # List of data collection points
        self.agent_position = (0, 0)  # Initial agent position (top-left corner)
        self.done = False
        self.cumulative_reward = 0  # Cumulative reward for the current episode
        # Action space: Four possible movements (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Observation space: Tuple representing the agent's position (row, column)
        self.observation_space = spaces.Tuple((spaces.Discrete(n), spaces.Discrete(m)))

        # Set the data collection points with a different color (e.g., violet/purple) in the grid
        for x, y in self.data_collection_points:
            self.grid[x, y] = 2  # You can use any other value (e.g., 2) to represent the data collection points
        
        # Create a UAV object within the environment
        self.uav = UAV(0, 0,UAV.START_ENERGY_S, UAV.ALPHA_S, UAV.BETA_S)  # Initialize UAV object with initial position (0, 0) and altitude 0
        self.time_step = 0  # Initialize the time step
        
        # Pygame setup
        self.cell_size = 40
        self.screen_width = self.m * self.cell_size
        self.screen_height = self.n * self.cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("UAV Data Collection")
        

    def consume_energy(self,t,x,y):
        self.uav.energy -= UAV.ALPHA_S*math.log10(self.uav.r(t,x,y)) + UAV.DEFAULT_ENLOSS
        
    def reset(self):
        self.grid = np.zeros((self.n, self.m))
        self.agent_position = (0, 0)
        self.done = False
        self.cumulative_reward = 0
        return self.agent_position

    def step(self, action,t):
        assert self.action_space.contains(action)

        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        row, col = self.agent_position
        rowbef, colbef = row,col
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.n - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, self.m - 1)

        self.agent_position = (row, col)

        reward = self._get_reward(row, col,self.time_step)
        self.consume_energy(t,row,col)
        if (self.uav.energy<=0):
            self.agent_position = rowbef,colbef
            self.done = True
            raise ValueError("Episode is done. Call reset() to start a new episode.")
        self.cumulative_reward += reward
        self.grid[row, col] = 1  # Mark the visited cell with 1

        # Check if the agent has collected all data (reached all data collection points)
        if (self.uav.energy <=0):
            self.done = True
        self.time_step += 1
        return self.agent_position, reward, self.done, {}

    def render(self, mode="human"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((255,255,255))  # Fill the screen with white color
        for row in range(self.n):
            for col in range(self.m):
                if self.grid[row,col] == 2:
                    pygame.draw.rect(self.screen, (0,0,255), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
        # Draw the grid
        for row in range(self.n):
            for col in range(self.m):
                # New code to change color for data collection points to violet (purple)
                if self.grid[row, col] == 0:
                    color = (0, 0, 0)
                elif self.grid[row, col] == 2:
                    color = (0, 0, 255)  # RGB value for violet color
                else:
                    color = (0, 255, 0)

                pygame.draw.rect(self.screen, color, (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

        # Draw the agent (UAV) as a red rectangle
        pygame.draw.rect(self.screen, (255, 0, 0), (self.agent_position[1] * self.cell_size, self.agent_position[0] * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.update()

        # Add a delay to make the animation visible
        time.sleep(0.1)


    #def _get_reward(self, row, col, t):
    #    # Use the r() function from the UAV class to calculate the reward
    #    x, y = row, col  # Convert row, col to x, y coordinates
    #    reward = self.uav.r(t, x, y) if (row,col) not in self.data_collection_points else 1000
    #    # Apply an amplification factor if the reward is less than a threshold (you can customize the threshold)
    #    if reward < 5:  # Example threshold: 5
    #        reward *= 2  # Amplification factor: 2
#
#        return reward
#    def _get_reward(self, row, col, t):
#        # Use the r() function from the UAV class to calculate the base reward
#        x, y = row, col  # Convert row, col to x, y coordinates
#        base_reward = self.uav.r(t, x, y)
#        alpha  = 0.3
#        # Reward for reaching data collection points
#        data_collection_reward = 100*abs(base_reward)*t if (row, col) in self.data_collection_points else 0
#
#        # Time penalty to encourage reaching data collection points quickly
#        time_penalty = -0.1  # You can customize the penalty value, e.g., -0.1 for each time step
#
#        # Calculate the final reward as a combination of the base reward, data collection reward, and time penalty
#        reward = alpha*(base_reward + data_collection_reward) + (1-alpha)*(time_penalty * t)
#        return reward

    def _get_reward(self,row,col,t):
        return self.uav.r(t,row,col)
    
# Example usage of the custom environment
if __name__ == "__main__":
    n_rows = 5
    m_cols = 5

    env = UAVDataCollectionEnv(n_rows, m_cols)

    # Example episode with random actions
    state = env.reset()
    done = False
    t = 0  # Initialize time step
    while not done:
        env.render()p
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action,t)

        # Call the r() function from the UAV class to get the reward for this time step
        x, y = state
        current_reward = env.uav.r(t, x, y)

        print(f"Time Step {t}: Reward at ({x}, {y}) - {current_reward:.2f}")

        t += 1  # Increment time step

    env.render()
    print("Cumulative Reward:", env.cumulative_reward)

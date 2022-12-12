import numpy as np


class Model(object):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon) -> None:
        self.state_dim, self.action_dim, self.noise_dom = state_dim, action_dim, noise_dim
        self.state_space = np.arange(state_dim)
        self.action_space = np.arange(action_dim)
        self.noise_space = np.arange(noise_dim)
        self.time_horizon = time_horizon       
        
    def update(self, state, action, noise, t):
        return state+action-noise, t == self.time_horizon
        
    def reward(self, state, action, noise, t):
        return action
                
    def noise_generator(self):
        return np.random.choice(self.noise_space)
        
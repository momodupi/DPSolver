import numpy as np


class Model(object):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon, parameter={}) -> None:
        self.state_dim, self.action_dim, self.noise_dim = state_dim, action_dim, noise_dim
        self.state_space = np.arange(state_dim, dtype=int)
        self.action_space = np.arange(action_dim, dtype=int)
        self.noise_space = np.arange(noise_dim, dtype=int)
        
        self.time_horizon = time_horizon   
        self.reward_bound = parameter['reward_bound'] if 'reward_bound' in parameter else 1e5
        
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.n_dim = noise_dim
        self.s_space = np.arange(state_dim, dtype=int)
        self.a_space = np.arange(action_dim, dtype=int)
        self.n_space = np.arange(noise_dim, dtype=int)
        self.reward_flag = True
        self.state2index = {}
        self.index2state = {}
        self.action2index = {}
        self.index2action = {}
        
        # noise distribution can be varied
        self.noise_distribution = parameter['noise_distriburion'] if 'noise_distriburion' in parameter else np.ones(shape=(self.time_horizon,noise_dim))/noise_dim
        
    def update(self, state, action, noise, t):
        return min(max(state + action - noise, 0), self.state_dim-1)
        
    def reward(self, state, action, noise, t):
        return min(max(state + action + noise, -self.reward_bound), self.reward_bound)
    
    def noise_generator(self, t=0):
        noise = np.random.choice(self.noise_space, 1, p=self.noise_distribution)[0]
        return noise, self.noise_distribution[noise]
        
    def acceptance_set(self, state, action, t=0):
        return True
        
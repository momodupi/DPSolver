import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool

class Solver(object):
    def __init__(self, model) -> None:
        self.model = model   
        self.state_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.action_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.noise_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.reward_history = np.zeros(self.model.time_horizon+1)
        
        self.value_function = np.zeros(shape=(self.model.time_horizon+1, self.model.state_dim))
        self.optimal_action = np.zeros(shape=(self.model.time_horizon+1, self.model.state_dim))
        
    def expectation(self, f, t):
        return np.dot( np.vectorize(lambda n: f(n))(self.model.noise_space), self.model.noise_distribution[t] )

    def backward(self, multiprocessing=False):
        # used for vectorization
        STATE_FIELD, ACTION_FIELD = self.model.state_space.reshape(self.model.state_dim, 1), self.model.action_space.reshape(1, self.model.action_dim)
        
        # terminal value funciton
        t = self.model.time_horizon-1
        
        # compute expected R_T
        # vectorize: online post and official doc said it is not used for performance
        # however, because the input has 2 vectors, it can benefit from the vector multiplication
        reward2go = lambda s, a: self.expectation(
            lambda n: 0 if self.model.acceptance_set(s, a, t) else -np.inf, 
            t
        )
        value_matrix = np.vectorize(reward2go)(STATE_FIELD, ACTION_FIELD)
        
        self.optimal_action[t] = value_matrix.argmax(axis=1)
        self.value_function[t] = value_matrix[self.model.state_space, self.action_history[t]]
        
        for t in tqdm(range(self.model.time_horizon-2, -1, -1)):
            reward2go = lambda s, a: self.expectation(
                lambda n: self.model.reward(s, a, n, t) + self.value_function[t+1][self.model.update(s, a, n, t)] if self.model.acceptance_set(s, a, t) else -np.inf, 
                t
            )
            value_matrix = np.vectorize(reward2go)(STATE_FIELD, ACTION_FIELD)
            
            self.optimal_action[t] = value_matrix.argmax(axis=1)
            self.value_function[t] = value_matrix[self.model.state_space, self.action_history[t]]

    def forward(self, initial_state, policy):
        state = initial_state
        t = 0

        for t in range(self.model.time_horizon):
            self.state_history[t] = state
            
            action = policy(state)
            self.action_history[t] = action
            
            noise, prob = self.model.noise_generator(t)
            self.noise_history[t] = noise
            
            reward = self.model.reward(state, action, noise, t)
            self.reward_history[t] = reward
            
            next_state = self.model.update(state, action, noise, t)
            state = next_state
            
    def save_result(self, file_name='results.pickle'):
        self.res_dict = {
            'state': self.state_history,
            'action': self.action_history,
            'noise': self.noise_history,
            'reward': self.reward_history,
            'value_function': self.value_function,
            'optimal_action': self.optimal_action
        }
        
        with open(f'results/{file_name}', 'wb') as pk:
            pickle.dump(self.res_dict, pk, protocol=pickle.HIGHEST_PROTOCOL)
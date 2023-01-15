import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool

class Solver(object):
    def __init__(self, model, risk_config={'measure': 'neutral', 'parameter': None}) -> None:
        self.model = model   
        self.state_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.action_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.noise_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.reward_history = np.zeros(self.model.time_horizon+1)
        
        self.value_function = np.zeros(shape=(self.model.time_horizon+1, self.model.state_dim))
        self.optimal_action = np.zeros(shape=(self.model.time_horizon+1, self.model.state_dim))
        
        self.risk_measure = risk_config['measure']
        self.risk_parameter = risk_config['parameter']
        
        # set risk measure
        self.risk = self.risk_measure_selector(risk_config)


    def risk_measure_selector(self, risk_config):
        if risk_config['measure'] == 'CVaR':
            return self.CVaR
        else:
            return self.expectation
            
               
        
    def expectation(self, f, t):
        return np.dot( np.vectorize(lambda n: f(n))(self.model.noise_space), self.model.noise_distribution[t] )
    
    # C-VAR
    def CVaR(self, f, t):
        return
        

    def backward(self, multiprocessing=False):
        # used for vectorization
        STATE_FIELD, ACTION_FIELD = self.model.state_space.reshape(self.model.state_dim, 1), self.model.action_space.reshape(1, self.model.action_dim)
        
        # terminal value funciton
        t = self.model.time_horizon-1
        
        # compute expected R_T
        # vectorize: online post and official doc said it is not used for performance
        # however, because the input has 2 vectors, it can benefit from the vector multiplication
        unacceptance_value = -np.inf if self.model.reward_flag else np.inf
        reward2go = lambda s, a: self.risk(
            lambda n: 0 if self.model.acceptance_set(s, a, t) else unacceptance_value, 
            t
        )
        value_matrix = np.vectorize(reward2go)(STATE_FIELD, ACTION_FIELD)
        
        self.optimal_action[t] = value_matrix.argmax(axis=1)
        self.value_function[t] = value_matrix[self.model.state_space, self.action_history[t]]
        
        for t in tqdm(range(self.model.time_horizon-2, -1, -1)):
            reward2go = lambda s, a: self.risk(
                lambda n: self.model.reward(s, a, n, t) + self.value_function[t+1][self.model.update(s, a, n, t)] if self.model.acceptance_set(s, a, t) else unacceptance_value, 
                t
            )
            value_matrix = np.vectorize(reward2go)(STATE_FIELD, ACTION_FIELD)
            
            if self.model.reward_flag:
                self.optimal_action[t] = value_matrix.argmax(axis=1)
            else:
                self.optimal_action[t] = value_matrix.argmin(axis=1)
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
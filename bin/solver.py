import numpy as np
import pickle

class Solver(object):
    def __init__(self, model) -> None:
        self.model = model   
        self.state_history = np.zeros(self.model.time_horizon+1)
        self.action_history = np.zeros(self.model.time_horizon+1)
        self.noise_history = np.zeros(self.model.time_horizon+1)
        self.reward_history = np.zeros(self.model.time_horizon+1)
        
        self.value_function = np.zeros(self.model.time_horizon+1)
    
    def run(self, initial_state, policy):
        state = initial_state
        t = 0
        terminal = False
        while not terminal:
            self.state_history[t] = state
            
            action = policy(state)
            self.action_history[t] = action
            
            noise = self.model.noise_generator()
            self.noise_history[t] = noise
            
            reward = self.model.reward(state, action, noise, t)
            self.reward_history[t] = reward
            
            next_state, terminal = self.model.update(state, action, noise, t)
            state = next_state
            t += 1
            
    def save_result(self, file_name='results.json'):
        self.res_dict = {
            'state': self.state_history,
            'action': self.action_history,
            'noise': self.noise_history,
            'reward': self.reward_history
        }
        
        with open(f'results/{self.reward_history}', 'wb') as pk:
            pickle.dump(self.res_dict, pk, protocol=pickle.HIGHEST_PROTOCOL)
import numpy as np
import pickle

class Solver(object):
    def __init__(self, model) -> None:
        self.model = model   
        self.state_history = np.zeros(shape=(self.model.time_horison, self.model.state_dim))
        self.action_history = np.zeros(shape=(self.model.time_horison, self.model.action_dim))
        self.noise_history = np.zeros(shape=(self.model.time_horison, self.model.noise_dim))
        self.reward_history = np.zeros(self.model.time_horison)
    
    def run(self, initial_state, policy):
        state = initial_state
        t = 0
        terminal = True
        while terminal:
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
        res_dict = {
            'state': self.state_history,
            'action': self.action_history,
            'noise': self.noise_history,
            'reward': self.reward_history
        }
        
        with open(f'res/{self.reward_history}', 'wb') as pk:
            pickle.dump(res_dict, pk, protocol=pickle.HIGHEST_PROTOCOL)
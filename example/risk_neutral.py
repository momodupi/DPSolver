from bin.model import Model
import numpy as np


class Model_risk_neutral(Model):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon, parameter) -> None:
        super().__init__(state_dim, action_dim, noise_dim, time_horizon, parameter)
        
        self.p = parameter['unit_selling_price']
        self.c = parameter['unit_order_cost']
        self.reward_flag = True
        
    def update(self, state, action, noise, t):
        return min(max(state + action - noise, 0), self.state_dim-1)
    
    def reward(self, state, action, noise, t):
        return self.p*min(state+action, noise) - self.c*action
    
    def acceptance_set(self, state, action, t=0):
        return True
        
       
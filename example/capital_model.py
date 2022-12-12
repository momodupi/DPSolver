from bin.model import Model

import numpy as np


class Model_capital(Model):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon, parameter) -> None:
        super().__init__(state_dim, action_dim, noise_dim, time_horizon)
        
        self.unit_selling_price = parameter['p']
        self.unit_order_cost = parameter['c']
        
    def update(self, state, action, noise, t):
        return state + action - noise, t == self.time_horizon
    
    def reward(self, state, action, noise, t):
        return self.unit_selling_price*min(state+action, noise) - self.unit_order_cost*action

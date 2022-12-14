from bin.model import Model

import numpy as np


class Model_cash_order(Model):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon, parameter) -> None:
        super().__init__(state_dim, action_dim, noise_dim, time_horizon, parameter)
        
        self.p = parameter['unit_selling_price']
        self.c = parameter['unit_order_cost']
        self.xi = parameter['targer_level']
        self.w_dim = parameter['investment_level_dim']
        self.R_max = parameter['reward_bound']
        
        self.s_dim = self.state_dim
        self.state_dim = self.s_dim * self.w_dim
        
        # new state space:
        # s x w
        self.state_space = np.arange(self.state_dim, dtype=int)
        self.state2index = {}
        self.index2state = {}
        idx = 0
        for i in range(self.state_dim):
            for j in range(self.w_dim):
                self.state2index[(i,j)] = idx
                self.index2state[idx] = (i, j)
                idx += 1
        
    def update(self, state, action, noise, t):
        (s, w) = self.index2state[state]
        s = min(max(s + action - noise, 0), self.s_dim-1)
        w = min(max(w + self.reward(state, action, noise, t), 0), self.w_dim-1)
        return self.state2index[(s,w)]
    
    def reward(self, state, action, noise, t):
        (s, w) = self.index2state[state]
        return self.p*min(s+action, noise) - self.c*action
    
    def acceptance_set(self, state, action, t=0):
        (s, w) = self.index2state[state]
        return self.c*action <= w
        
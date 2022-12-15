from example.cash_order import Model_cash_order

import numpy as np


class Model_standard_capital(Model_cash_order):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon, parameter) -> None:
        super().__init__(state_dim, action_dim, noise_dim, time_horizon, parameter)
        
        self.zeta = parameter['targer_level']
        self.z_dim = parameter['investment_level_dim']
        self.a_dim = self.action_dim
        self.action_dim = self.a_dim * self.z_dim
        
        # new action space:
        self.action_space = np.arange(self.action_dim, dtype=int)
        self.action2index = {}
        self.index2action = {}
        idx = 0
        for a in range(self.a_dim):
            for z in range(self.z_dim):
                self.action2index[(a,z)] = idx
                self.index2action[idx] = (a, z)
                idx += 1
        
    def update(self, state, action, noise, t):
        (s, w) = self.index2state[state]
        (a, z) = self.index2action[action]
        s = min(max(s + a - noise, 0), self.s_dim-1)
        w = min(max(w - z, 0), self.w_dim-1)
        return self.state2index[(s,w)]
    
    def reward(self, state, action, noise, t):
        (s, w) = self.index2state[state]
        (a, z) = self.index2action[action]
        return self.p*min(s+a, noise) - self.c*a
    
    def acceptance_set(self, state, action, noise, t=0):
        (s, w) = self.index2state[state]
        (a, z) = self.index2action[action]
        
        r = self.reward(s, a, noise, t)
        if t == self.time_horizon-1:
            return w + r - z >= 0 and r + z >= self.zeta
        else:
            return r + z >= self.zeta
        
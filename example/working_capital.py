from bin.model import Model

import numpy as np


class Model_working_capital(Model):
    def __init__(self, state_dim, action_dim, noise_dim, time_horizon, parameter) -> None:
        super().__init__(state_dim, action_dim, noise_dim, time_horizon, parameter)
        
        self.p = parameter['unit_selling_price']
        self.c = parameter['unit_order_cost']
        self.w0_dim = parameter['init_investment_level_dim']
        self.R_max = parameter['reward_bound']
        self.zeta = parameter['targer_level']
        
        # compute w space based on R_max and p?
        self.w_dim = parameter['investment_level_dim']
        
        self.state_dim = self.s_dim * self.w_dim
        
        # new state space:
        # s x w
        self.state_space = np.arange(self.state_dim, dtype=int)

        idx = 0
        for s in range(self.s_dim):
            for w in range(self.w_dim):
                self.state2index[(s,w)] = idx
                self.index2state[idx] = (s, w)
                idx += 1
                
        self.reward_flag = True
                
        
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
        return w >= self.zeta and self.c*action <= w
        
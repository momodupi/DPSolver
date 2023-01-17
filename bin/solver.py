import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from scipy.integrate import quad
from functools import partial
import dill
from pathos.multiprocessing import ProcessingPool   

class Solver(object):
    def __init__(self, model, mdp_parameter={'measure': 'neutral', 'parameter': None}) -> None:
        self.model = model   
        self.state_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.action_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.noise_history = np.zeros(self.model.time_horizon+1, dtype=int)
        self.reward_history = np.zeros(self.model.time_horizon+1)
        
        self.value_function = np.zeros(shape=(self.model.time_horizon+1, self.model.state_dim))
        self.optimal_action = np.zeros(shape=(self.model.time_horizon+1, self.model.state_dim))
        
        self.risk_measure = mdp_parameter['measure']
        self.risk_parameter = mdp_parameter['parameter']
        
        # set risk measure
        self.risk = self.risk_measure_selector(self.risk_measure)
        self.cpu_num = 4

    def risk_measure_selector(self, risk_measure):
        if risk_measure == 'CVaR':
            return self.CVaR
        else:
            return self.expectation
            

       
    def expectation(self, f, t):
        # notes: n: every realization of noise --> noise_space
        # f(n): every cost2go under each the realization --> support of cost2go
        return np.dot( np.vectorize(lambda n: f(n))(self.model.noise_space), self.model.noise_distribution[t] )
    
    # C-VAR
    def CVaR(self, f, t):
        # computer VaR: at level a
        # notes: cdf of f --> cumsum of noise pdf --> since noise_space is [0,1,2,...]
        # step 1: get all possible value of f:
        f_support = np.vectorize(lambda n: f(n))(self.model.noise_space)
        # step 2: sort to get the support of f
        argsort_f_support = np.argsort(f_support)
        # step 3: cumsum to get cdf
        f_cdf = np.cumsum(self.model.noise_distribution[t][argsort_f_support])
        # step 4: var at a = min{x: F(x) > a}
        # f_cdf>a, f_cdf is already sorted 
        # position of f_cdf that make f_cdf>a
        # recover actual position by using original array
        VAR = lambda a: f_support[np.argmax(f_cdf>a)]
        # CVAR = 1/a int_0^a var_r d r
        return (1/self.risk_parameter['alpha'])*quad(VAR, 0, self.risk_parameter['alpha'])[0]
        
        
    def vectorize(self, f, x, y):
        f_partial = lambda _x: f(_x, y)
        sub_x = np.array_split(x, self.cpu_num)
        pool = ProcessingPool(nodes=self.cpu_num)
        res = pool.map(f_partial, [_x for _x in sub_x])
        return np.vstack(res)
    
    
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
        
        if multiprocessing:
            value_matrix = self.vectorize(np.vectorize(reward2go), STATE_FIELD, ACTION_FIELD)
        else:
            value_matrix = np.vectorize(reward2go)(STATE_FIELD, ACTION_FIELD)
        
        self.optimal_action[t] = value_matrix.argmax(axis=1)
        self.value_function[t] = value_matrix[self.model.state_space, self.action_history[t]]
        
        for t in tqdm(range(self.model.time_horizon-2, -1, -1)):
            reward2go = lambda s, a: self.risk(
                lambda n: self.model.reward(s, a, n, t) + self.value_function[t+1][self.model.update(s, a, n, t)] if self.model.acceptance_set(s, a, t) else unacceptance_value, 
                t
            )
            
            if multiprocessing:
                value_matrix = self.vectorize(np.vectorize(reward2go), STATE_FIELD, ACTION_FIELD)
            else:
                value_matrix = np.vectorize(reward2go)(STATE_FIELD, ACTION_FIELD)
            
            
            if self.model.reward_flag:
                self.optimal_action[t] = value_matrix.argmax(axis=1)
            else:
                self.optimal_action[t] = value_matrix.argmin(axis=1)
            self.value_function[t] = value_matrix[self.model.state_space, self.action_history[t]]

    def forward(self, initial_state, policy, seed=0):
        state = initial_state
        t = 0

        for t in range(self.model.time_horizon):
            self.state_history[t] = state
            
            action = policy(state, t)
            self.action_history[t] = action
            
            noise, prob = self.model.noise_generator(t)
            self.noise_history[t] = noise
            
            reward = self.model.reward(state, action, noise, t)
            self.reward_history[t] = reward
            
            next_state = self.model.update(state, action, noise, t)
            state = next_state
            
    def monte_carlo_trajectory(self, policy, seeds=np.arange(10), file_name='results'):
        self.trajectory = []
        for seed in seeds:
            self.state_history = np.zeros(self.model.time_horizon+1, dtype=int)
            self.action_history = np.zeros(self.model.time_horizon+1, dtype=int)
            self.noise_history = np.zeros(self.model.time_horizon+1, dtype=int)
            self.reward_history = np.zeros(self.model.time_horizon+1)
            
            initial_state = self.model.state2index[(0,20)]
            self.forward(initial_state=initial_state, policy=policy, seed=seed)
            
            self.trajectory.append({
                'state': self.state_history,
                'action': self.action_history,
                'noise': self.noise_history,
                'reward': self.reward_history,
            })
        with open(f'results/trj_{file_name}.pkl', 'wb') as pk:
            pickle.dump(self.trajectory, pk, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.trajectory
    
    def save_dp_solution(self, file_name='results'):
        self.res_dict = {
            'risk': {'measure': self.risk_measure, 'parameter': self.risk_parameter},
            'time_horizon': self.model.time_horizon,
            'model': self.model,
            'value_function': self.value_function,
            'optimal_action': self.optimal_action,
        }
        
        with open(f'results/{file_name}.pkl', 'wb') as pk:
            pickle.dump(self.res_dict, pk, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(self.res_dict)
        return self.res_dict
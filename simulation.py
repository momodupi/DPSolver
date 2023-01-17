from bin.model import Model
from example.cash_order import Model_cash_order
from example.std_capital import Model_standard_capital
from example.working_capital import Model_working_capital
from bin.model import Model
from bin.solver import Solver

def policy(state):
    return 1

if __name__ == '__main__':
    model_parameter = {
        'unit_selling_price': 10,
        'unit_order_cost': 5,
        'targer_level': 1,
        'init_investment_level_dim': 20,
        'investment_level_dim': 100,
        'reward_bound': 100
    }
    
    mdp_parameter = {
        'measure': 'CVaR',
        # 'measure': 'neutral',
        'parameter': {'alpha': 0.6},
    }

    # m = Model(5, 5, 5, 3, model_parameter)
    # m = Model_working_capital(state_dim=10, action_dim=10, noise_dim=10, time_horizon=12, parameter=model_parameter)
    m = Model_cash_order(state_dim=10, action_dim=10, noise_dim=10, time_horizon=12, parameter=model_parameter)
    # m = Model_standard_capital(state_dim=5, action_dim=5, noise_dim=5, time_horizon=5, parameter=model_parameter)
    
    s = Solver(m, mdp_parameter)
    
    s.backward(multiprocessing=True)
    s.save_dp_solution(f'{m.model_name}_{mdp_parameter["measure"]}')
    
    mdp_trajectories = []
    policy = lambda state, t: s.optimal_action[t][state]
    s.monte_carlo_trajectory(policy=policy, file_name=f'{m.model_name}_{mdp_parameter["measure"]}')
    
    

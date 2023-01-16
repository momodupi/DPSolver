from bin.model import Model
from example.cash_order import Model_cash_order
from example.std_capital import Model_standard_capital
from example.working_capital import Model_working_capital
from example.risk_neutral import Model_risk_neutral
from bin.model import Model
from bin.solver import Solver

def policy(state):
    return 1

if __name__ == '__main__':
    model_parameter = {
        'unit_selling_price': 10,
        'unit_order_cost': 5,
        'targer_level': 1,
        'init_investment_level_dim': 1000,
        'investment_level_dim': 10,
        'reward_bound': 1000
    }
    
    mdp_parameter = {
        'measure': 'CVaR',
        'parameter': {'alpha': 0.9}
    }
    
    m = Model_cash_order(5, 5, 5, 3, model_parameter)
    # m = Model_standard_capital(100, 100, 10, 10, parameter)
    # m = Model_working_capital(100, 100, 10, 10, parameter)
    # m = Model_risk_neutral(5, 5, 5, 3, parameter)
    
    
    s = Solver(m, mdp_parameter)
    
    s.backward()
    
    s.save_result()
    print(s.res_dict)
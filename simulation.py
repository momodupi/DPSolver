from bin.model import Model
from example.cash_order import Model_cash_order
from bin.solver import Solver

def policy(state):
    return 1

if __name__ == '__main__':
    parameter = {
        'unit_selling_price': 10,
        'unit_order_cost': 2,
        'targer_level': 3,
        'investment_level_dim': 40,
        'reward_bound': 1000
    }
    
    m = Model_cash_order(30, 30, 30, 24, parameter)
    s = Solver(m)
    
    s.backward()
    
    s.save_result()
    print(s.res_dict)
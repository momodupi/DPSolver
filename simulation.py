from bin.model import Model
from bin.solver import Solver

def policy(state):
    return 0

if __name__ == '__main__':
    m = Model(2, 2, 4, 24)
    s = Solver(m)
    
    s.run(0, policy=policy)
    
    s.save_result()
    print(s.res_dict)
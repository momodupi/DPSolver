from bin.solver import Solver

class Solver_capital(Solver):
    def __init__(self, model, parameter) -> None:
        super().__init__(model)
        
    
    def backprop(self):
        # termial value function
        self.value_function[-1] = 0
        
        for t in range(self.model.time_horizon-2, -1, -1):
            pass
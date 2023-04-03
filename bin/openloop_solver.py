import numpy as np
from scipy.optimize import linprog
from numpy.linalg import matrix_power
from scipy.optimize import LinearConstraint

class OpenLoopSolver(object):
    def __init__(self, parameters: dict) -> None:
        pass 
        self.A = np.array(parameters["A"])
        self.B = np.array(parameters["B"])
        self.T = np.array(parameters["T"])
        self.c_list = [np.array(_) for _ in parameters["c"]]
        self.H_list = [np.array(_) for _ in parameters["H"]]
        self.G_list = [np.array(_) for _ in parameters["G"]]
        self.d_list = [np.array(_) for _ in parameters["d"]]
    
        self.x_0 = np.array(parameters["x_0"])
        
        self.x_dim = len(self.x_0)
        self.u_dim = self.B.shape[1]
        self.d_dim = len(self.d_list[0])
        
        # A should be a square matrix with dim = x
        assert(self.A.shape[1] == self.x_dim)
        assert(self.A.shape[0] == self.A.shape[1])
        # B rows = A rows
        assert(self.B.shape[0] == self.A.shape[0])
        # dim in c H G = T
        assert(len(self.c_list) == self.T)
        assert(len(self.H_list) == self.T)
        assert(len(self.G_list) == self.T)
        assert(len(self.d_list) == self.T)
        
        # d dim = H G rows
        # u dim = c 
        for t in range(self.T):
            assert(self.u_dim == self.c_list[t].shape[0])
            assert(self.d_dim == self.H_list[t].shape[0])
            assert(self.d_dim == self.G_list[t].shape[0])
        
        # I didn"t assert the dim in H G matching x u d
        # please carefully give the input
        
    def get_c(self, c):
        return np.array(c).flatten()
    
    def get_bd(self, time_horizon, u_dim, lb=0, ub=np.inf):
        return np.array([np.ones(time_horizon*u_dim)*lb, np.ones(time_horizon*u_dim)*ub]).T
    
    def get_ineq(self, time_horizon, u_dim, d_dim):
        """
        update: x_{t+1} = A x_t + B u_t
        constraint: H_t x_t + G_t u_t + d_t <= 0
        we need to convert them into a function of x_0

        G_t u_t <= -d_t - H_t x_t
                <= -d_t - H_t ( A x_{t-1} + B u_{t-1} )
                <= -d_t - H_t( A^2 x_{t-2} + AB u_{t-2} + B u_{t-1} )
                <= -d_t - H_t( A^3 x_{t-3} + A^2B u_{t-3} + AB u_{t-2} + B u_{t-1} )
                ...
                <= -d_t - H_t( A^t x_0 + A^{t-1} B u_0 + A^{t-2} B u_1 + ... + AB u_{t-2} + B u_{t-1} )
        which is
        H_t A^{t-1} B u_0 + H_t A^{t-2} B u_1 + ... + B u_{t-1} + G_t u_t <= -d_t - H_t A^t x_0
        
        now let"s put them into a big matrix:
        G_0,     0,     0,   ..., 0   <-->   u_0 <= -d_0 - H_0 x_0
        H_1 B,   G_1,   0,   ..., 0   <-->   u_1 <= -d_1 - H_1 A x_0
        H_2 A B, H_2 B, G_1, ..., 0   <-->   u_2 <= -d_2 - H_2 A^2 x_0
        """
        # dim of inequality
        Ineq_A_shape = (time_horizon*u_dim, time_horizon*d_dim)
        Ineq_A_list, Ineq_b_list = [], []
        
        for t in range(time_horizon):
            """
            at time t: there are t+1 non-zero blocks, and T-t-1 zero blocks
            non_zero blocks: the i-th one: H_t A^{t-i-1} B
                            the t-th one: G_t
            the rest are zero blocks: with dim=(d, u)
            """
            blocks_t = []
            # i = 0,...,t-1
            for i in range(t):
                blocks_t.append(
                    self.H_list[t].dot( matrix_power(self.A, t-i-1) ).dot(self.B)
                )
            # i = t
            blocks_t.append(self.G_list[t])
            # i = t+1 ... T-1
            for i in range(t+1, time_horizon):
                blocks_t.append(np.zeros(shape=(d_dim,u_dim)))

            # convert small blocks into a fat matrix
            blocks = np.hstack(blocks_t)
            # and append them into inequality matrix
            Ineq_A_list.append(blocks)
            
            """
            b_ineq = -d_t-H_t A^t x_0
            """
            Ineq_b_list.append( -self.d_list[t] - self.H_list[t].dot( matrix_power(self.A, t) ).dot(self.x_0) )
        
        print("A list: ", Ineq_A_list)
        
        # convert all fat matrices into a big matrix
        Ineq_A = np.vstack(Ineq_A_list)
        Ineq_b = np.hstack(Ineq_b_list)
        
        return Ineq_A, Ineq_b

    def run(self, method="highs"):
        # `method='interior-point'` is deprecated and will be removed in SciPy 1.11.0: use highs instead
        c = self.get_c(self.c_list)
        A_ineq, b_ineq = self.get_ineq(time_horizon=self.T, u_dim=self.u_dim, d_dim=self.d_dim)
        bd = self.get_bd(time_horizon=self.T, u_dim=self.u_dim, lb=0, ub=np.inf)
        print(c)
        print(A_ineq)
        print(b_ineq)
        print(bd)
        res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, bounds=bd, method=method)
        return res
        
        
    
if __name__ == "__main__":
    # x_dim = 5
    # u_dim = 5
    # time_horizon = 3
    # PARAMETER = {
    #     "A": np.eye(x_dim),
    #     "B": np.eye(u_dim),
    #     "T": time_horizon,
    #     "c": [np.random.uniform(0,1, size=u_dim) for _ in range(time_horizon)],
    #     "H": [np.eye(x_dim) for _ in range(time_horizon)],
    #     "G": [np.eye(u_dim) for _ in range(time_horizon)],
    #     "d": [-np.ones(x_dim)*10 for _ in range(time_horizon)],
    #     "x_0": np.ones(x_dim),
    # }
    
    x_dim = 4
    u_dim = 3
    d_dim = 2
    time_horizon = 3
    PARAMETER = {
        "A": np.eye(x_dim)*5,
        "B": [[-0.5,0,0],[0,-0.5,0],[0,-0.2,-0.3], [-0.1,0,-0.4]],
        "T": time_horizon,
        "c": [np.random.uniform(0,1, size=u_dim) for _ in range(time_horizon)],
        "H": [[[1,0,1,0],[0,1,0,1]] for _ in range(time_horizon)],
        "G": [[[2,2,0],[0,2,2]] for _ in range(time_horizon)],
        "d": [-np.ones(d_dim)*100 for _ in range(time_horizon)],
        "x_0": np.ones(x_dim),
    }
    
    
    ol_solver = OpenLoopSolver(parameters=PARAMETER)
    res = ol_solver.run()
    print(res)
    print(res.ineqlin.marginals)
    
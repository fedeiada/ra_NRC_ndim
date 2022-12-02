import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
import math as mt
import itertools as it

'''
Compute the optimal point with a centralized solution. The result is used to print the plots of MSE and distance until opt
'''

class CentralOptim():

    def __init__(self, n_agents, dim, id, simulation_function_xtx_btx):
        self.n_agents = n_agents
        self.f = 0
        self.compute = False
        self.AA = np.zeros((dim, dim))
        self.bb = np.zeros(dim)
        self.x0 = np.zeros(dim)
        self.x = np.zeros(dim)
        self.count = 0
        self.id = id
        self.simulation_function_xtx_btx = simulation_function_xtx_btx


    def OptimalCentralSolution(self, A, b, x0):
        self.x0 += x0 / (self.n_agents)
        self.AA += A
        det = np.linalg.det(self.AA)
        self.bb += b
        self.count += 1
        if self.count == self.n_agents:
            self.compute = True
        if self.compute:
            res = opt.minimize(self.simulation_function_xtx_btx.get_fn,
                               self.x0,
                               args=(self.AA, self.bb, self.id),
                               method='Newton-CG',
                               # constraints=cons,
                               # bounds=bnds,
                               jac=self.simulation_function_xtx_btx.get_gradient_fn,
                               hess=self.simulation_function_xtx_btx.get_hessian_fn,
                               tol=1e-3,
                               options={'maxiter': 2000,
                                        'disp': True,
                                        # 'gtol':1e-3,
                                        })
            return res.x
        return
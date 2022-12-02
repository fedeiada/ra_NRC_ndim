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
        self.compute = False
        self.AA = np.zeros((dim, dim))
        self.bb = np.array([np.zeros(dim)])
        self.x0 = np.array([np.zeros(dim)])
        #self.x = np.zeros(dim)
        self.count = 0
        self.id = id
        self.simulation_function_xtx_btx = simulation_function_xtx_btx


    def OptimalCentralSolution(self, A, b, x0):
        self.x0 += (x0 / (self.n_agents))
        '''self.AA.append(A)
        det = np.linalg.det(self.AA)
        self.bb.append(b)'''
        self.AA += A
        self.bb += b
        self.count += 1
        if self.count == self.n_agents:
            self.compute = True
        if self.compute:
            self.res = opt.minimize(self.simulation_function_xtx_btx.get_fn,
                               self.x0,
                               args=(self.AA, self.bb, self.id),
                               method='Newton-CG', #'trust-ncg',
                               # constraints=cons,
                               # bounds=bnds,
                               jac=self.simulation_function_xtx_btx.get_gradient_fn,
                               hess=self.simulation_function_xtx_btx.get_hessian_fn,
                               tol=1e-3,
                               options={'maxiter': 2000,
                                        'disp': True,
                                        # 'gtol':1e-3,
                                        })
            return
        return

    def f(self, x):
        f = self.simulation_function_xtx_btx.get_fn(x, self.AA[0], self.bb[0], self.id) + \
            self.simulation_function_xtx_btx.get_fn(x, self.AA[1], self.bb[1], self.id) + \
            self.simulation_function_xtx_btx.get_fn(x, self.AA[2], self.bb[2], self.id)
        return f
    def g(self, x):
        g = self.simulation_function_xtx_btx.get_gradient_fn(x, self.AA[0], self.bb[0], self.id) + \
            self.simulation_function_xtx_btx.get_gradient_fn(x, self.AA[1], self.bb[1], self.id) + \
            self.simulation_function_xtx_btx.get_gradient_fn(x, self.AA[2], self.bb[2], self.id)
        return g
    def h(self, x):
        h = self.simulation_function_xtx_btx.get_hessian_fn(x, self.AA[0], self.bb[0], self.id) + \
            self.simulation_function_xtx_btx.get_hessian_fn(x, self.AA[1], self.bb[1], self.id) + \
            self.simulation_function_xtx_btx.get_hessian_fn(x, self.AA[2], self.bb[2], self.id)
        return h
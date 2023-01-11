import matplotlib.pyplot as plt
#import numpy as np
from scipy import optimize as opt
import math as mt
import itertools as it
import autograd.numpy as np
from autograd import jacobian
'''
Compute the optimal point with a centralized solution. The result is used to print the plots of MSE and distance until opt
'''

class CentralOptim():

    def __init__(self, n_agents, dim, id, simulation_function_xtx_btx):
        self.n_agents = n_agents
        self.compute = False
        self.AA = []  # np.zeros((dim, dim))
        self.bb = []  # np.array([np.zeros(dim)])
        self.x0 = np.array([np.zeros(dim)])
        #self.x = np.zeros(dim)
        self.count = 0
        self.id = id
        self.simulation_function_xtx_btx = simulation_function_xtx_btx


    def OptimalCentralSolution(self, A, b, x0):
        self.x0 += x0 / (self.n_agents)
        self.AA.append(A)
        self.bb.append(b)
        '''self.AA += A
        self.bb += b'''
        self.count += 1
        if self.count == self.n_agents:
            self.compute = True
        if self.compute:
            self.res = opt.minimize(self.fs,
                               self.x0,
                               args=(self.AA, self.bb, self.id, self.simulation_function_xtx_btx),
                               method='Newton-CG',
                               jac=self.gs,
                               hess=self.hs,
                               tol=1e-4,
                               options={'maxiter': 5000,
                                        'disp': True,
                                         #'gtol':1e-4,
                                        })
            return
        return

    @staticmethod
    def fs(x, AA, bb, id, sim_fun):
        f = sim_fun.get_fn(x, AA[0], bb[0], id) + \
            sim_fun.get_fn(x, AA[1], bb[1], id) + \
            sim_fun.get_fn(x, AA[2], bb[2], id)
        #f = sim_fun.get_fn(x, AA, bb, id)
        return f

    @staticmethod
    def gs(x, AA, bb, id, sim_fun):
        g = sim_fun.get_gradient_fn(x, AA[0], bb[0], id) + \
            sim_fun.get_gradient_fn(x, AA[1], bb[1], id) + \
            sim_fun.get_gradient_fn(x, AA[2], bb[2], id)
        return g
    @staticmethod
    def hs( x, AA, bb, id, sim_fun):
        h = sim_fun.get_hessian_fn(x, AA[0], bb[0], id) + \
            sim_fun.get_hessian_fn(x, AA[1], bb[1], id) + \
            sim_fun.get_hessian_fn(x, AA[2], bb[2], id)
        return h

    '''@staticmethod
        def gs(x, AA, bb, id, sim_fun):
            g = jacobian(sim_fun.get_fn)(x, AA[0], bb[0], id) + \
                jacobian(sim_fun.get_fn)(x, AA[1], bb[1], id) + \
                jacobian(sim_fun.get_fn)(x, AA[2], bb[2], id)
            #g = jacobian(sim_fun.get_fn)(x, AA, bb, id)
            return g

        @staticmethod
        def hs(x, AA, bb, id, sim_fun):
            h = jacobian(jacobian(sim_fun.get_fn))(x, AA[0], bb[0], id) + \
                jacobian(jacobian(sim_fun.get_fn))(x, AA[1], bb[1], id) + \
                jacobian(jacobian(sim_fun.get_fn))(x, AA[2], bb[2], id)
            #h = jacobian(jacobian(sim_fun.get_fn))(x, AA, bb, id)
            return h'''
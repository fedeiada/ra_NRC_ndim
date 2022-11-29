import numpy
import random

import numpy as np


class SimulationFunctionXTX_BTX:
    """This class representing the simulation function. This class can be used to calculate the gradient and hessian of the mentioned function. """


    @staticmethod
    def get_fn(x: numpy.array, cost_fun, coeff) -> numpy.array:
        """This method can be used to calculate the outcome of the function for each given Xi and Bi"""
        if cost_fun == 3:
            f = coeff[0] * (np.exp(coeff[1] * (x + coeff[2])))
        elif cost_fun == 2:
            f = coeff[0] * (x + coeff[1]) ** 2
        elif cost_fun == 5:
            f = coeff[0] * ((x+2) ** 4) #+ coeff[1] * (x ** 3) + coeff[2] * (x ** 2) + coeff[3] * x + coeff[4]
        return numpy.array(f)

    @staticmethod
    def get_gradient_fn(x: numpy.array, cost_fun, coeff) -> numpy.array:
        """This method can be used to calculate the gradient for any given Xi."""
        if cost_fun == 3:
            g = (coeff[0] * coeff[1]) * (np.exp(coeff[1] * (x + coeff[2])))
        elif cost_fun == 2:
            g = 2 * coeff[0] * (x + coeff[1])
        elif cost_fun == 5:
            g = 4 * coeff[0] * ((x+2) ** 3) #+ 3 * coeff[1] * (x ** 2) + 2 * coeff[2] * x + coeff[3]
        return numpy.array(g)

    @staticmethod
    def get_hessian_fn(x: numpy.array, cost_fun, coeff) -> numpy.array:
        """This method can be used to calculate the hessian for any given Xi."""
        if cost_fun == 3:
            h = (coeff[0] * (coeff[1] ** 2)) * (np.exp(coeff[1] * (x + coeff[2])))
        elif cost_fun == 2:
            h = 2 * coeff[0]
        elif cost_fun == 5:
            h = 12 * coeff[0] * ((x+2) ** 2) + 0.000005 #+ 6 * coeff[1] * x + 2 * coeff[2]
        return numpy.array(h)


    '''@staticmethod
    def get_fn(x: numpy.array) -> numpy.array:
        """This method can be used to calculate the outcome of the function for each given Xi and Bi"""
        # f = 3 * (x+1)**2
        f = 0.1 * ((x) ** 4)
        # f = numpy.exp(-x)
        return numpy.array(f)

    @staticmethod
    def get_gradient_fn(x: numpy.array) -> numpy.array:
        """This method can be used to calculate the gradient for any given Xi."""
        # g = 6 * (x+1)
        g = 0.4 * (x ** 3)
        # g = - numpy.exp(-x)
        return numpy.array(g)

    @staticmethod
    def get_hessian_fn(x: numpy.array) -> numpy.array:
        """This method can be used to calculate the hessian for any given Xi."""
        # h = 6
        h = 1.2 * (x ** 2)
        # h = numpy.exp(-x)
        return numpy.array(h)'''

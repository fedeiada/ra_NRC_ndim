import numpy
import random

import numpy as np


class SimulationFunctionXTX_BTX:
    """This class representing the simulation function. This class can be used to calculate the gradient and hessian of the mentioned function. """


    @staticmethod
    def get_fn(x: numpy.array, A, b) -> numpy.array:
        """This method can be used to calculate the outcome of the function for each given Xi, Ai and Bi"""
        f = numpy.matmul(numpy.matmul(numpy.transpose(x),A), x) + numpy.matmul(numpy.transpose(b), x)
        return numpy.array(f)

    @staticmethod
    def get_gradient_fn(x: numpy.array, A, b) -> numpy.array:
        """This method can be used to calculate the gradient for any given Xi."""
        g = 2*np.matmul(A, x) + b
        return numpy.array(g)

    @staticmethod
    def get_hessian_fn(x: numpy.array, A) -> numpy.array:
        """This method can be used to calculate the hessian for any given Xi."""
        return numpy.array(2*A)


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

import numpy
import random

import numpy as np


class SimulationFunctionXTX_BTX:
    """This class representing the simulation function. This class can be used to calculate the gradient and hessian of the mentioned function. """


    @staticmethod
    def get_fn(x: numpy.array, A, b, id) -> numpy.array:
        """This method can be used to calculate the outcome of the function for each given Xi, Ai and Bi"""
        if id == 'quad':
            f = numpy.matmul(numpy.matmul(numpy.transpose(x), A), x) + numpy.matmul(numpy.transpose(b), x)
        elif id == 'exp':
            f = np.matmul(b[0], np.exp(np.matmul(A[0], x))) + np.matmul(b[1], np.exp(np.matmul(-A[1], x)))
        return numpy.array(f)

    @staticmethod
    def get_gradient_fn(x: numpy.array, A, b, id) -> numpy.array:
        """This method can be used to calculate the gradient for any given Xi."""
        if id == 'quad':
            g = 2*np.matmul(A, x) + b
        elif id == 'exp':
            g = np.matmul(np.matmul(b[0], A[0].transpose()), np.exp(np.matmul(A[0], x))) - \
                np.matmul(np.matmul(b[1], A[1].transpose()), np.exp(np.matmul(-A[1], x)))
        return numpy.array(g)

    @staticmethod
    def get_hessian_fn(x: numpy.array, A, b, id) -> numpy.array:
        """This method can be used to calculate the hessian for any given Xi."""
        if id == 'quad':
            h = 2*A
        elif id == 'exp':
            h = np.matmul(np.matmul(np.matmul(b[0], A[0].transpose()), A[0]), np.exp(np.matmul(A[0], x))) + \
                np.matmul(np.matmul(np.matmul(b[1], A[1].transpose()), A[1]), np.exp(np.matmul(-A[1], x))) #+ np.array([[0.0001, 0],[0, 0.0001]])
        return numpy.array(h)



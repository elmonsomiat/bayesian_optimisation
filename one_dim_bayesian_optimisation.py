import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  RBF, ConstantKernel as C

class BayesianOptimizer(object):
    """docstring for BayesianOptimizer"""
    def __init__(self, kernel=None):
        if kernel==None:
            self.kernel = C(1.0, (1, 1e2)) * RBF(1, (1e-2, 1))
        else:
            self.kernel=kernel
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=50)

    def acquisition_function_calculation(self, y_pred, sigma):
        '''
        Calculate the (discrete) acquisition function following:
            Efficient Global Optimization of Expensive Black-Box Functions 
        '''
        # Get position of the minimum y
        y_min = y_pred.min()
        index_y_min = np.argmin(y_pred)
        # Shift y to y_min=0
        y_shift = y_pred - y_min
        y_shift = y_shift.reshape(len(y_shift),)
        # Calculate the acquisition function 
        Z = np.divide(y_shift, sigma)
        a_func = y_shift*norm.pdf(Z) + sigma*norm.pdf(Z)
        a_func[sigma==0] = 0
        return a_func

    def calculate_next_x(self, a_func, x):
        '''
        Calculate the next x point
        The logic is as follows: the aquisition function indicates the uncertainty of the 
        function we are trying to reproduce. Hence, the next x point needs to be where
        the uncertainty is the maximum
        '''
        index_a_func_max = np.argmax(a_func)
        x_next = x[index_a_func_max]
        return x_next

    def predict_gaussian_process(self, X, Y, x, show_acquisition, show_fit=True):
        '''
        predict the
        '''
         
        self.gpr.fit(X, Y)
        y_pred, sigma = self.gpr.predict(x, return_std=show_acquisition)
        if show_fit:
            self.plot_fit(X, Y, x, y_pred, sigma)
        return y_pred, sigma

    def plot_fit(self, X, Y, x, y_pred, sigma):
        fig = plt.figure()
        plt.plot(X, Y, 'r.', markersize=10, label='Loss')
        plt.plot(x, y_pred, 'b-', label='Predicted loss')
        plt.fill_between(x.reshape(len(x), ), y_pred.reshape(len(y_pred),)-sigma,
             y_pred.reshape(len(y_pred),)+sigma, alpha=0.5)
        plt.show()
        pass

    def append_next_point(self, X, Y, x, show_acquisition):
        '''
        Calculate and append the next X point
        Args:
            X (np.array): Initial values of the parameter which wants to be optimised
            Y (np.array): Corresponding values of the loss function given X
            x (np.array): Array of numbers between the values that X wants to be searched between

        Returns:
            X, Y: Updated values for X and Y (with new points)
        '''
        y_pred, sigma = self.predict_gaussian_process(X, Y, x, show_acquisition)
        a_func = self.acquisition_function_calculation(y_pred, sigma)
        x_next = self.calculate_next_x(a_func, x)
        if show_acquisition:
            plt.plot(x, a_func, 'b-', label='Aquisition function')
            plt.show()
        X = np.append(X, x_next).reshape(-1, 1)
        y_next = self.calculate_loss_func(x_next)
        Y = np.append(Y, y_next)
        return X, Y

    def run_optimizer(self,  X, Y, x, show_acquisition=True, n_steps=5):
        '''
        run the optimiser and return the gaussian process regression
        '''
        for i in range(n_steps):
            X, Y = self.append_next_point(X, Y, x, show_acquisition)
        return self.gpr


    def calculate_loss_func(self):
        '''This function needs to be implemented by user'''
        pass


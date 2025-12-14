#various loss function with their derivatives here

import numpy as np

def mse(y_true , y_pred):
    # return 0.5*(y_true - y_pred)**2
    return np.mean(np.power(y_true - y_pred, 2))
    
def mse_prime(y_true , y_pred):
    return 2 * (y_true - y_pred)/np.size(y_true)

def binary_cross_entropy(y_true , y_pred):
    return 0.5*(-y_true * np.log(y_pred) - (1 - y_true)*np.log(1 - y_pred)) #only two values for binary cross entropy function
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true)*np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true , y_pred):
    # return ( (1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
    return grad / np.size(y_true)
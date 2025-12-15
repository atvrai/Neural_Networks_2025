#various loss function with their derivatives here

import numpy as np

def mse(y_true , y_pred):
    # return 0.5*(y_true - y_pred)**2
    return np.mean(np.power(y_true - y_pred, 2))
    
def mse_prime(y_true , y_pred):
    return 2 * (y_true - y_pred)/np.size(y_true)

def binary_cross_entropy(y_true , y_pred):
    # return 0.5*(-y_true * np.log(y_pred) - (1 - y_true)*np.log(1 - y_pred)) #only two values for binary cross entropy function
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) #for clippping the output when it goes to -inf in any case and prevents NaN error 
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true)*np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true , y_pred):
    # return ( (1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    grad = (y_pred - y_true) / (y_pred * (1 - y_pred))
    return grad / np.size(y_true)

def categorical_cross_entropy(y_true , y_pred): 
    # Clip probabilities to avoid log(0) which results in infinite loss.The smallest number should be larger than 0.
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    
    # Loss formula: - (1/N) * sum(y_true * log(y_pred))
    # where the sum is over all classes and all samples (N).
    # np.sum(y_true * np.log(y_pred)) effectively only includes the log(predicted probability)
    # for the correct (one-hot) class.
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# def categorical_cross_entropy_prime(y_true , y_pred): #SLOWER
#     # Clip probabilities to avoid division by zero.
#     y_pred = np.clip(y_pred, 1e-12, 1.0)
    
#     # Gradient w.r.t. the prediction: - (1/N) * (y_true / y_pred)
#     # This is the derivative of the mean loss w.r.t. the Softmax output (y_pred).
#     grad = -y_true / y_pred    
#     # Divide by the number of samples (N) to account for the mean loss.
#     return grad / y_true.shape[0]

def categorical_cross_entropy_prime(y_true , y_pred): #Faster
    # 'y_pred' here is the Softmax output (A)
    # This calculates the final gradient to pass backward to the previous Dense layer (dJ/dz)
    return (y_pred - y_true) / y_true.shape[0] # Divide by batch size for mean loss


#activation layer for the step with forward and backward propgations included applied to the output of a dense layer

import numpy as np
from layer import Layer

class Activation:
    def __init__(self , activation , activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self,input):
        self.input = input
        return self.activation(self.input) #pass the input to the activation function
    
    def backward(self , output_gradient , learning_rate): #returning the reviced error from front layer and applying the derivative of the activation function
        return np.multiply(output_gradient , self.activation_prime(self.input))   #product of the dervatives from the activation fn and the output gradient from
        
#forward and backward pass functions added
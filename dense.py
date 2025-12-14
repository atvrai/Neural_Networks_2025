#dense layer of the NN conataining the forward and the backward passes main hear where the tuning of the weights and biases takes place___

import numpy as np
from layer import Layer

class Dense:
    def __init__(self , input_size , output_size):      #i have intialised with zeros but can be done with random values as well in a particular range...
        #shape of the weights to match the dot product of the input for the output..
        
        # self.weights = np.zeros(output_size , input_size)
        # self.bias = np.zeros(output_size , 1)
        
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        
    def forward(self , input):
        self.input = input
        return np.dot(self.weights , self.input) + self.bias
        
    def backward(self , output_gradient , learning_rate):  #backward pass to the previous later if there and adjustment of weights and biases
        weights_gradient = np.dot(output_gradient , self.input.T)
        input_gradient = np.dot(self.weights.T , output_gradient)
        self.weights = self.weights - (learning_rate * weights_gradient)
        self.bias = self.bias - (learning_rate * output_gradient)
        return input_gradient #value passed to the previous layer...
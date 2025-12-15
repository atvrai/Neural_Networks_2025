#Convolution from scracth as well function added with its implementation in for it in this....

import numpy as np
from scipy import signal
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape     #equal to the input layer like say if its and RGB image then its going to be 3 x 224 x 224
        self.depth = depth                                       #this depth is the number of kernels used to convolve with the given input
        self.input_shape = input_shape    
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)    #output shape = (I - K + 1) / S if S given..
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)     #kernal structure 4D matrix --> no_of_krn's * input_depth * h_k * w_k
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)    #shape of bias is equal to the output shape

    def forward(self, input):   #Y = b + X *|K  ...note order of convolution operation matters a lot
        self.input = input
        self.output = np.copy(self.biases)                   #output intialsed with bias as its need to be added anyway only a shallow copy made
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")     #we are taking the cor-relate as convolution over here...
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)   #intialise the gradeints
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")     #check from the derivation...
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")      #this was a conv op needed for us...

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
#valid correlation is the type in which we are reducing the input kernel's size to a smaller size
#full correlation is the type of correlation in which we are reducing the making use of the img padding for increasing the output size...needed in backpropogation step
#we are implementing cross-correlation from the scipy library and not doing ourselve's
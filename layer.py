#layer class all the other classes are calling for layer implementaion when needed..

#so main use of layer is it takes the input and passes it to the type of layer we want either a Dense One or an activation one we can just pass it to it hehe...
#and the function definition is exactly same so we can just write the name of the type of layer we want and the code will work
#in netwrok just writing the work Dense() , Tanh() specifies the type of layer wanted and call the respective class 

#this is the beauty of the object oriented programming implementation we can make how so many layer and which ever type we want without changing much of the code

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input): #same as in dense forward prediction compute karke aage send kar deta hai
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):  #same as the function in the dense layer where this backward step is being passed for the update
        # TODO: update parameters and return input gradient
        pass
import numpy as np
from keras.datasets import mnist
# from keras.utils import np_utils --->outdated not working
from tensorflow.keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime  #we using binary cross entropy_loss function here as we are onyly trying for 0 and 1 identification
from network import train, predict

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]  #no if inputs to take based on the limit 
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index)) #hstack is used to horizontally stack these the array's side by side
    #see as we are making a new dataset kindof for the 0 and 1 values only so we stacking them side by side after extracting and on the stacked new array
    #we are applying a permuatation so that dataset becomes random and prevent biases from developing in the model..
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)  #reshape in the X as a 4D matrix = Number of Samples/Batch Size * Depth/Channels(RGB/Greysacale) * Height * Width 
    x = x.astype("float32") / 255
    # y = np_utils.to_categorical(y)
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)       #same kinda reshape but with output having only 2 value to give out [0 , 1] -->
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# Convolutional neural network implemnetation we can take 2D matrix as input in this...
network = [
        Convolutional((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
]

#Prameters to pass to netwrok class for train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
train(network,binary_cross_entropy,binary_cross_entropy_prime,x_train,y_train,epochs=20,learning_rate=0.1)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
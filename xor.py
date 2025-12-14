import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dense import Dense
from activations import Tanh , Sigmoid , Softmax
from losses import mse , mse_prime , binary_cross_entropy , binary_cross_entropy_prime
from network import train, predict


# input's to train XOR on these...
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# pseudo network architecture made over here for proper understanding...
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)

points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

        
#showing the output of this neral network as a 3D block
points = np.array(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()

#type this in console to see output
#%run xor.py
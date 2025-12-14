#network fucntion called for the training of the layer 

def predict(network, input):
    output = input
    for layer in network:       #see this is the hidden gem in the code this predict function did the full forward pass to get the output for the neural net
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)  #predicting is indirectly the output for each layer

            # 2. CALCULATE ERROR(LOSS)
            error += loss(y, output)

            # 3. BACKWARD PASS (The learning part!) after calculating the gradient
            grad = loss_prime(y, output)
            for layer in reversed(network): #going backwards in the netwrok array
                grad = layer.backward(grad, learning_rate)    #taking the values for each layer and passing the output values # <-- THIS IS WHERE WEIGHTS ARE UPDATED

        error /= len(x_train)
        if verbose and e%10==0:
            print(f"{e + 1}/{epochs}, error={error}")
import numpy as np
import random
class MLP:
    def __init__(self, input_dim, output_dim, f_act='sigma', personal = None, personalPrime = None, weightsRange = (-1,1), biasRange = (-1,1)):
        #TODO: Aggiungere possibilità di inizializzare weights e bias in un certo range
        """
        Creates the Multi Layer Perceptron. Creates a shallow network with no hidden layer. Initializes
        randomly the weights and bias.

        :param input_dim: input dimension
        :param output_dim: output dimension
        :param f_act: activation function of the layer -> 'sigma', 'relu', 'identity, 'personal'
        :param personal: if f_act is 'personal' this is a function
        :param personalPrime: derivative of personal function
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function_list = [] #List of activation functions (one per layer). Every element of the list is a tuple, first item is the activation function, second item is its derivative
        self.num_layers = 0 #start: zero layers
        self.layers = []
        biasTemp = np.array([np.random.uniform(biasRange[0], biasRange[1]) for i in range(output_dim)])
        weightsTemp = np.array([np.random.uniform(weightsRange[0], weightsRange[1]) for i in range(output_dim*input_dim)]).reshape(output_dim, input_dim)
        self.weights = [np.hstack((np.array([biasTemp]).T, weightsTemp))] #Bias column vector and weights dimensionality is outputDim * inputDim+1
        self.curr_layer_dim = self.input_dim
        self.__add_activation_function(f_act, personal, personalPrime)

    def add_layer(self, dim, f_act, personal = None, personalPrime = None, weightsRange = (-1,1), biasRange = (-1,1), leakyPar = 0):
        #TODO: Aggiungere possibilità di inizializzare weights e bias in un certo range
        """
        Description: Adds a new fully connected layer of dimension dim between the output layer and the layer before it.
        The activation function is applied to the whole layer.

        :param dim: dimension of the layer to add
        :param f_act: activation function of the layer -> 'sigma', 'relu', 'identity', 'personal'
        :param personal: if f_act is 'personal' this is a function
        :param personalPrime: derivative of personal function
        :param leaky parameter to be applied if leaky relu
        :return: Nothing
        """
        self.__add_activation_function(f_act, personal, personalPrime)
        #W = np.random.randn(dim, self.curr_layer_dim+1)

        biasTemp = np.array([np.random.uniform(biasRange[0], biasRange[1]) for i in range(dim)])
        weightsTemp = np.array([np.random.uniform(weightsRange[0], weightsRange[1]) for i in range(dim*self.curr_layer_dim)]).reshape(dim, self.curr_layer_dim)
        self.weights[self.num_layers] = np.hstack((np.array([biasTemp]).T, weightsTemp)) #Bias column vector and weights

        #self.weights[self.num_layers] = W
        #W1 = np.random.randn(self.output_dim, dim+1)


        biasTemp = np.array([np.random.uniform(biasRange[0], biasRange[1]) for i in range(self.output_dim)])
        weightsTemp = np.array([np.random.uniform(weightsRange[0], weightsRange[1]) for i in range(self.output_dim*dim)]).reshape(self.output_dim, dim)
        self.weights.append(np.hstack((np.array([biasTemp]).T, weightsTemp)))#Bias column vector and weights dimensionality is outputDim * inputDim+1


        #self.weights.append(W1)
        self.num_layers += 1
        self.curr_layer_dim = dim
        self.currentError = 0

    def __add_activation_function(self, f_act, personal = None, personalPrime = None):
        """
        Private method. Not to be called by the user as it may (and surely) break the Multi Layer Perceptron
        Adds a tuple to the activation function list. In the first position of the tuple will be added f_act (or personal), in the
        second position its derivative.

        :param f_act: activation function of the layer -> 'sigma', 'relu', 'personal'
        :param personal: if f_act is 'personal' this is a function
        :param personalPrime: derivative of personal function
        :return:
        """
        if f_act=='sigma':
            self.activation_function_list.append((self.sigmoid, self.sigmoidPrime))
        elif f_act=='relu':
            self.activation_function_list.append((self.relu, self.reluPrime))
        elif f_act=='identity':
            self.activation_function_list.append((self.identityFunction, self.identityFunctionPrime))
        elif f_act=='personal':
            self.activation_function_list.append((personal, personalPrime))
        else: #Default: sigmoid
            self.activation_function_list.append((self.sigmoid, self.sigmoidPrime))

    def learn(self, epochs, num, tr, loss_fun='mse', eta=0.5, regularizationLambda = 0, momentum_term = 0, personal_loss_fun=None, personal_loss_fun_prime=None, verbose = False):
        """
        :param tr: A TR object. This will allow to make operations on the data set.
        :param epochs: Number of epochs
        :param num: A number between 1 and the training set size. This indicates the batch size. (1 = online, tr.training_number = batch, in between is minibatch)
        :param loss_fun: The loss function can be 'mse' for mean square error or 'personal'. Default is 'mse'
        :param eta: The learning rate. Default is 0.5
        :param personal_loss_fun: The personal loss function, if loss_fun = 'personal'
        :param personal_loss_fun_prime: The derivative of the loss function, must be specified if loss_fun is 'personal'
        :param regularizationLambda is the regularization term
        :return:
        """
        gradient = []
        i = len(self.weights)-1
        while i >=0:
            gradient.append(np.zeros((self.weights[i].shape[0], self.weights[i].shape[1]))) #Initiate gradient matrices with right dimensions
            i -= 1
        #Repeat epochs times the training process.
        for i in range(epochs):
            for j in range(num):
                x = tr.X[j, :]
                y = tr.Y[j, :]
                g = self.backprop(x,y)
                for k in range(len(g)):
                    gradient[k] += g[k]/num #Add a fraction of the gradient (avg over all the gradients calculated for each training example)
            #Update the weights
            gradient = [eta*x for x in gradient]
            t = len(gradient)-1
            for k in range(len(gradient)):
                self.weights[k] =  self.weights[k] - gradient[t] - regularizationLambda*self.weights[k]
                t -= 1
            if verbose:
                print("Epoch:", str(i)," Current error: ", self.errorCalculator(num, True))

    def forwardprop(self, x):
        """
        Forward propagates the error, returning A and O
        :param x: The example to forward propagate in the network
        :return: A = list of activations (w*x), O = list of outputs (f(w*x))
        """
        A = []
        O = [x]
        for w, af in zip(self.weights, self.activation_function_list):
            x = np.insert(x, 0, 1) #Inser 1 on top of vector because bias is inside w
            a = w.dot(x)
            A.append(a)
            x = af[0](a) #Apply activation function to neurons activation
            O.append(x)
        return A, O

    def backprop(self, x, y, loss_fun = 'mse', personal_loss_fun = None, personal_loss_fun_prime=None):
        """
        Forward propagates x into the Neural network, calculates the Loss, backward propagates the error calculating the gradient
        which is then returned.
        :param x: Training example to be fed into the network
        :param y: The target value
        :param loss_fun: The loss function
        :param personal_loss_fun: The personal loss function, if loss_fun = 'personal'
        :param personal_loss_fun_prime: The derivative of the personal lostt function
        :return: The gradient for the example x
        """
        A = [] #List that stores the activations of the neuorns. Position k = layer k, value i = activation of perceptron i. A[k][i] = w.dot(x)
        O = [x] #List that stores the outputs of the neurons. Position k = layer k, value i = output of perceptron i. A[k][i] = f(w.dot(x))

        #Forward Propagation
        A, O = self.forwardprop(x)

        #Backwards Propagation
        E = 0 #Error
        if loss_fun == 'mse':
            E = self.se(A[-1], y)
            Ep = self.se(A[-1],y, True)
        else:
            E = personal_loss_fun(A[-1], y)
            Ep = personal_loss_fun_prime(A[-1], y)

        self.currentError += E
        #Output layer
        delta = Ep * self.activation_function_list[-1][1](A[-1]) #Ep * derivative of activation = Ep * (y - y_target)
        nablaE = [np.zeros((self.weights[-1].shape[0], self.weights[-1].shape[1]))] #Initialization of nablaE as list of matrices
        nablaE[0] = np.hstack((np.array([delta]).T, np.outer(delta, O[len(O)-2]))) #Calculate the gradient wrt output layer

        #Hidden layers
        k = len(self.weights)-2
        while k >= 0:
            newDelta = self.activation_function_list[k][1](A[k])*delta.dot(self.weights[k+1][:,1:self.weights[k+1].shape[1]]) #fprime(a) at layer k multiplied my the Dot product between delta of layer k+1 and weights connecting k and k+1 (backpropagation) (basically weighted sum of error contribution from next layer), (Is row vector * matrix = matrix)
            tempE = np.outer(newDelta, O[k]) #Delta * output - column vector * row vector -> gives matrix
            nablaE.append(np.hstack((np.array([newDelta]).T,np.outer(newDelta, O[k]))))
            delta = newDelta
            k -= 1

        return nablaE

    def errorCalculator(self, num=1, reset=False):
        """
        Function to handle the error
        :param num: The number of training example seen since last reset. The error will be divided by num to calculate avg. If avg is not wanted, num=1 (default) will return just the sum of error
        :param reset: If True resets the self.currentError to 0
        :return: Returns self.currentError / num
        """
        t = self.currentError / num
        if reset:
            self.currentError = 0
        return t

    def predict(self, x):
        """
        Runs the Multi Layer Perceptron with input x
        :param x: Vector to be predicted
        :return: Prediction for the vector x
        """
        i = 0
        for w in self.weights: #TODO: forse si può fare in maniera più elegante (cioe x*w_1*w_2*w_3*...*w_m)
            x = np.insert(x, 0, 1)
            x = self.activation_function_list[i][0](w.dot(x))
            i += 1

        return x

    def relu(self, x):
        """
        ReLu activation function defined as max(0, x)
        :param x: Vector
        :return: Max(0, x)
        """
        return np.maximum(0,x)

    def reluPrime(self,x):
        """
        Derivative of the ReLu activation function defined as 1 if max(0, x) > 0, 0 otherwise
        :param x: Vector
        :return: 1 if Max(0, x) > 0, 0 otherwhise
        """
        return 1 * (np.maximum(0,x) > 0)

    def sigmoid(self,x):
        """
        Sigmoid activation function defined as 1/(1 + e^(-x))
        :param x: Vector
        :return: 1/(1 + e^(-x))
        """
        return 1 / (1 + np.exp(-x))

    def sigmoidPrime(self,x):
        """
        Derivative of the Sigmoid activation function defined as sigmoid(x) * (1 - sigmoid(x))
        :param x: Vector
        :return: sigmoid(x) * (1 - sigmoid(x))
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def identityFunction(self, x):
        """
        Returns x
        :param x: Vector to calculate the function on
        :return: x
        """
        return x

    def identityFunctionPrime(self, x):
        """
        Returns 1
        :param x: Vector to calculate the function on
        :return:
        """
        return np.ones(x.shape[0])

    def se(self, y, y_target, derivative = False):
        """
        Square error
        :param y_target: The underlying truth
        :param y: The calculated value
        :param derivative: Default is false. If true returns the derivative of mse, otherwise returns mse
        :return: if derivative returns (1/2)*(y_target - y)**2 otherwise returns (y_target - y)
        """
        if derivative:
            return (y - y_target)
        else:
            return (1/2)*(y - y_target)**2


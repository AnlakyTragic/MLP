import numpy as np
import random

class TR:
    """
    Permette di creare un data set, suddividerlo in training, validation e test set
    """
    def __init__(self, X, Y):
        """
        :param X: Training examples in matrix form
        :param Y: Target values in matrix form
        Esempio: Per ogni riga di X c'è un esempio, per ogni riga di Y ci sono i target value per la
        corrispondente riga di X
            X         Y
        [0.1 3 0.5] [3 7]
        [1.1 2 1.9] [2 9]
        """
        if X.ndim == 0 or Y.ndim == 0:
            raise Exception("Invalid dimensions")

        self.X = X
        self.Y = Y

    def subdivide(self, t, v, s, shuffle = True):
        """
        t + v + s = 1
        :param t: 0<=t<=1 Percentage of data set to be assigned to the Training set
        :param v: 0<=v<=1 Percentage of data set to be assigned to the Validation set
        :param s: 0<=s<=1 Percentage of data set to be assigned to the Test set
        :return:
        """
        if t + v + s != 1:
            raise Exception

        tot = self.X.shape[0]
        self.training_number = int(t * tot)
        self.validation_number = int(v * tot)
        self.test_number = int(s*tot)

        leftovers = tot - (self.training_number + self.validation_number + self.test_number)
        if leftovers > 0:
            self.test_number += leftovers

        self.data_numbers = [self.training_number, self.validation_number, self.test_number]

        if shuffle:
            self.shuffle()

    def shuffle(self):
        """
        Shuffles the data set. Uses a random state to shuffle them in the same exact way
        Shuffles the different subsets keeping them separated.
        :return: Nothing
        """
        t = []
        t[:] = self.data_numbers
        t.insert(0,0)
        for i in range(0, len(t)-1):
            t[i+1] += t[i]
            random_state = np.random.get_state()
            np.random.shuffle(self.X[t[i]:t[i+1],])
            np.random.set_state(random_state)
            np.random.shuffle(self.Y[t[i]:t[i+1],])
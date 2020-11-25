from MLP import MLP
from TR import TR
import numpy as np
import matplotlib.pyplot as plt

input_dim = 1
output_dim = 1
a = MLP(input_dim, output_dim, weightsRange=(-2, 2), biasRange=(-1,1))


def personal(x):
    return x


def personalP(x):
    return np.ones(x.shape[0])


def leakyRelu(x):
    return np.where(x > 0, x, x * 0.01)


def leakyReluP(x):
    return np.where(x > 0, 1, 0.01)


a.add_layer(20, 'personal', leakyRelu, leakyReluP, weightsRange=(-2, 2), biasRange=(-1, 1))

a.add_layer(20, 'personal', leakyRelu, leakyReluP, weightsRange=(-1, 1), biasRange=(-1, 1))
a.add_layer(15, 'personal', leakyRelu, leakyReluP, weightsRange=(-2, 2), biasRange=(-1, 1))

a.add_layer(10, 'personal', leakyRelu, leakyReluP, weightsRange=(-1, 1), biasRange=(-1, 1))
print(a.weights[0])
# X = np.array([[0],[1], [2],[3], [4],[5], [6],[7], [8],[9], [10]])
# Y = np.array([[0],[2], [4], [6], [8], [10], [12], [14], [16], [18],[20]])

X = np.linspace(0, 5, 100)
X = np.array([X]).T
Y = np.sin(X) + 3 + np.random.random()/10

print(X.shape)
print(Y.shape)

plt.plot(X, Y)

tr = TR(X, Y)
tr.subdivide(1, 0, 0)
a.learn(500, tr.training_number, tr, eta=0.01, verbose=True, regularizationLambda = 0.0001)

x_calc = []
y_calc = []
X = np.sort(X)
for x in X:
    x_calc.append(x[0])
    y_calc.append(a.predict([x])[0])
"""
A = np.hstack((np.array([x_calc]).T, np.array([y_calc]).T))
A = A[np.lexsort(A.T[::-1])]
print(A)
x_calc = A[:, 0]
y_calc = A[:, 1]
"""
print(a.weights[0])
plt.plot(x_calc, y_calc, 'o')
plt.show()
